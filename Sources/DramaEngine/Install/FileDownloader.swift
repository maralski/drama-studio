// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import CryptoKit
import Foundation

/// One file to fetch and the hash it must have.
package struct DownloadTarget: Sendable, Equatable {
    package var url: URL
    package var destination: URL
    /// Nil when only the server knows; the response's length is used.
    package var size: Int64?
    /// Lowercase hex.
    package var sha256: String

    package init(url: URL, destination: URL, size: Int64?, sha256: String) {
        self.url = url
        self.destination = destination
        self.size = size
        self.sha256 = sha256
    }
}

package enum DownloadError: Error, Equatable, CustomStringConvertible {
    case httpStatus(Int, URL)
    /// The server answered a resume request with a range that does not start where the partial file ends.
    case badRange(URL, String)
    case sizeMismatch(file: String, expected: Int64, actual: Int64)
    case checksumMismatch(file: String, expected: String, actual: String)
    case transport(String)

    package var description: String {
        switch self {
        case .httpStatus(let status, let url): "The server answered \(status) for \(url.lastPathComponent)."
        case .badRange(let url, let range): "The server resumed \(url.lastPathComponent) at the wrong place (\(range))."
        case .sizeMismatch(let file, let expected, let actual): "\(file) is \(actual) bytes; expected \(expected)."
        case .checksumMismatch(let file, let expected, let actual): "\(file) failed its checksum: sha256 \(actual.prefix(12))…, expected \(expected.prefix(12))…."
        case .transport(let message): message
        }
    }
}

/// Fetches one file into `<destination>.partial`, resuming a partial file
/// with an HTTP Range request, and renames it into place only after its
/// size and sha256 match. A checksum or size mismatch deletes the partial
/// file; cancellation and network failures keep it for the next attempt,
/// including after the app quits.
package final class FileDownloader: Sendable {
    /// Called with the bytes of the file on disk so far and the file's size when known.
    package typealias Progress = @Sendable (_ bytes: Int64, _ total: Int64?) -> Void

    private let makeConfiguration: @Sendable () -> URLSessionConfiguration

    /// `configuration` is called once per download; tests pass one with a stub `URLProtocol`.
    package init(configuration: @escaping @Sendable () -> URLSessionConfiguration = { .default }) {
        makeConfiguration = configuration
    }

    package static func partialURL(for destination: URL) -> URL {
        destination.deletingLastPathComponent().appending(path: destination.lastPathComponent + ".partial")
    }

    package func download(_ target: DownloadTarget, progress: @escaping Progress = { _, _ in }) async throws {
        let fm = FileManager.default
        let partial = Self.partialURL(for: target.destination)
        try fm.createDirectory(at: partial.deletingLastPathComponent(), withIntermediateDirectories: true)
        if !fm.fileExists(atPath: partial.path) { fm.createFile(atPath: partial.path, contents: nil) }
        var offset = Int64(DramaSetup.fileSize(partial))
        if let size = target.size, offset > size {
            try Data().write(to: partial)
            offset = 0
        }
        var hasher = try Self.hash(partial, upTo: offset)
        progress(offset, target.size)

        var total = target.size
        if offset != target.size {
            let transfer = try Transfer(target: target, partial: partial, offset: offset, hasher: hasher, progress: progress)
            do {
                try await transfer.run(configuration: makeConfiguration())
            } catch DownloadError.httpStatus(416, let url) {
                // The partial file is as long as the server's copy or longer; start over.
                try? fm.removeItem(at: partial)
                throw DownloadError.httpStatus(416, url)
            }
            hasher = transfer.hasher
            total = target.size ?? transfer.serverTotal
        }

        let name = target.destination.lastPathComponent
        let written = Int64(DramaSetup.fileSize(partial))
        if let total, written != total {
            if written > total { try? fm.removeItem(at: partial) }
            throw DownloadError.sizeMismatch(file: name, expected: total, actual: written)
        }
        let digest = hasher.finalize().map { String(format: "%02x", $0) }.joined()
        guard digest == target.sha256.lowercased() else {
            try? fm.removeItem(at: partial)
            throw DownloadError.checksumMismatch(file: name, expected: target.sha256, actual: digest)
        }
        guard rename(partial.path, target.destination.path) == 0 else {
            throw DownloadError.transport("Couldn't move \(name) into place: \(String(cString: strerror(errno))).")
        }
    }

    /// The hash of a partial file's first `count` bytes, so a resumed download
    /// verifies the whole file without fetching the start again.
    private static func hash(_ url: URL, upTo count: Int64) throws -> SHA256 {
        var hasher = SHA256()
        guard count > 0 else { return hasher }
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var left = count
        while left > 0, let chunk = try handle.read(upToCount: Int(min(left, 8 << 20))), !chunk.isEmpty {
            hasher.update(data: chunk)
            left -= Int64(chunk.count)
        }
        return hasher
    }
}

/// One HTTP transfer appending to a partial file. Unchecked because the
/// session's serial delegate queue is the only thread that touches the
/// mutable state after `run` starts the task, and `run` reads it only after
/// the task completes.
private final class Transfer: NSObject, URLSessionDataDelegate, @unchecked Sendable {
    let target: DownloadTarget
    let progress: FileDownloader.Progress
    private let handle: FileHandle
    private var offset: Int64
    private var written: Int64 = 0
    private(set) var hasher: SHA256
    private(set) var serverTotal: Int64?
    private var failure: Error?
    private var continuation: CheckedContinuation<Void, Error>?

    init(target: DownloadTarget, partial: URL, offset: Int64, hasher: SHA256, progress: @escaping FileDownloader.Progress) throws {
        self.target = target
        self.offset = offset
        self.hasher = hasher
        self.progress = progress
        handle = try FileHandle(forWritingTo: partial)
        try handle.truncate(atOffset: UInt64(offset))
        try handle.seekToEnd()
    }

    func run(configuration: URLSessionConfiguration) async throws {
        configuration.urlCache = nil
        configuration.requestCachePolicy = .reloadIgnoringLocalCacheData
        let queue = OperationQueue()
        queue.maxConcurrentOperationCount = 1
        let session = URLSession(configuration: configuration, delegate: self, delegateQueue: queue)
        defer { session.finishTasksAndInvalidate() }
        let task = session.dataTask(with: request(for: target.url))
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
                queue.addOperation {
                    self.continuation = continuation
                    task.resume()
                }
            }
        } onCancel: {
            task.cancel()
        }
    }

    private func request(for url: URL) -> URLRequest {
        var request = URLRequest(url: url)
        if offset > 0 { request.setValue("bytes=\(offset)-", forHTTPHeaderField: "Range") }
        return request
    }

    private func fail(_ error: Error, _ task: URLSessionTask) {
        if failure == nil { failure = error }
        task.cancel()
    }

    /// Hugging Face and GitHub redirect to a CDN; the Range header must survive the hop.
    func urlSession(_ session: URLSession, task: URLSessionTask, willPerformHTTPRedirection response: HTTPURLResponse,
                    newRequest: URLRequest, completionHandler: @escaping (URLRequest?) -> Void) {
        var next = newRequest
        if offset > 0 { next.setValue("bytes=\(offset)-", forHTTPHeaderField: "Range") }
        completionHandler(next)
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive response: URLResponse,
                    completionHandler: @escaping (URLSession.ResponseDisposition) -> Void) {
        guard let http = response as? HTTPURLResponse else {
            fail(DownloadError.transport("\(target.url.lastPathComponent): the answer was not HTTP."), dataTask)
            return completionHandler(.cancel)
        }
        switch http.statusCode {
        case 206:
            let range = http.value(forHTTPHeaderField: "Content-Range") ?? ""
            guard let (start, total) = Self.parseContentRange(range), start == offset else {
                fail(DownloadError.badRange(target.url, range), dataTask)
                return completionHandler(.cancel)
            }
            serverTotal = total
        case 200:
            if offset > 0 {
                do {
                    try handle.truncate(atOffset: 0)
                } catch {
                    fail(error, dataTask)
                    return completionHandler(.cancel)
                }
                offset = 0
                hasher = SHA256()
            }
            serverTotal = http.expectedContentLength >= 0 ? http.expectedContentLength : nil
        default:
            fail(DownloadError.httpStatus(http.statusCode, target.url), dataTask)
            return completionHandler(.cancel)
        }
        completionHandler(.allow)
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        guard failure == nil else { return }
        do {
            try handle.write(contentsOf: data)
        } catch {
            return fail(error, dataTask)
        }
        hasher.update(data: data)
        written += Int64(data.count)
        progress(offset + written, target.size ?? serverTotal)
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        try? handle.synchronize()
        try? handle.close()
        let continuation = continuation
        self.continuation = nil
        if let failure {
            continuation?.resume(throwing: failure)
        } else if let error = error as? URLError, error.code == .cancelled {
            continuation?.resume(throwing: CancellationError())
        } else if let error {
            continuation?.resume(throwing: DownloadError.transport("\(target.url.lastPathComponent): \(error.localizedDescription)"))
        } else {
            continuation?.resume()
        }
    }

    /// `bytes 100-199/1000` to (100, 1000); the total may be `*`.
    static func parseContentRange(_ value: String) -> (start: Int64, total: Int64?)? {
        let parts = value.split(whereSeparator: { " -/".contains($0) })
        guard parts.count == 4, parts[0] == "bytes", let start = Int64(parts[1]) else { return nil }
        return (start, Int64(parts[3]))
    }
}

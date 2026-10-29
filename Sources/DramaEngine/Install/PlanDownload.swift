// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation

/// How far a download has come: the whole and the file under way.
package struct DownloadProgress: Sendable, Equatable {
    package var totalBytes: Int64
    /// Bytes on disk toward `totalBytes`: finished files plus the partial file.
    package var doneBytes: Int64
    package var file: String?
    package var fileBytes: Int64 = 0
    package var fileSize: Int64 = 0
    package var filesDone: Int
    package var fileCount: Int

    package init(totalBytes: Int64, doneBytes: Int64, file: String? = nil, fileBytes: Int64 = 0, fileSize: Int64 = 0, filesDone: Int, fileCount: Int) {
        self.totalBytes = totalBytes
        self.doneBytes = doneBytes
        self.file = file
        self.fileBytes = fileBytes
        self.fileSize = fileSize
        self.filesDone = filesDone
        self.fileCount = fileCount
    }

    package var fraction: Double { totalBytes > 0 ? min(1, Double(doneBytes) / Double(totalBytes)) : 0 }
    package var fileFraction: Double { fileSize > 0 ? min(1, Double(fileBytes) / Double(fileSize)) : 0 }
}

package enum PlanDownloadError: Error, CustomStringConvertible {
    /// The file failed `attempts` times; `last` is the final failure.
    case fileFailed(path: String, attempts: Int, last: Error)

    package var description: String {
        switch self {
        case .fileFailed(let path, let attempts, let last): "\(path) failed \(attempts) times. \(last)"
        }
    }
}

/// Downloads every file of a plan that is not yet in place, one at a time,
/// retrying a failed file with backoff. Rerunning after a crash, a pause or
/// a quit resumes from the partial files and skips the finished ones.
package struct PlanDownload: Sendable {
    package var plan: ModelDownloadPlan
    package var directory: URL
    package var downloader: FileDownloader
    /// The waits before each retry; three retries by default.
    package var retryDelays: [Duration] = [.seconds(2), .seconds(8), .seconds(30)]
    package var sleep: @Sendable (Duration) async throws -> Void = { try await Task.sleep(for: $0) }
    /// The shortest gap between two progress reports, other than at file boundaries.
    package var reportInterval: Duration = .milliseconds(100)

    package init(plan: ModelDownloadPlan, directory: URL, downloader: FileDownloader) {
        self.plan = plan
        self.directory = directory
        self.downloader = downloader
    }

    package var progress: DownloadProgress {
        DownloadProgress(totalBytes: plan.totalBytes, doneBytes: plan.bytesOnDisk(in: directory),
                         filesDone: plan.files.count - plan.missingFiles(in: directory).count, fileCount: plan.files.count)
    }

    package func run(progress report: @escaping @Sendable (DownloadProgress) -> Void) async throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        var finished = plan.files.filter { plan.isInPlace($0, in: directory) }.reduce(Int64(0)) { $0 + $1.size }
        var filesDone = plan.files.count - plan.missingFiles(in: directory).count
        let gate = ProgressThrottle(interval: reportInterval)
        for file in plan.missingFiles(in: directory) {
            let base = DownloadProgress(totalBytes: plan.totalBytes, doneBytes: finished, file: file.path, fileSize: file.size,
                                        filesDone: filesDone, fileCount: plan.files.count)
            report(base)
            let target = DownloadTarget(url: plan.url(for: file), destination: directory.appending(path: file.path), size: file.size, sha256: file.sha256)
            try await withRetries(file.path) {
                try await downloader.download(target) { bytes, _ in
                    guard gate.allows() else { return }
                    var now = base
                    now.fileBytes = bytes
                    now.doneBytes = base.doneBytes + bytes
                    report(now)
                }
            }
            finished += file.size
            filesDone += 1
        }
        report(DownloadProgress(totalBytes: plan.totalBytes, doneBytes: finished, filesDone: filesDone, fileCount: plan.files.count))
    }

    package func withRetries(_ name: String, _ attempt: () async throws -> Void) async throws {
        for retry in 0...retryDelays.count {
            do {
                return try await attempt()
            } catch is CancellationError {
                throw CancellationError()
            } catch {
                guard retry < retryDelays.count else { throw PlanDownloadError.fileFailed(path: name, attempts: retry + 1, last: error) }
                try await sleep(retryDelays[retry])
            }
        }
    }
}

/// Lets a progress report through at most once per interval.
package final class ProgressThrottle: @unchecked Sendable {
    private let interval: Duration
    private let lock = NSLock()
    private var last: ContinuousClock.Instant?

    package init(interval: Duration = .milliseconds(100)) { self.interval = interval }

    package func allows() -> Bool {
        let now = ContinuousClock.now
        return lock.withLock {
            if let last, now - last < interval { return false }
            last = now
            return true
        }
    }
}

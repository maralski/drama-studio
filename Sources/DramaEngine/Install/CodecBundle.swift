// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation

package enum CodecBundleError: Error, Equatable, CustomStringConvertible {
    case unzipFailed(status: Int32, output: String)
    case missingModels([String])

    package var description: String {
        switch self {
        case .unzipFailed(let status, let output): "Unzipping the codec models failed (ditto exited \(status)). \(output)"
        case .missingModels(let names): "The codec download is missing \(names.joined(separator: ", "))."
        }
    }
}

/// The release asset that holds the three Core ML codec models, so no one
/// needs Python to build them: a zip of `DramaCodecEncoder.mlpackage`,
/// `DramaCodecDecoder.mlpackage` and `DramaCodecProbe.mlpackage` with the
/// license files. `tools/make_release.sh` writes its URL and sha256 into the
/// app's Info.plist; a build without them has no bundle.
package struct CodecBundle: Sendable, Equatable {
    package static let urlKey = "DSCodecBundleURL"
    package static let sha256Key = "DSCodecBundleSHA256"
    /// The directories the zip must hold.
    package static let packageNames = DramaCodec.modelNames.map { "\($0).mlpackage" }

    package var url: URL
    package var sha256: String

    package init(url: URL, sha256: String) {
        self.url = url
        self.sha256 = sha256
    }

    /// Nil unless both keys are present, the URL is https and the hash is 64 hex digits.
    package init?(info: [String: Any]?) {
        guard let text = info?[Self.urlKey] as? String, let url = URL(string: text), url.scheme == "https",
              let sha = (info?[Self.sha256Key] as? String)?.lowercased(),
              sha.count == 64, sha.allSatisfy(\.isHexDigit) else { return nil }
        self.init(url: url, sha256: sha)
    }

    /// Where the zip goes in `downloads`; it is there only once verified.
    package func archive(in downloads: URL) -> URL { downloads.appending(path: url.lastPathComponent) }

    /// Fetches and verifies the zip unless it is already in place, and returns its path.
    package func download(into downloads: URL, with downloader: FileDownloader, progress: @escaping FileDownloader.Progress) async throws -> URL {
        let archive = archive(in: downloads)
        if !FileManager.default.fileExists(atPath: archive.path) {
            try await downloader.download(DownloadTarget(url: url, destination: archive, size: nil, sha256: sha256), progress: progress)
        }
        return archive
    }

    /// Unzips `archive` with `ditto` into a staging folder beside
    /// `codecDirectory`, checks that the three packages are there, and moves
    /// each into `codecDirectory` with a rename, replacing an older package
    /// and its compiled `.mlmodelc`. Leftover staging folders from an
    /// interrupted install are removed first, so running it again converges.
    package static func install(archive: URL, into codecDirectory: URL) throws {
        let fm = FileManager.default
        let parent = codecDirectory.deletingLastPathComponent()
        try fm.createDirectory(at: codecDirectory, withIntermediateDirectories: true)
        for stale in (try? fm.contentsOfDirectory(atPath: parent.path)) ?? [] where stale.hasPrefix(".codec-staging-") {
            try? fm.removeItem(at: parent.appending(path: stale))
        }
        let staging = parent.appending(path: ".codec-staging-\(UUID().uuidString)", directoryHint: .isDirectory)
        defer { try? fm.removeItem(at: staging) }

        let ditto = Process()
        let output = Pipe()
        ditto.executableURL = URL(filePath: "/usr/bin/ditto")
        ditto.arguments = ["-x", "-k", archive.path, staging.path]
        ditto.standardOutput = output
        ditto.standardError = output
        try ditto.run()
        let printed = output.fileHandleForReading.readDataToEndOfFile()
        ditto.waitUntilExit()
        guard ditto.terminationStatus == 0 else {
            throw CodecBundleError.unzipFailed(status: ditto.terminationStatus, output: String(decoding: printed, as: UTF8.self))
        }

        var isDirectory: ObjCBool = false
        let missing = packageNames.filter { !(fm.fileExists(atPath: staging.appending(path: $0).path, isDirectory: &isDirectory) && isDirectory.boolValue) }
        guard missing.isEmpty else { throw CodecBundleError.missingModels(missing) }

        for name in packageNames {
            let destination = codecDirectory.appending(path: name)
            try? fm.removeItem(at: destination.deletingPathExtension().appendingPathExtension("mlmodelc"))
            if fm.fileExists(atPath: destination.path) { try fm.removeItem(at: destination) }
            try fm.moveItem(at: staging.appending(path: name), to: destination)
        }
    }
}

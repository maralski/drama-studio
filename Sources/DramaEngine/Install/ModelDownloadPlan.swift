// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation

/// The files of one pinned Hugging Face revision, with the size and sha256
/// each must have. The values are committed with the app
/// (`tools/pin_model_plan.py` writes ``drama3``), so a download is checked
/// against hashes the app did not fetch.
package struct ModelDownloadPlan: Sendable, Equatable {
    package struct File: Sendable, Equatable {
        /// The path inside the repository and inside the download folder.
        package var path: String
        package var size: Int64
        /// Lowercase hex.
        package var sha256: String

        package init(path: String, size: Int64, sha256: String) {
            self.path = path
            self.size = size
            self.sha256 = sha256
        }
    }

    package var repo: String
    package var revision: String
    package var files: [File]

    package init(repo: String, revision: String, files: [File]) {
        self.repo = repo
        self.revision = revision
        self.files = files
    }

    package var totalBytes: Int64 { files.reduce(0) { $0 + $1.size } }

    package func url(for file: File) -> URL {
        URL(string: "https://huggingface.co/\(repo)/resolve/\(revision)/\(file.path)")!
    }

    /// The folder name a download of this plan goes into, such as `drama-3-e0a9218`.
    package var folderName: String {
        "\(repo.split(separator: "/").last ?? "model")-\(revision.prefix(7))"
    }

    /// The file is in place when it exists with its planned size. Only a
    /// verified download is renamed to that path, so the size stands for
    /// the hash without reading 40 GB at every launch.
    package func isInPlace(_ file: File, in directory: URL) -> Bool {
        DramaSetup.fileSize(directory.appending(path: file.path)) == UInt64(file.size)
    }

    package func missingFiles(in directory: URL) -> [File] {
        files.filter { !isInPlace($0, in: directory) }
    }

    /// Bytes already on disk toward the plan: finished files plus partial downloads.
    package func bytesOnDisk(in directory: URL) -> Int64 {
        files.reduce(0) { sum, file in
            if isInPlace(file, in: directory) { return sum + file.size }
            let partial = Int64(DramaSetup.fileSize(FileDownloader.partialURL(for: directory.appending(path: file.path))))
            return sum + min(partial, file.size)
        }
    }
}

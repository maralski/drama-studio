// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import DramaEngine
import Foundation

/// Where Drama Studio keeps its downloads and settings.
///
/// `DRAMA_STUDIO_HOME=<folder>` moves the model download, the codec models,
/// the kernel tuning and the settings (a `defaults.plist` there) into that
/// folder and stops the app from using the Hugging Face cache, so a first run
/// can be tried without touching the real ones.
enum AppHome {
    static let override: URL? = ProcessInfo.processInfo.environment["DRAMA_STUDIO_HOME"]
        .flatMap { $0.isEmpty ? nil : URL(filePath: $0, directoryHint: .isDirectory) }

    static var support: URL {
        override ?? URL.applicationSupportDirectory.appending(path: "DramaStudio", directoryHint: .isDirectory)
    }

    /// The folder the setup window downloads the pinned Drama 3 revision into.
    static var models: URL {
        support.appending(path: "Models/\(ModelDownloadPlan.drama3.folderName)", directoryHint: .isDirectory)
    }

    /// Where the codec zip waits between its download and its unpacking.
    static var downloads: URL { support.appending(path: "Downloads", directoryHint: .isDirectory) }

    static var codec: URL { override?.appending(path: "Codec", directoryHint: .isDirectory) ?? DramaCodec.defaultDirectory }

    static var tuning: URL { override?.appending(path: "Tuning", directoryHint: .isDirectory) ?? DramaSynthesizer.defaultTuningDirectory }

    static var usesHuggingFaceCache: Bool { override == nil }

    /// A suite named by an absolute path keeps its plist at that path.
    static let defaults: UserDefaults = override.flatMap { UserDefaults(suiteName: $0.appending(path: "defaults.plist").path) } ?? .standard
}

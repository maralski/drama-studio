// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation

/// Where one engine's checkpoint comes from and what setup checks for it.
/// ``SpeechEngineID/install`` is the one table: setup, the download, the
/// Hugging Face cache lookup and the API's model ids all read it.
public struct EngineInstall: Sendable {
    /// The Hugging Face repository that holds the checkpoint, which the API
    /// also reports as the engine's model id.
    public let repo: String
    /// What ``DramaSetup`` checks for the engine, in the order the engine uses them.
    public let components: [SetupItem.Component]
    /// The pinned revision the app downloads and verifies.
    package let plan: ModelDownloadPlan

    /// The shell command that downloads the checkpoint into the Hugging Face cache.
    public var downloadCommand: String { "uvx --from huggingface_hub hf download \(repo)" }
}

public extension SpeechEngineID {
    var install: EngineInstall {
        switch self {
        case .drama3:
            EngineInstall(repo: "fishaudio/drama-3", components: [.weights, .tokenizer, .codecCheckpoint, .codecModels, .tuning, .wordRecognizer],
                          plan: .drama3)
        case .ming:
            EngineInstall(repo: "inclusionAI/Ming-3.0-tts-0.5B", components: [.weights, .tokenizer, .speakerEncoder, .wordRecognizer], plan: .ming)
        }
    }

    /// The engine `name` stands for: its id (`ming`), its model id
    /// (`inclusionAI/Ming-3.0-tts-0.5B`) or its name (`Ming`), in any case.
    /// Nil for anything else, such as an OpenAI model name.
    static func named(_ name: String) -> SpeechEngineID? {
        let wanted = name.trimmingCharacters(in: .whitespaces).lowercased()
        return allCases.first { [$0.rawValue, $0.install.repo, $0.capabilities.name].contains { $0.lowercased() == wanted } }
    }
}

/// The Hugging Face hub cache that `hf download` fills: `HF_HUB_CACHE`, else
/// `HF_HOME/hub`, else `~/.cache/huggingface/hub`.
public enum HuggingFaceCache {
    public static var directory: URL {
        let environment = ProcessInfo.processInfo.environment
        if let hub = environment["HF_HUB_CACHE"], !hub.isEmpty { return URL(filePath: hub, directoryHint: .isDirectory) }
        if let home = environment["HF_HOME"], !home.isEmpty { return URL(filePath: home, directoryHint: .isDirectory).appending(path: "hub") }
        return FileManager.default.homeDirectoryForCurrentUser.appending(path: ".cache/huggingface/hub", directoryHint: .isDirectory)
    }

    /// The most recently modified snapshot of `repo`; nil when there is none.
    public static func latestSnapshot(of repo: String) -> URL? {
        let snapshots = directory.appending(path: "models--" + repo.replacingOccurrences(of: "/", with: "--") + "/snapshots")
        let found = (try? FileManager.default.contentsOfDirectory(at: snapshots, includingPropertiesForKeys: [.contentModificationDateKey])) ?? []
        return found.max { modified($0) < modified($1) }
    }

    private static func modified(_ url: URL) -> Date {
        (try? url.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
    }
}

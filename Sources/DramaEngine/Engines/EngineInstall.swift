// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation

/// Where one engine's checkpoint comes from and what setup checks for it.
/// ``SpeechEngineID/install`` is the one table: setup, the download, the
/// Hugging Face cache lookup and the API's model ids all read it.
public struct EngineInstall: Sendable {
    /// The license the engine's publisher puts on its checkpoint.
    public struct License: Sendable, Equatable {
        public let name: String
        public let url: URL
        /// The license binds whoever runs the model to use restrictions, so
        /// a person must agree to them before the checkpoint downloads or speaks.
        public let requiresAgreement: Bool
    }

    /// Who publishes the checkpoint.
    public let publisher: String
    public let license: License
    /// The Hugging Face repository that holds the checkpoint, which the API
    /// also reports as the engine's model id.
    public let repo: String
    /// The least memory the engine is expected to run well in; nil when no floor is known.
    public let recommendedMemory: UInt64?
    /// What ``DramaSetup`` checks for the engine, in the order the engine uses them.
    public let components: [SetupItem.Component]
    /// The pinned revision the app downloads and verifies.
    package let plan: ModelDownloadPlan

    /// The size of the pinned checkpoint download.
    public var downloadBytes: Int64 { plan.totalBytes }

    /// The shell command that downloads the checkpoint into the Hugging Face cache.
    public var downloadCommand: String { "uvx --from huggingface_hub hf download \(repo)" }
}

public extension SpeechEngineID {
    var install: EngineInstall {
        switch self {
        case .drama3:
            EngineInstall(publisher: "Fish Audio",
                          license: .init(name: "Fish Audio Community License",
                                         url: URL(string: "https://huggingface.co/fishaudio/drama-3/blob/\(ModelDownloadPlan.drama3.revision)/LICENSE")!,
                                         requiresAgreement: true),
                          repo: "fishaudio/drama-3", recommendedMemory: 64 << 30,
                          components: [.weights, .tokenizer, .codecCheckpoint, .codecModels, .tuning, .wordRecognizer], plan: .drama3)
        case .ming:
            EngineInstall(publisher: "inclusionAI",
                          license: .init(name: "MIT License", url: URL(string: "https://huggingface.co/inclusionAI/Ming-3.0-tts-0.5B")!,
                                         requiresAgreement: false),
                          repo: "inclusionAI/Ming-3.0-tts-0.5B", recommendedMemory: nil,
                          components: [.weights, .tokenizer, .speakerEncoder, .wordRecognizer], plan: .ming)
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

    /// The newest snapshot of `repo` that `isComplete` accepts, else the
    /// newest snapshot, so setup can name what it lacks; nil when there is
    /// none. The repo folder matches in any case: `hf download` names it
    /// with the case the repo id was typed in.
    public static func latestSnapshot(of repo: String, isComplete: (URL) -> Bool = { _ in true }) -> URL? {
        let fm = FileManager.default
        let folder = "models--" + repo.replacingOccurrences(of: "/", with: "--").lowercased()
        let repos = ((try? fm.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)) ?? []).filter { $0.lastPathComponent.lowercased() == folder }
        let snapshots = repos
            .flatMap { (try? fm.contentsOfDirectory(at: $0.appending(path: "snapshots"), includingPropertiesForKeys: [.contentModificationDateKey, .isDirectoryKey])) ?? [] }
            .filter { (try? $0.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true }
            .sorted { modified($0) > modified($1) }
        return snapshots.first(where: isComplete) ?? snapshots.first
    }

    private static func modified(_ url: URL) -> Date {
        (try? url.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
    }
}

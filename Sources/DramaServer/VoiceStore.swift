// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import DramaEngine
import Foundation

/// Where the API keeps reference voices. The app passes its `VoiceLibrary`
/// so voices added over HTTP show up in the window; the CLI and tests use a
/// `VoiceDirectory`.
public protocol VoiceStore: VoiceEncodingStore {
    func allVoices() async -> [ReferenceVoice]
    /// Keeps the voice and, when given, its clip as mono 44.1 kHz samples.
    func add(_ voice: ReferenceVoice, clip: [Float]?) async throws
    /// Removes the voice and its clip.
    func remove(_ id: ReferenceVoice.ID) async throws
}

public struct VoiceDirectory: VoiceStore {
    public let url: URL
    /// The Drama 3 codec that recreates the clip of a voice saved without one.
    public let codecDirectory: URL

    public static var defaultURL: URL {
        URL.applicationSupportDirectory.appending(path: "DramaStudio/Voices", directoryHint: .isDirectory)
    }

    public init(url: URL = VoiceDirectory.defaultURL, codecDirectory: URL = DramaCodec.defaultDirectory) {
        self.url = url
        self.codecDirectory = codecDirectory
    }

    public func load() -> (voices: [ReferenceVoice], unreadable: [String]) {
        let files = (try? FileManager.default.contentsOfDirectory(at: url, includingPropertiesForKeys: nil)) ?? []
        var voices: [ReferenceVoice] = []
        var unreadable: [String] = []
        for file in files where file.pathExtension == "json" {
            do {
                voices.append(try JSONDecoder().decode(ReferenceVoice.self, from: Data(contentsOf: file)))
            } catch {
                unreadable.append(file.lastPathComponent)
            }
        }
        return (voices.sorted { $0.name.localizedStandardCompare($1.name) == .orderedAscending }, unreadable.sorted())
    }

    public func allVoices() async -> [ReferenceVoice] { load().voices }

    /// Writes the clip before the voice file, so a voice file never names a clip that is not there yet.
    public func add(_ voice: ReferenceVoice, clip: [Float]?) throws {
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        var voice = voice
        voice.clipSampleCount = clip?.count
        if let clip {
            try ReferenceAudio.write(clip, to: clipFile(for: voice.id))
        } else {
            try? FileManager.default.removeItem(at: clipFile(for: voice.id))
        }
        try JSONEncoder().encode(voice).write(to: file(for: voice.id), options: .atomic)
    }

    public func remove(_ id: ReferenceVoice.ID) throws {
        try FileManager.default.removeItem(at: file(for: id))
        if FileManager.default.fileExists(atPath: clipFile(for: id).path) { try FileManager.default.removeItem(at: clipFile(for: id)) }
    }

    /// Whether the voice's clip is on disk, so reading it needs no codec.
    public func hasClip(for id: ReferenceVoice.ID) -> Bool {
        FileManager.default.fileExists(atPath: clipFile(for: id).path)
    }

    /// The voice's clip. A voice saved before the library kept clips gets
    /// one decoded from its Drama 3 codes, kept as its wav so that happens
    /// once. Nil when the voice has neither.
    public func clip(for id: ReferenceVoice.ID) async throws -> [Float]? {
        let clip = clipFile(for: id)
        if FileManager.default.fileExists(atPath: clip.path) { return try ReferenceAudio.load(url: clip) }
        guard FileManager.default.fileExists(atPath: file(for: id).path) else { return nil }
        let voice = try JSONDecoder().decode(ReferenceVoice.self, from: Data(contentsOf: file(for: id)))
        guard let samples = try await DramaCodec.recreateClip(of: voice, directory: codecDirectory) else { return nil }
        let partial = url.appending(path: "\(id.uuidString).partial.wav")
        try ReferenceAudio.write(samples, to: partial)
        _ = try FileManager.default.replaceItemAt(clip, withItemAt: partial)
        return samples
    }

    /// Rewrites the voice file with `encoding` added; the clip is untouched.
    public func storeEncoding(_ encoding: VoiceEncoding, for id: ReferenceVoice.ID) throws {
        var voice = try JSONDecoder().decode(ReferenceVoice.self, from: Data(contentsOf: file(for: id)))
        voice.store(encoding)
        try JSONEncoder().encode(voice).write(to: file(for: id), options: .atomic)
    }

    private func file(for id: ReferenceVoice.ID) -> URL {
        url.appending(path: "\(id.uuidString).json")
    }

    private func clipFile(for id: ReferenceVoice.ID) -> URL {
        url.appending(path: "\(id.uuidString).wav")
    }
}

public enum VoiceLookup {
    public static func find(_ key: VoiceKey, in voices: [ReferenceVoice]) -> Result<ReferenceVoice, APIError> {
        if let id = UUID(uuidString: key), let voice = voices.first(where: { $0.id == id }) { return .success(voice) }
        for matches in [voices.filter { $0.name == key }, voices.filter { $0.name.caseInsensitiveCompare(key) == .orderedSame }] {
            if matches.count == 1 { return .success(matches[0]) }
            if matches.count > 1 { return .failure(.bad("More than one voice is named \"\(key)\"; pass its id.")) }
        }
        return .failure(.bad("No voice \"\(key)\" in the library. GET /v1/voices lists them."))
    }

    /// What a `voice` key names. A library voice wins over a built-in voice
    /// of the same name, so every key that named a library voice before
    /// built-in voices still names it. `engine:name`, such as `ming:US_Emma`,
    /// means the built-in one, which a library voice called `US_Emma` cannot
    /// stand in for.
    public static func resolve(_ key: VoiceKey, in voices: [ReferenceVoice]) -> Result<NamedVoice, APIError> {
        switch find(key, in: voices) {
        case .success(let voice):
            return .success(.library(voice))
        case .failure(let error):
            let ambiguous = voices.contains { $0.name.caseInsensitiveCompare(key) == .orderedSame }
            if !ambiguous, let voice = BuiltInVoice.named(key) { return .success(.builtIn(voice)) }
            if !ambiguous, let qualified = BuiltInVoice.qualifier(of: key) {
                return .failure(.bad("\(qualified.engine.capabilities.name) has no built-in voice \"\(qualified.name)\". GET /v1/voices lists them."))
            }
            return .failure(ambiguous ? error : .bad("No voice \"\(key)\" in the library or built into an engine. GET /v1/voices lists them."))
        }
    }
}

/// A voice a request names by key.
public enum NamedVoice: Sendable, Equatable {
    /// A cloned voice from the library.
    case library(ReferenceVoice)
    case builtIn(BuiltInVoice)
}

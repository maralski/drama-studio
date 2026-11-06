// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import DramaEngine
import Foundation

/// Where the API keeps reference voices. The app passes its `VoiceLibrary`
/// so voices added over HTTP show up in the window; the CLI and tests use a
/// `VoiceDirectory`.
public protocol VoiceStore: Sendable {
    func allVoices() async -> [ReferenceVoice]
    /// Keeps the voice and, when given, its clip as mono 44.1 kHz samples.
    func add(_ voice: ReferenceVoice, clip: [Float]?) async throws
    /// Removes the voice and its clip.
    func remove(_ id: ReferenceVoice.ID) async throws
    /// The voice's clip, or nil when the store has none.
    func clip(for id: ReferenceVoice.ID) async throws -> [Float]?
}

public struct VoiceDirectory: VoiceStore {
    public let url: URL

    public static var defaultURL: URL {
        URL.applicationSupportDirectory.appending(path: "DramaStudio/Voices", directoryHint: .isDirectory)
    }

    public init(url: URL = VoiceDirectory.defaultURL) { self.url = url }

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

    public func clip(for id: ReferenceVoice.ID) throws -> [Float]? {
        let clip = clipFile(for: id)
        return FileManager.default.fileExists(atPath: clip.path) ? try ReferenceAudio.load(url: clip) : nil
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
}

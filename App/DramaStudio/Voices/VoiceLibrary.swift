// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import DramaEngine
import DramaServer
import Foundation

/// The HTTP API writes through this object, not the directory, so its
/// changes show up in the window at once.
@MainActor @Observable
final class VoiceLibrary: VoiceStore {
    private(set) var voices: [ReferenceVoice] = []
    private(set) var unreadableFiles: [String] = []
    @ObservationIgnored private let files: VoiceDirectory
    @ObservationIgnored private let mayUseDramaCodec: @MainActor () -> Bool

    /// - Parameter mayUseDramaCodec: Whether the Drama 3 codec may run now.
    ///   The app passes the terms agreement, which covers the codec too.
    init(directory: URL = VoiceDirectory.defaultURL, codecDirectory: URL = AppHome.codec, mayUseDramaCodec: @escaping @MainActor () -> Bool = { true }) {
        files = VoiceDirectory(url: directory, codecDirectory: codecDirectory)
        self.mayUseDramaCodec = mayUseDramaCodec
        reload()
    }

    func reload() {
        (voices, unreadableFiles) = files.load()
    }

    func allVoices() -> [ReferenceVoice] { voices }

    func add(_ voice: ReferenceVoice, clip: [Float]?) throws {
        try files.add(voice, clip: clip)
        reload()
    }

    func remove(_ id: ReferenceVoice.ID) throws {
        try files.remove(id)
        reload()
    }

    /// A voice saved without a clip gets one from the Drama 3 codec, which
    /// is a Fish Audio model: until it may run, such a voice fails with
    /// ``ClipNeedsDrama3``.
    func clip(for id: ReferenceVoice.ID) async throws -> [Float]? {
        if !files.hasClip(for: id), !mayUseDramaCodec(), let voice = voice(id), voice.encodings[.drama3] != nil {
            throw ClipNeedsDrama3(voiceName: voice.name)
        }
        return try await files.clip(for: id)
    }

    func storeEncoding(_ encoding: VoiceEncoding, for id: ReferenceVoice.ID) throws {
        try files.storeEncoding(encoding, for: id)
        reload()
    }

    func voice(_ id: ReferenceVoice.ID?) -> ReferenceVoice? {
        voices.first { $0.id == id }
    }
}

/// A voice kept only as Drama 3 codes was asked to speak on another engine before Drama 3 was set up.
struct ClipNeedsDrama3: LocalizedError, Equatable {
    var voiceName: String
    var errorDescription: String? {
        "The voice \(voiceName) was saved without its recording, and only Drama 3's voice codec can recreate it. Set up Drama 3 in Settings, or add the voice again from a recording."
    }
}

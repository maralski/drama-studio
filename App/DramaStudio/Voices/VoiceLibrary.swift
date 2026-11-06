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

    init(directory: URL = VoiceDirectory.defaultURL) {
        files = VoiceDirectory(url: directory)
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

    func clip(for id: ReferenceVoice.ID) throws -> [Float]? {
        try files.clip(for: id)
    }

    func voice(_ id: ReferenceVoice.ID?) -> ReferenceVoice? {
        voices.first { $0.id == id }
    }
}

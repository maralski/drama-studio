// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import DramaEngine
import Foundation

struct ArchivedTake: Codable, Sendable, Equatable {
    var id: UUID
    var number: Int
    var note: String?
    var take: Take
    var sentences: [NarrationTake.Sentence]
    var basis: TakesBasis?
    var isIncomplete: Bool

    init(_ value: SessionTake) {
        id = value.id
        number = value.number
        note = value.note
        take = value.take
        sentences = value.sentences
        basis = value.basis
        isIncomplete = value.isIncomplete
    }

    var audioPath: String { "audio/take-\(id.uuidString).wav" }

    func restore(in bundle: URL) throws -> SessionTake {
        SessionTake(id: id, number: number, note: note, take: take,
                    samples: try StudioProject.readSamples(from: bundle.appending(path: audioPath)),
                    sentences: sentences, basis: basis, isIncomplete: isIncomplete)
    }
}

struct ProjectArchive: Codable, Sendable, Equatable {
    var plainTakes: [ArchivedTake]
    var sectionTakes: [UUID: [ArchivedTake]]
    var sectionExports: [UUID: UUID]
    var selectedTake: UUID?
    var exportTake: UUID?
    var sectionSelections: [UUID: UUID]
    var selectedSegment: UUID?
    var originalProjectURL: URL?
    var voices: [Int: VoiceChoice]
    var sessionVoices: [VoiceChoice: Take]
    var screenplayMode: Bool

    func restoreTakes(in bundle: URL) throws -> [UUID: SessionTake] {
        var restored: [UUID: SessionTake] = [:]
        for take in allTakes {
            guard restored[take.id] == nil else { throw CocoaError(.fileReadCorruptFile) }
            restored[take.id] = try take.restore(in: bundle)
        }
        for id in [selectedTake, exportTake].compactMap({ $0 }) {
            guard plainTakes.contains(where: { $0.id == id }) else { throw CocoaError(.fileReadCorruptFile) }
        }
        for (section, take) in Array(sectionExports) + Array(sectionSelections) {
            guard sectionTakes[section]?.contains(where: { $0.id == take }) == true else { throw CocoaError(.fileReadCorruptFile) }
        }
        return restored
    }

    var allTakes: [ArchivedTake] { plainTakes + sectionTakes.values.flatMap { $0 } }
}

// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import DramaEngine
import Foundation

/// How a document's sections become speech requests: one request per
/// narrated segment, cast for that segment's speaker alone, with the
/// speaker's earlier sections as its voice anchor and history.
enum SectionRequests {
    /// The segments Generate narrates, in document order: not skipped, with spoken text.
    static func queue(_ document: NarrationDocument) -> [NarrationDocument.Segment.ID] {
        document.segments.filter { !$0.skip && !spokenText(of: $0).isEmpty }.map(\.id)
    }

    static func spokenText(of segment: NarrationDocument.Segment) -> String {
        segment.text.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    static func speakerName(_ tag: Int) -> String {
        tag == 0 ? "Narrator" : "Speaker \(tag)"
    }

    /// The distinct speakers of a document's segments, speaker 0 first.
    static func speakers(of document: NarrationDocument) -> [Speaker] {
        Set(document.segments.map { $0.speakerTag ?? 0 }).sorted().map { Speaker(index: $0, name: speakerName($0)) }
    }

    /// A one-speaker casting whose speaker 0 carries the segment speaker's assignment.
    static func casting(forSpeaker tag: Int, assignments: [Int: VoiceChoice], library: [ReferenceVoice],
                        sessionVoices: [VoiceChoice: Take] = [:]) -> Casting {
        Casting(speakers: [Speaker(index: 0, name: speakerName(tag))], assignments: [0: assignments[tag] ?? .modelDefault],
                library: library, sessionVoices: sessionVoices)
    }

    /// The request for one segment. `takes` are the chosen takes of narrated segments.
    static func request(for segment: NarrationDocument.Segment, in document: NarrationDocument,
                        casting: Casting, takes: [NarrationDocument.Segment.ID: Take],
                        config: GenerationConfig) -> SpeechRequest {
        let speaker = segment.speakerTag ?? 0
        let earlier = document.segments.prefix { $0.id != segment.id }
            .filter { !$0.skip && ($0.speakerTag ?? 0) == speaker }
            .compactMap { takes[$0.id] }
            .filter { $0.frameCount > 0 }
        var previous: [Take] = []
        if let anchor = casting.voiceTake ?? (casting.references.isEmpty ? earlier.first : nil), anchor.frameCount > 0 {
            previous.append(anchor)
        }
        if let latest = earlier.last, !previous.contains(latest) { previous.append(latest) }
        // Narration is exported, so a dropped sentence costs more than a later start:
        // sentence mode keeps the short-chunk retry that low-latency streaming gives up.
        var narration = config
        narration.streaming = .sentence
        return SpeechRequest(text: spokenText(of: segment), references: casting.references, character: casting.character,
                             previousTakes: previous, config: narration)
    }
}

// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation

/// Ming's view of a request: one sentence per generation turn, each with its
/// Ming text, style and speaker, and exactly one source for the voice. Ming
/// has no history, so earlier takes count only as the voice to clone.
package struct MingPlan: Sendable, Equatable {
    package struct Chunk: Sendable, Equatable {
        /// The script text of the turn, cues as written; the word check and the sentence record use it.
        package var line: String
        package var spoken: MingCues.Line
    }

    /// Where the voice comes from. One source per request. A designed or
    /// built-in voice describes or names itself on the first chunk, which is
    /// that chunk's one task, and is cloned from its audio for the rest.
    package enum Voice: Sendable, Equatable {
        /// Reference voices in speaker order; a chunk's speaker tag picks one.
        case clone([MingVoiceEncoding])
        case design(String)
        /// A built-in voice by its Ming name.
        case builtIn(String)

        /// The encodings chunks clone before any chunk has spoken.
        package var references: [MingVoiceEncoding] {
            if case .clone(let encodings) = self { encodings } else { [] }
        }
    }

    package var chunks: [Chunk]
    package var voice: Voice
    /// The language the request names or its text is written in, or nil when that is uncertain.
    package var language: String?
    package var turnText: String
    package var notices: [SpeechNotice]
    package var expectedSeconds: [Double]

    package init(_ request: SpeechRequest) throws {
        let outline = try SpeechOutline(request, engine: .ming)
        let translation = MingCues.translate(lines: outline.lines, note: request.note)
        chunks = zip(outline.lines, translation.lines).map { Chunk(line: $0, spoken: $1) }
        expectedSeconds = outline.expectedSeconds
        turnText = GenerationPlan.turnText(text: request.text, note: request.note, character: request.character)
        var notices = outline.notices
        let previous = request.previousTakes.compactMap { take -> MingVoiceEncoding? in
            if case .ming(let reference, _) = take.context { reference } else { nil }
        }
        if !request.references.isEmpty {
            var encodings: [MingVoiceEncoding] = []
            for voice in request.references {
                guard case .ming(let reference)? = voice.encodings[.ming] else { throw PromptError.notEncoded(voice.name, .ming) }
                encodings.append(reference)
            }
            voice = .clone(encodings)
        } else if let first = previous.first {
            voice = .clone([first])
        } else if let character = request.character {
            try Drama3Prompt.validate(character)
            voice = .design(character.description)
        } else {
            voice = .builtIn(request.builtInVoice?.name ?? MingVoices.defaultName)
        }
        if case .clone(let encodings) = voice {
            for chunk in chunks where chunk.spoken.speaker >= encodings.count {
                let tag = "<|speaker:\(chunk.spoken.speaker)|>"
                if !notices.contains(.unsupportedCue(tag)) { notices.append(.unsupportedCue(tag)) }
            }
        } else if case .cue(let cue, _)? = chunks.first?.spoken.style, !notices.contains(.unsupportedCue(cue)) {
            notices.append(.unsupportedCue(cue))
        }
        self.notices = notices
        language = try? ScriptLanguage.language(of: request)
    }

    /// The reference a chunk clones, once the voice is known: its speaker's
    /// encoding, or the last one when the tag names a speaker with no voice.
    package func reference(for chunk: Chunk, among encodings: [MingVoiceEncoding]) -> MingVoiceEncoding? {
        guard !encodings.isEmpty else { return nil }
        return encodings[min(chunk.spoken.speaker, encodings.count - 1)]
    }
}

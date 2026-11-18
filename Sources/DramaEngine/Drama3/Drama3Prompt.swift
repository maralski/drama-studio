// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation
package import DramaTTS

/// Why the prompt builder rejected a request. The `String` payloads name the
/// part of the request, such as `"text"` or `"reference 1"`.
public enum PromptError: Error, CustomStringConvertible, Equatable {
    /// The part is empty or white space.
    case emptyText(String)
    /// The part contains a token only the prompt builder may write, such as
    /// `<|im_start|>`; the second string names it.
    case structuralToken(String, String)
    /// A speaker tag is not `<|speaker:0|>` to `<|speaker:15|>`.
    case badSpeakerTag(String)
    /// Codes are not 10 codebooks of at least one frame with values under 4096.
    case badCodes(String)
    /// The prompt has this many rows, over ``Drama3Prompt/maxRows``.
    case tooLong(Int)
    /// The request names more than one of reference voices, a designed voice and a built-in voice.
    case voiceConflict
    /// The named voice has no encoding for the engine.
    case notEncoded(String, SpeechEngineID)
    /// The engine cannot take the request; the string says what.
    case engineUnsupported(SpeechEngineID, String)

    /// A message for a person.
    public var description: String {
        switch self {
        case .voiceConflict: return "a request takes one voice: reference voices, a designed character or a built-in voice"
        case .notEncoded(let name, let engine): return "the voice \(name) has not been encoded for \(engine.capabilities.name)"
        case .engineUnsupported(let engine, let what): return "\(engine.capabilities.name) can't \(what)"
        case .emptyText(let w): return "\(w): text must not be empty"
        case .structuralToken(let w, let t): return "\(w): \(t) is written by the prompt builder"
        case .badSpeakerTag(let w): return "\(w): speaker tags must be <|speaker:0|> to <|speaker:15|>"
        case .badCodes(let w): return "\(w): codes must be [10, frames >= 1] in [0, 4096)"
        case .tooLong(let rows): return "prompt has \(rows) rows; at most \(Drama3Prompt.maxRows)"
        }
    }
}


/// The Drama 3 prompt layout, `build_drama3_prompt` from the reference.
///
/// The public entry is ``render(text:references:character:history:)``, which
/// shows the prompt one chunk would get, for previews and debugging. The
/// engine builds the token rows itself.
public enum Drama3Prompt: Sendable {
    /// The most rows a prompt may have. Each text token and each reference
    /// or history frame is one row.
    public static let maxRows = DramaPrompt.maxRows
    /// Token 252196; the design instruction names it between consecutive lines.
    public static let audioSeparator = DramaPrompt.audioSeparator
    package static let designSpeakerTag = DramaPrompt.designSpeakerTag
    package static var speakerTag: NSRegularExpression { DramaPrompt.speakerTag }

    /// A voice reference as the prompt holds it.
    public struct Reference: Sendable, Equatable {
        /// What the clip says. Without an explicit speaker tag, the reference's
        /// array index identifies its speaker.
        public var transcript: String
        /// `[10][T]` codec codes.
        public var codes: [[Int32]]
        /// Creates a reference.
        public init(transcript: String, codes: [[Int32]]) { self.transcript = transcript; self.codes = codes }

        /// The voice's Drama 3 encoding as a reference.
        /// - Throws: ``PromptError/notEncoded(_:_:)`` when the voice has no Drama 3 encoding.
        package init(_ voice: ReferenceVoice) throws {
            guard case .drama3(let codes)? = voice.encodings[.drama3] else { throw PromptError.notEncoded(voice.name, .drama3) }
            self.init(transcript: voice.transcript, codes: codes)
        }

        package var drama: DramaPrompt.Reference { DramaPrompt.Reference(transcript: transcript, codes: codes) }
    }

    /// An earlier turn fed back as history: its user turn text and its codes.
    public struct HistoryTurn: Sendable, Hashable {
        /// The user turn as the prompt held it.
        public var text: String
        /// `[10][T]` codec codes.
        public var codes: [[Int32]]
        /// Creates a turn.
        public init(text: String, codes: [[Int32]]) { self.text = text; self.codes = codes }

        /// A Drama 3 take as a turn; nil for another engine's take.
        package init?(_ take: Take) {
            switch take.context {
            case .drama3(let codes): self.init(text: take.text, codes: codes)
            case .ming: return nil
            }
        }

        package var drama: DramaPrompt.HistoryTurn { DramaPrompt.HistoryTurn(text: text, codes: codes) }
    }

    /// A designed voice, which replaces the instruction with its description.
    public typealias Character = VoiceDesign

    static func validate(_ character: Character) throws {
        try Drama3Errors.translating { try DramaPrompt.validate(character.drama) }
    }

    package static func withSpeakerTag(_ text: String, speaker: Int = 0) -> String {
        DramaPrompt.withSpeakerTag(text, speaker: speaker)
    }

    /// The prompt one chunk would get, with each run of codec frames shown
    /// as `<codes:N>`.
    /// - Throws: ``PromptError`` for a part the prompt builder rejects.
    public static func render(text: String, references: [Reference] = [], character: Character? = nil, history: [HistoryTurn] = []) throws -> String {
        try Drama3Errors.translating {
            try DramaPrompt.render(text: text, references: references.map(\.drama), character: character?.drama, history: history.map(\.drama))
        }
    }

    /// The prompt for parts of a ``GenerationPlan``, which holds the engine's own types.
    package static func render(text: String, references: [DramaPrompt.Reference], character: Character? = nil,
                               history: [DramaPrompt.HistoryTurn] = []) throws -> String {
        try Drama3Errors.translating {
            try DramaPrompt.render(text: text, references: references, character: character?.drama, history: history)
        }
    }
}

/// How the engine cuts text into generation turns.
///
/// The public entry is ``splitSentencesTagged(_:)``, the cut the default
/// ``GenerationConfig/Chunking/sentences`` makes. Use it to preview the
/// chunks a request will produce, for captions, highlights, or a
/// ``GenerationEstimate``.
public enum Drama3Chunking: Sendable {
    /// The chunks of `text` under the default sentence chunking, in order.
    ///
    /// Text is cut at sentence ends (`.`, `!`, `?`, `;`, `…`, a newline, or a
    /// CJK one). Decimals such as 3.14, `[cues]`, and `<|special|>` tokens
    /// are never cut. Paired passage tags are reopened for each sentence. Short sentences stay separate so
    /// an end-of-speech after one cannot omit the next. With `<|speaker:N|>`
    /// tags each tagged segment is split on its own and every chunk keeps its
    /// tag; text before the first tag is speaker 0.
    public static func splitSentencesTagged(_ text: String) -> [String] {
        DramaChunking.splitSentencesTagged(text)
    }
}

/// Why `tokenizer.json` could not be read.
public enum TokenizerError: Error, Sendable {
    /// The file is missing a part the tokenizer needs; the string names it.
    case invalidFile(String)
}

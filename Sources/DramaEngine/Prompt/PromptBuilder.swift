// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation

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
    /// The request has both a designed voice and reference voices.
    case voiceConflict

    /// A message for a person.
    public var description: String {
        switch self {
        case .voiceConflict: return "a designed character and reference voices can't be combined"
        case .emptyText(let w): return "\(w): text must not be empty"
        case .structuralToken(let w, let t): return "\(w): \(t) is written by the prompt builder"
        case .badSpeakerTag(let w): return "\(w): speaker tags must be <|speaker:0|> to <|speaker:15|>"
        case .badCodes(let w): return "\(w): codes must be [10, frames >= 1] in [0, 4096)"
        case .tooLong(let rows): return "prompt has \(rows) rows; at most \(Drama3Prompt.maxRows)"
        }
    }
}

/// The Unicode White_Space set the reference strips with.
enum UnicodeWhitespace {
    static let scalars: Set<Unicode.Scalar> = {
        var s = Set<Unicode.Scalar>()
        for v in [0x9, 0xA, 0xB, 0xC, 0xD, 0x20, 0x85, 0xA0, 0x1680, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000] { s.insert(Unicode.Scalar(UInt32(v))!) }
        for v in 0x2000...0x200A { s.insert(Unicode.Scalar(UInt32(v))!) }
        return s
    }()

    static func strip(_ text: String) -> String {
        String(String.UnicodeScalarView(text.unicodeScalars.drop(while: { scalars.contains($0) }).reversed().drop(while: { scalars.contains($0) }).reversed()))
    }

    static func stripLeading(_ text: String) -> String {
        String(String.UnicodeScalarView(text.unicodeScalars.drop(while: { scalars.contains($0) })))
    }
}

/// The Drama 3 prompt layout, `build_drama3_prompt` from the reference.
///
/// The public entry is ``render(text:references:character:history:)``, which
/// shows the prompt one chunk would get, for previews and debugging. The
/// engine builds the token rows itself.
public enum Drama3Prompt: Sendable {
    package static let ttsInstruction = "convert the provided text to speech"
    package static let referenceInstruction = "convert the provided text to speech reference to the following:"
    package static let messageEnd = "<|im_end|>\n"
    package static let generationStart = "<|im_start|>assistant\n<|voice|>"
    /// The most rows a prompt may have. Each text token and each reference
    /// or history frame is one row.
    public static let maxRows = 8192
    static let structuralTokens = ["<|im_start|>", "<|im_end|>", "<|voice|>", "<|text|>", "<|interleave|>", "<|semantic:"]
    static let speakerTag = try! NSRegularExpression(pattern: "<\\|speaker:([0-9]+)\\|>")

    /// Token 252196; the design instruction names it between consecutive lines.
    public static let audioSeparator = "<|audio_separator|>"

    /// A voice reference as the prompt holds it.
    public struct Reference: Sendable, Equatable {
        /// What the clip says.
        public var transcript: String
        /// `[10][T]` codec codes.
        public var codes: [[Int32]]
        /// Creates a reference. `ReferenceVoice` converts as
        /// `Reference(transcript: voice.transcript, codes: voice.codes)`.
        public init(transcript: String, codes: [[Int32]]) { self.transcript = transcript; self.codes = codes }
    }

    /// An earlier turn fed back as history.
    public typealias HistoryTurn = Take
    /// A designed voice, which replaces the instruction with its description.
    public typealias Character = VoiceDesign

    static func validate(_ character: Character) throws {
        for (field, value) in [("character name", character.name), ("character description", character.description)] + (character.scene.map { [("scene", $0)] } ?? []) {
            if field == "scene" && UnicodeWhitespace.strip(value).isEmpty { continue }
            try validate(value, field)
            if value.contains(audioSeparator) { throw PromptError.structuralToken(field, audioSeparator) }
        }
    }

    static func validate(_ text: String, _ location: String) throws {
        guard !UnicodeWhitespace.strip(text).isEmpty else { throw PromptError.emptyText(location) }
        for token in structuralTokens where text.contains(token) { throw PromptError.structuralToken(location, token) }
        let ns = text as NSString
        var search = 0
        while true {
            let r = ns.range(of: "<|speaker", range: NSRange(location: search, length: ns.length - search))
            if r.location == NSNotFound { break }
            guard let m = speakerTag.firstMatch(in: text, options: .anchored, range: NSRange(location: r.location, length: ns.length - r.location)) else {
                throw PromptError.badSpeakerTag(location)
            }
            let digits = ns.substring(with: m.range(at: 1))
            guard let n = Int(digits), String(n) == digits, n <= ModelConstants.maxSpeakerID else { throw PromptError.badSpeakerTag(location) }
            search = r.location + 1
        }
    }

    /// Prefix `<|speaker:0|> ` unless a tag already leads the text after white space.
    package static func withSpeakerTag(_ text: String) -> String {
        let body = UnicodeWhitespace.stripLeading(text)
        if speakerTag.firstMatch(in: body, options: .anchored, range: NSRange(location: 0, length: (body as NSString).length)) != nil { return text }
        return "<|speaker:0|> " + text
    }

    static func checkCodes(_ codes: [[Int32]], _ location: String) throws {
        guard codes.count == ModelConstants.numCodebooks, let frames = codes.first?.count, frames >= 1,
              codes.allSatisfy({ $0.count == frames }),
              codes.allSatisfy({ $0.allSatisfy { $0 >= 0 && $0 < Int32(ModelConstants.codebookSize) } }) else {
            throw PromptError.badCodes(location)
        }
    }

    static func codeRows(_ codes: [[Int32]]) -> [Frame] {
        (0..<codes[0].count).map { t in [codes[0][t] + ModelConstants.semanticStart] + codes.map { $0[t] } }
    }

    package static func build(text: String, tokenizer: Tokenizer, references: [Reference] = [], character: Character? = nil, history: [HistoryTurn] = []) throws -> [Frame] {
        try buildWithPrefix(text: text, tokenizer: tokenizer, references: references, character: character, history: history).rows
    }

    enum Segment: Equatable { case text(String), codes([[Int32]]) }

    /// The prompt for one chunk as text, with each block of codes shown as
    /// `<codes:N>` for N frames. The engine tokenizes exactly these pieces.
    ///
    /// - Parameters:
    ///   - text: One chunk's user turn. A missing speaker tag becomes
    ///     `<|speaker:0|>`.
    ///   - references: Voice references, in speaker order.
    ///   - character: A designed voice; mutually exclusive with `references`.
    ///   - history: Earlier turns, oldest first.
    /// - Throws: ``PromptError`` for text or codes the builder rejects.
    public static func render(text: String, references: [Reference] = [], character: Character? = nil, history: [HistoryTurn] = []) throws -> String {
        try layout(text: text, references: references, character: character, history: history).segments.map { segment in
            switch segment {
            case .text(let s): s
            case .codes(let c): "<codes:\(c.first?.count ?? 0)>"
            }
        }.joined()
    }

    /// Rows plus the number of leading rows that make up the system message
    /// (instruction and references), which every chunk of a request shares.
    /// A `character` replaces the instruction with its design prompt and
    /// speaks every turn as `Name: text` instead of a speaker tag.
    package static func buildWithPrefix(text: String, tokenizer: Tokenizer, references: [Reference] = [], character: Character? = nil,
                                       history: [HistoryTurn] = []) throws -> (rows: [Frame], prefixRows: Int) {
        let (segments, systemSegments) = try layout(text: text, references: references, character: character, history: history)
        var rows: [Frame] = []
        var prefixRows = 0
        for (i, segment) in segments.enumerated() {
            if i == systemSegments { prefixRows = rows.count }
            switch segment {
            case .text(let s):
                // Every literal is tokenized on its own: joining changes ids at the seams.
                for id in tokenizer.encode(s) { rows.append([id] + [Int32](repeating: 0, count: ModelConstants.numCodebooks)) }
            case .codes(let c):
                rows += codeRows(c)
            }
        }
        guard rows.count <= maxRows else { throw PromptError.tooLong(rows.count) }
        return (rows, prefixRows)
    }

    static func layout(text: String, references: [Reference], character: Character?, history: [HistoryTurn]) throws -> (segments: [Segment], systemSegments: Int) {
        try validate(text, "text")
        if let character {
            guard references.isEmpty else { throw PromptError.voiceConflict }
            try validate(character)
        }
        let turnText: (String) -> String = { character?.turn($0) ?? withSpeakerTag($0) }
        let instruction = character?.systemPrompt ?? (references.isEmpty ? ttsInstruction : referenceInstruction)
        var segments: [Segment] = [.text("<|im_start|>system\n"), .text(instruction)]
        for (index, ref) in references.enumerated() {
            let location = "reference \(index + 1)"
            try validate(ref.transcript, location)
            try checkCodes(ref.codes, location)
            // The tag must follow "Text: " directly; leading white space moves behind it.
            let tagged = withSpeakerTag(ref.transcript)
            let body = UnicodeWhitespace.stripLeading(tagged)
            let leading = String(tagged.unicodeScalars.prefix(tagged.unicodeScalars.count - body.unicodeScalars.count))
            let ns = body as NSString
            let m = speakerTag.firstMatch(in: body, options: .anchored, range: NSRange(location: 0, length: ns.length))!
            let transcript = ns.substring(with: m.range) + leading + ns.substring(from: m.range.location + m.range.length)
            segments.append(.text("\n\n[Reference \(index + 1)]\nText: \(transcript)\nSpeech: "))
            segments.append(.codes(ref.codes))
        }
        segments.append(.text(messageEnd))
        let systemSegments = segments.count
        for (index, turn) in history.enumerated() {
            let location = "history \(index + 1)"
            try validate(turn.text, location)
            try checkCodes(turn.codes, location)
            segments += [.text("<|im_start|>user\n"), .text(turnText(turn.text)), .text(messageEnd), .text(generationStart), .codes(turn.codes), .text(messageEnd)]
        }
        segments += [.text("<|im_start|>user\n"), .text(turnText(text)), .text(messageEnd), .text(generationStart)]
        return (segments, systemSegments)
    }
}

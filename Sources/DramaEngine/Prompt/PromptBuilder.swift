import Foundation

public enum PromptError: Error, CustomStringConvertible, Equatable {
    case emptyText(String)
    case structuralToken(String, String)
    case badSpeakerTag(String)
    case badCodes(String)
    case tooLong(Int)

    public var description: String {
        switch self {
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

/// `build_drama3_prompt` from the reference: frame rows ending in the open
/// assistant voice turn.
public enum Drama3Prompt {
    public static let ttsInstruction = "convert the provided text to speech"
    public static let referenceInstruction = "convert the provided text to speech reference to the following:"
    public static let messageEnd = "<|im_end|>\n"
    public static let generationStart = "<|im_start|>assistant\n<|voice|>"
    public static let maxRows = 8192
    static let structuralTokens = ["<|im_start|>", "<|im_end|>", "<|voice|>", "<|text|>", "<|interleave|>", "<|semantic:"]
    static let speakerTag = try! NSRegularExpression(pattern: "<\\|speaker:([0-9]+)\\|>")

    public struct Reference: Sendable {
        public var transcript: String
        public var codes: [[Int32]]
        public init(transcript: String, codes: [[Int32]]) { self.transcript = transcript; self.codes = codes }
    }

    public struct HistoryTurn: Sendable {
        public var text: String
        public var codes: [[Int32]]
        public init(text: String, codes: [[Int32]]) { self.text = text; self.codes = codes }
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
    public static func withSpeakerTag(_ text: String) -> String {
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

    public static func build(text: String, tokenizer: Tokenizer, references: [Reference] = [], history: [HistoryTurn] = []) throws -> [Frame] {
        try buildWithPrefix(text: text, tokenizer: tokenizer, references: references, history: history).rows
    }

    /// Rows plus the number of leading rows that make up the system message
    /// (instruction and references), which every chunk of a request shares.
    public static func buildWithPrefix(text: String, tokenizer: Tokenizer, references: [Reference] = [], history: [HistoryTurn] = []) throws -> (rows: [Frame], prefixRows: Int) {
        try validate(text, "text")
        enum Segment { case text(String), codes([[Int32]]) }
        var segments: [Segment] = [.text("<|im_start|>system\n"), .text(references.isEmpty ? ttsInstruction : referenceInstruction)]
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
            segments += [.text("<|im_start|>user\n"), .text(withSpeakerTag(turn.text)), .text(messageEnd), .text(generationStart), .codes(turn.codes), .text(messageEnd)]
        }
        segments += [.text("<|im_start|>user\n"), .text(withSpeakerTag(text)), .text(messageEnd), .text(generationStart)]
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
}

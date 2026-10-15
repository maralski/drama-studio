import Foundation

/// Long-text chunking from the reference (`chunking.py`): about 300 UTF-8
/// bytes per chunk, at most 3 speakers, cut at sentence ends, then clause
/// marks, then spaces. Operates on Unicode scalars like Python's `str`.
public enum Drama3Chunking {
    public static let chunkBytes = 300
    public static let maxSpeakers = 3
    public static let chunkMaxNewFrames = 600
    static let sentenceEnds = Set(".!?;。！？；…\n".unicodeScalars)
    static let clauseMarks = Set(",:，：、".unicodeScalars)
    static let spaces = Set(" ".unicodeScalars)
    // Never split inside special tokens, paired tags such as <whisper>…</whisper>, or [bracketed] cues.
    static let protectedSpan = try! NSRegularExpression(pattern: "<\\|[^|>]*\\|>|<([A-Za-z][\\w-]*)\\b[^>]*>.*?</\\1>|\\[[^\\[\\]]*\\]", options: .dotMatchesLineSeparators)

    typealias Piece = (text: [Unicode.Scalar], protected: Bool)

    static func utf8Len(_ s: [Unicode.Scalar]) -> Int { s.reduce(0) { $0 + $1.utf8.count } }
    static func string(_ s: [Unicode.Scalar]) -> String { String(String.UnicodeScalarView(s)) }

    static func protectedPieces(_ text: String) -> [Piece] {
        let scalars = Array(text.unicodeScalars)
        let ns = text as NSString
        var pieces: [Piece] = []
        var last = 0
        // Convert UTF-16 match ranges to scalar offsets.
        var utf16ToScalar: [Int: Int] = [:]
        var u = 0
        for (i, sc) in scalars.enumerated() { utf16ToScalar[u] = i; u += sc.utf16.count }
        utf16ToScalar[u] = scalars.count
        for m in protectedSpan.matches(in: text, range: NSRange(location: 0, length: ns.length)) {
            let start = utf16ToScalar[m.range.location]!, end = utf16ToScalar[m.range.location + m.range.length]!
            if start > last { pieces.append((Array(scalars[last..<start]), false)) }
            pieces.append((Array(scalars[start..<end]), true))
            last = end
        }
        if last < scalars.count { pieces.append((Array(scalars[last...]), false)) }
        return pieces
    }

    static func breaksAt(_ text: [Unicode.Scalar], _ index: Int, _ marks: Set<Unicode.Scalar>) -> Bool {
        guard marks.contains(text[index]) else { return false }
        // Keep decimals such as 3.14 in one piece.
        func isDigit(_ s: Unicode.Scalar) -> Bool { s.properties.numericType == .decimal || s.properties.numericType == .digit }
        return !(text[index] == "." && index > 0 && index < text.count - 1 && isDigit(text[index - 1]) && isDigit(text[index + 1]))
    }

    static func breakAtMarks(_ pieces: [Piece], _ length: Int, _ marks: Set<Unicode.Scalar>) -> [Piece] {
        var out: [Piece] = []
        for (text, protected) in pieces {
            if protected || utf8Len(text) <= length { out.append((text, protected)); continue }
            var start = 0
            for index in 0..<text.count where breaksAt(text, index, marks) {
                out.append((Array(text[start...index]), false))
                start = index + 1
            }
            if start < text.count { out.append((Array(text[start...]), false)) }
        }
        return out
    }

    static func breakByLength(_ pieces: [Piece], _ length: Int) -> [Piece] {
        var out: [Piece] = []
        for (text, protected) in pieces {
            if protected || utf8Len(text) <= length { out.append((text, protected)); continue }
            var current: [Unicode.Scalar] = []
            var size = 0
            for ch in text {
                if !current.isEmpty && size + ch.utf8.count > length { out.append((current, false)); current = []; size = 0 }
                current.append(ch)
                size += ch.utf8.count
            }
            out.append((current, false))
        }
        return out
    }

    public static func splitByLength(_ text: String, length: Int = chunkBytes) -> [String] {
        if UnicodeWhitespace.strip(text).isEmpty { return [] }
        if text.utf8.count <= length { return [text] }
        var pieces = breakAtMarks(protectedPieces(text), 0, sentenceEnds)
        for marks in [clauseMarks, spaces] { pieces = breakAtMarks(pieces, length, marks) }
        pieces = breakByLength(pieces, length)
        var chunks: [[Unicode.Scalar]] = []
        var current: [Unicode.Scalar] = []
        for (piece, _) in pieces {
            if !current.isEmpty && utf8Len(current) + utf8Len(piece) > length { chunks.append(current); current = [] }
            current += piece
        }
        chunks.append(current)
        return chunks.map { UnicodeWhitespace.strip(string($0)) }.filter { !$0.isEmpty }
    }

    /// (speaker id, text) per tagged segment; text before the first tag is speaker 0.
    public static func parseSpeakerSegments(_ text: String) -> [(String, String)] {
        let ns = text as NSString
        let matches = Drama3Prompt.speakerTag.matches(in: text, range: NSRange(location: 0, length: ns.length))
        var segments: [(String, String)] = []
        let first = ns.substring(to: matches.first?.range.location ?? ns.length)
        if !UnicodeWhitespace.strip(first).isEmpty { segments.append(("0", first)) }
        for (i, m) in matches.enumerated() {
            let speaker = ns.substring(with: m.range(at: 1))
            let bodyStart = m.range.location + m.range.length
            let bodyEnd = i + 1 < matches.count ? matches[i + 1].range.location : ns.length
            let body = ns.substring(with: NSRange(location: bodyStart, length: bodyEnd - bodyStart))
            if !UnicodeWhitespace.strip(body).isEmpty { segments.append((speaker, body)) }
        }
        return segments
    }

    public static func split(_ text: String, maxBytes: Int = chunkBytes, maxSpeakers: Int = maxSpeakers) -> [String] {
        let ns = text as NSString
        if Drama3Prompt.speakerTag.firstMatch(in: text, range: NSRange(location: 0, length: ns.length)) == nil {
            return splitByLength(text, length: maxBytes)
        }
        var segments: [(String, String)] = []
        for (speaker, body) in parseSpeakerSegments(text) {
            for piece in splitByLength(body, length: maxBytes) { segments.append((speaker, "<|speaker:\(speaker)|> \(piece)")) }
        }
        var chunks: [String] = []
        var batch: [String] = []
        var speakers: [String] = []
        var size = 0
        for (speaker, segment) in segments {
            var newSpeaker = !speakers.contains(speaker)
            let tooManySpeakers = newSpeaker && speakers.count >= maxSpeakers
            let tooLong = !batch.isEmpty && size + segment.utf8.count > maxBytes
            if tooManySpeakers || tooLong {
                chunks.append(batch.joined(separator: "\n"))
                batch = []; speakers = []; size = 0
                newSpeaker = true
            }
            batch.append(segment)
            if newSpeaker { speakers.append(speaker) }
            size += segment.utf8.count
        }
        if !batch.isEmpty { chunks.append(batch.joined(separator: "\n")) }
        return chunks
    }
}

/// Context for the next chunk: the first chunk stays pinned plus the latest turn.
public struct Drama3History {
    public private(set) var pinned: Drama3Prompt.HistoryTurn?
    public private(set) var recent: [Drama3Prompt.HistoryTurn] = []
    public let recentTurns: Int

    public init(recentTurns: Int = 1) { self.recentTurns = recentTurns }

    public mutating func push(text: String, codes: [[Int32]]) {
        let turn = Drama3Prompt.HistoryTurn(text: text, codes: codes)
        if pinned == nil { pinned = turn } else {
            recent.append(turn)
            if recent.count > recentTurns { recent.removeFirst(recent.count - recentTurns) }
        }
    }

    /// Histories to try, longest first, ending with no history.
    public var candidates: [[Drama3Prompt.HistoryTurn]] {
        let p = pinned.map { [$0] } ?? []
        var out: [[Drama3Prompt.HistoryTurn]] = []
        for start in 0...recent.count { out.append(p + recent[start...]) }
        if !p.isEmpty { out.append([]) }
        return out
    }
}

// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation

/// The retake recipe from Fish Audio's Drama 3 Space: the line again, with
/// the note as a direction and as a cue before every sentence.
package enum DirectorsNote {
    package static let maxNoteLength = 120
    package static let literalReadingDirection = "Enunciate each word precisely"
    static let markup = try! NSRegularExpression(pattern: #"<\|[^|>]*\|>|\[[^\[\]]*\]|</?[A-Za-z][^<>]*>"#)
    static let sentenceBreak = try! NSRegularExpression(pattern: #"(?<=[.!?…])\s+"#)
    static let removedFromNotes = Set("[]<>|".unicodeScalars)

    /// The note with `[ ] < > |` removed and white space collapsed, at most
    /// 120 characters; nil when nothing is left.
    package static func clean(_ note: String) -> String? {
        let kept = String(String.UnicodeScalarView(note.unicodeScalars.filter { !removedFromNotes.contains($0) }))
        let collapsed = collapseWhitespace(kept)
        guard !collapsed.isEmpty else { return nil }
        return UnicodeWhitespace.strip(String(collapsed.prefix(maxNoteLength)))
    }

    /// The words the line speaks: `[cues]`, `<tags>` and `<|speaker:N|>` tags removed.
    package static func spokenWords(_ line: String) -> String {
        let ns = line as NSString
        let bare = markup.stringByReplacingMatches(in: line, range: NSRange(location: 0, length: ns.length), withTemplate: " ")
        return collapseWhitespace(bare).replacingOccurrences(of: #"\s+([,!?;:…]|\.(?!\d))"#, with: "$1", options: .regularExpression)
    }

    package static func sentences(_ line: String) -> [String] {
        let words = spokenWords(line)
        let ns = words as NSString
        var out: [String] = []
        var start = 0
        for m in sentenceBreak.matches(in: words, range: NSRange(location: 0, length: ns.length)) {
            out.append(ns.substring(with: NSRange(location: start, length: m.range.location - start)))
            start = m.range.location + m.range.length
        }
        out.append(ns.substring(from: start))
        return out.filter { !$0.isEmpty }
    }

    /// `Direction: <note>\nLine: [<note>] <sentence 1> [<note>] <sentence 2> …`
    package static func retakeText(line: String, note: String) -> String {
        directedText(line: line, notes: [note].compactMap(clean))
    }

    package static func literalRetry(line: String, direction: String?) -> String {
        directedText(line: line, notes: [direction, literalReadingDirection].compactMap { $0 }.compactMap(clean))
    }

    private static func directedText(line: String, notes: [String]) -> String {
        var body = line
        var speaker = ""
        let source = body as NSString
        if let tag = Drama3Prompt.speakerTag.firstMatch(in: body, options: .anchored, range: NSRange(location: 0, length: source.length)) {
            speaker = source.substring(with: tag.range) + " "
            body = UnicodeWhitespace.stripLeading(source.substring(from: NSMaxRange(tag.range)))
        }
        let cues = notes.map { "[\($0)]" }.joined(separator: " ")
        let cued = Drama3Chunking.splitSentencesTagged(body).map { "\(cues) \($0)" }.joined(separator: " ")
        return "\(speaker)Direction: \(notes.joined(separator: ". "))\nLine: \(cued)"
    }

    /// The spoken words of a take's turn text, for using the take as a voice
    /// reference: drops a character's `Name: ` and a retake's direction.
    package static func spokenText(ofTurn text: String, character: VoiceDesign?) -> String {
        var line = character?.removingName(from: text) ?? text
        let source = line as NSString
        if let tag = Drama3Prompt.speakerTag.firstMatch(in: line, options: .anchored, range: NSRange(location: 0, length: source.length)) {
            let body = UnicodeWhitespace.stripLeading(source.substring(from: NSMaxRange(tag.range)))
            if body.hasPrefix("Direction: ") { line = body }
        }
        if line.hasPrefix("Direction: "), let range = line.range(of: "\nLine: ") { line = String(line[range.upperBound...]) }
        return spokenWords(line)
    }

    static func collapseWhitespace(_ text: String) -> String {
        text.split(whereSeparator: { $0.isWhitespace }).joined(separator: " ")
    }
}

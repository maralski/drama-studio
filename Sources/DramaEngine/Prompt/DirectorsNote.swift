// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation

/// The retake recipe from Fish Audio's Drama 3 Space: the line again, with
/// the note as a direction and as a cue before every sentence.
package enum DirectorsNote {
    package static let maxNoteLength = 120
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

    /// The words the line speaks: `[cues]`, `<tags>` and speaker tags removed.
    package static func spokenWords(_ line: String) -> String {
        let ns = line as NSString
        let bare = markup.stringByReplacingMatches(in: line, range: NSRange(location: 0, length: ns.length), withTemplate: " ")
        return collapseWhitespace(bare).replacingOccurrences(of: #"\s+([,.!?;:…])"#, with: "$1", options: .regularExpression)
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
        let note = clean(note) ?? ""
        let cued = sentences(line).map { "[\(note)] \($0)" }.joined(separator: " ")
        return "Direction: \(note)\nLine: \(cued)"
    }

    /// The spoken words of a take's turn text, for using the take as a voice
    /// reference: drops a character's `Name: ` and a retake's direction.
    package static func spokenText(ofTurn text: String, character: VoiceDesign?) -> String {
        var line = text
        if let character {
            let prefix = character.turn("")
            if line.hasPrefix(prefix) { line.removeFirst(prefix.count) }
        }
        if line.hasPrefix("Direction: "), let range = line.range(of: "\nLine: ") { line = String(line[range.upperBound...]) }
        return spokenWords(line)
    }

    static func collapseWhitespace(_ text: String) -> String {
        text.split(whereSeparator: { $0.isWhitespace }).joined(separator: " ")
    }
}

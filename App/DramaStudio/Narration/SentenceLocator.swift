// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation

/// Finds spoken sentences in the editor text. Both sides are compared as
/// spoken text (speaker tags and markup removed, whitespace collapsed), and
/// each search starts where the previous match ended, so a repeated sentence
/// highlights in order.
enum SentenceLocator {
    /// The UTF-16 range in `text` of each sentence, nil for one not found.
    static func ranges(of sentences: [String], in text: String) -> [NSRange?] {
        let stripped = Stripped(text)
        var cursor = 0
        return sentences.map { sentence in
            let needle = Captions.spokenText(sentence)
            guard !needle.isEmpty else { return nil }
            let found = stripped.string.range(of: needle, range: NSRange(location: cursor, length: stripped.string.length - cursor))
            guard found.location != NSNotFound else { return nil }
            cursor = found.location + found.length
            let start = stripped.origin[found.location]
            let end = stripped.origin[cursor - 1] + 1
            return NSRange(location: start, length: end - start)
        }
    }

    /// The script's own sentences for a take whose text the pronunciation
    /// dictionary rewrote: the original lines when they rewrite to exactly
    /// what was spoken, else the spoken lines.
    static func scriptTexts(spoken: [String], original: [String], rewrite: (String) -> String) -> [String] {
        original.map(rewrite) == spoken ? original : spoken
    }

    private static let markup = try! NSRegularExpression(pattern: #"<\|[^|]*\|>|</?[A-Za-z][^<>]*>"#)

    /// `text` as spoken, with each UTF-16 unit's offset in the original.
    private struct Stripped {
        let string: NSString
        let origin: [Int]

        init(_ text: String) {
            let source = text as NSString
            let removed = SentenceLocator.markup.matches(in: text, range: NSRange(location: 0, length: source.length)).map(\.range)
            var units: [unichar] = []
            var origin: [Int] = []
            var pendingSpace: Int?
            var offset = 0
            var nextRemoved = 0
            while offset < source.length {
                if nextRemoved < removed.count, removed[nextRemoved].location == offset {
                    offset += removed[nextRemoved].length
                    nextRemoved += 1
                    continue
                }
                let unit = source.character(at: offset)
                if let scalar = UnicodeScalar(unit), CharacterSet.whitespacesAndNewlines.contains(scalar) {
                    if !units.isEmpty, pendingSpace == nil { pendingSpace = offset }
                } else {
                    if let space = pendingSpace {
                        units.append(0x20)
                        origin.append(space)
                        pendingSpace = nil
                    }
                    units.append(unit)
                    origin.append(offset)
                }
                offset += 1
            }
            string = NSString(characters: units, length: units.count)
            self.origin = origin
        }
    }
}

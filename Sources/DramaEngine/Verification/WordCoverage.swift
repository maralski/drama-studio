// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation
import NaturalLanguage

/// Compares an audio-only transcript with the requested words. No aggregate
/// error threshold: even one missing word, repetition or negation fails.
package enum WordCoverage {
    private struct ReactionAnnotation: Decodable {
        var cue_pattern: String
        var label_pattern: String
    }
    private struct EnglishForms: Decodable {
        var titles: [String: String]
        var contractions: [String: String]
        var reaction_pattern: String
        var reaction_words: Set<String>
        var reaction_annotations: [ReactionAnnotation]
    }
    private static let forms = Result { () throws -> EnglishForms in
        guard let url = EngineResources.bundle().url(forResource: "EnglishSpokenForms", withExtension: "json") else {
            throw SpeechVerificationError.unavailable("the word-normalization resource is missing")
        }
        return try JSONDecoder().decode(EnglishForms.self, from: Data(contentsOf: url))
    }
    private static var englishForms: EnglishForms { (try? forms.get()) ?? EnglishForms(titles: [:], contractions: [:], reaction_pattern: "(?!)", reaction_words: [], reaction_annotations: []) }
    package static func prepare() throws { _ = try forms.get() }

    package static func tokens(_ text: String, locale: Locale) -> [String] {
        var original = text
        if locale.language.languageCode?.identifier == "en" {
            for (title, spoken) in englishForms.titles {
                original = original.replacingOccurrences(of: "\\b" + title + "\\.?(?=\\s+\\p{L})", with: spoken, options: .regularExpression)
            }
        }
        var text = original.precomposedStringWithCompatibilityMapping
            .replacingOccurrences(of: "’", with: "'")
            .folding(options: [.caseInsensitive, .widthInsensitive], locale: locale)
        let number = NumberFormatter()
        number.locale = locale
        number.numberStyle = .spellOut
        if locale.language.languageCode?.identifier == "en" {
            text = text.folding(options: .diacriticInsensitive, locale: locale)
                .replacingOccurrences(of: #"\bcannot\b"#, with: "can not", options: .regularExpression)
            text = expandingOrdinals(text, formatter: number)
            let contractions = englishForms.contractions
            for (short, long) in contractions {
                text = text.replacingOccurrences(of: "\\b" + NSRegularExpression.escapedPattern(for: short) + "\\b",
                                                with: long, options: .regularExpression)
            }
            for (symbol, spoken) in [("-", "minus"), ("−", "minus"), ("+", "plus")] {
                text = text.replacingOccurrences(of: #"(?<![\p{L}\p{N}.])"# + NSRegularExpression.escapedPattern(for: symbol) + #"(?=\d|\.\d)"#,
                                                with: " \(spoken) ", options: .regularExpression)
            }
        }
        text = text.replacingOccurrences(of: "%", with: locale.language.languageCode?.identifier == "en" ? " percent " : " percent_symbol ")
        let digits = try! NSRegularExpression(pattern: #"(?<![\p{L}\p{N}.])[−-]?(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)(?![\p{L}\p{N}]|\.[0-9])"#)
        for match in digits.matches(in: text, range: NSRange(text.startIndex..., in: text)).reversed() {
            guard let range = Range(match.range, in: text) else { continue }
            let numeric = String(text[range]).replacingOccurrences(of: "−", with: "-")
            let value: NSNumber
            if let integer = Int64(numeric) { value = NSNumber(value: integer) }
            else {
                guard numeric.filter(\.isNumber).count <= 18 else { continue }
                value = NSDecimalNumber(string: numeric, locale: Locale(identifier: "en_US_POSIX"))
            }
            guard value.doubleValue.isFinite, let spoken = number.string(from: value) else { continue }
            text.replaceSubrange(range, with: spoken.replacingOccurrences(of: "-", with: " "))
        }
        text = text.replacingOccurrences(of: "-", with: " ")
        let tokenizer = NLTokenizer(unit: .word)
        tokenizer.setLanguage(NLLanguage(rawValue: locale.language.languageCode?.identifier ?? "und"))
        tokenizer.string = text
        var words: [String] = []
        tokenizer.enumerateTokens(in: text.startIndex..<text.endIndex) { range, _ in
            let word = String(text[range])
            if word.unicodeScalars.contains(where: CharacterSet.alphanumerics.contains) { words.append(word) }
            return true
        }
        return words
    }

    package static func matches(line: String, transcript: String, locale: Locale, uncertainSpans: [String] = []) -> Bool {
        let expected = tokens(DirectorsNote.spokenWords(line), locale: locale)
        let heard = recognizedTokens(transcript, line: line, locale: locale)
        let reactions = permitsReactions(line)
        for span in uncertainSpans {
            let words = recognizedTokens(span, line: line, locale: locale)
            let insertedReaction = reactions && words.allSatisfy(nonverbal)
                && words.allSatisfy { !expected.contains($0) }
            if !words.isEmpty && !insertedReaction { return false }
        }
        return match(expected: expected, heard: heard, reactions: reactions) != nil
    }

    package static func recognizedTokens(_ transcript: String, line: String, locale: Locale) -> [String] {
        var text = removingReactionAnnotations(transcript, for: line)
        if locale.language.languageCode?.identifier == "en" {
            text = expandingDottedClockTimes(text, line: line, locale: locale)
        }
        return tokens(text, locale: locale)
    }

    private static let ordinalNumber = try! NSRegularExpression(
        pattern: #"(?<![\p{L}\p{N}_.+−-])([1-9][0-9]{0,3})(st|nd|rd|th)(?![\p{L}\p{N}_]|\.[0-9])"#)
    private static let dottedClockTime = try! NSRegularExpression(
        pattern: #"(?<![\p{L}\p{N}_])at\s+([0-9]{1,2})\.([0-9]{2})(?![\p{L}\p{N}_]|\.[0-9]|\s*[%:+−–—/*×÷-]|\s+\.[0-9])"#,
        options: .caseInsensitive)

    private static func expandingOrdinals(_ source: String, formatter: NumberFormatter) -> String {
        var text = source
        for match in ordinalNumber.matches(in: text, range: NSRange(text.startIndex..., in: text)).reversed() {
            let ns = text as NSString
            guard let value = Int(ns.substring(with: match.range(at: 1))) else { continue }
            let suffix = (11...13).contains(value % 100) ? "th" : [1: "st", 2: "nd", 3: "rd"][value % 10] ?? "th"
            guard ns.substring(with: match.range(at: 2)) == suffix,
                  let spoken = formatter.string(from: NSNumber(value: value)),
                  let range = Range(match.range, in: text) else { continue }
            let prefix = text[..<range.lowerBound].trimmingCharacters(in: .whitespacesAndNewlines)
            guard !["+", "-", "−"].contains(prefix.last.map(String.init) ?? "") else { continue }
            var words = spoken.replacingOccurrences(of: "-", with: " ").split(separator: " ").map(String.init)
            guard let last = words.last else { continue }
            let irregular = ["one": "first", "two": "second", "three": "third", "five": "fifth",
                             "eight": "eighth", "nine": "ninth", "twelve": "twelfth"]
            words[words.count - 1] = irregular[last] ?? (last.hasSuffix("y") ? String(last.dropLast()) + "ieth" : last + "th")
            text.replaceSubrange(range, with: words.joined(separator: " "))
        }
        return text
    }

    private static func clockSpelling(_ match: NSTextCheckingResult, in text: String, formatter: NumberFormatter) -> String? {
        let ns = text as NSString
        guard let hour = Int(ns.substring(with: match.range(at: 1))), (1...23).contains(hour),
              let minute = Int(ns.substring(with: match.range(at: 2))), (10...59).contains(minute),
              let hours = formatter.string(from: NSNumber(value: hour)),
              let minutes = formatter.string(from: NSNumber(value: minute)) else { return nil }
        return "at " + hours + " " + minutes
    }

    private static func expandingDottedClockTimes(_ text: String, line: String, locale: Locale) -> String {
        let candidates = dottedClockTime.matches(in: text, range: NSRange(text.startIndex..., in: text))
        guard !candidates.isEmpty else { return text }
        let expected = tokens(DirectorsNote.spokenWords(line), locale: locale)
        let formatter = NumberFormatter()
        formatter.locale = locale; formatter.numberStyle = .spellOut
        var result = "", cursor = text.startIndex
        for candidate in candidates {
            guard let spoken = clockSpelling(candidate, in: text, formatter: formatter),
                  let range = Range(candidate.range, in: text) else { continue }
            result += text[cursor..<range.lowerBound]
            // A dot remains a decimal unless the requested words at this exact
            // position spell the same clock time. Earlier decimals stay intact.
            let prefix = match(expected: expected, heard: tokens(result, locale: locale),
                               reactions: permitsReactions(line), complete: false)
            if let prefix, expected.dropFirst(prefix.count).starts(with: tokens(spoken, locale: locale)) {
                result += spoken
            } else {
                result += text[range]
            }
            cursor = range.upperBound
        }
        return result + text[cursor...]
    }

    package static func unambiguousHeardOrdinal(line: String, transcript: String, expectedOrdinal: Int, locale: Locale) -> Int? {
        let expected = tokens(DirectorsNote.spokenWords(line), locale: locale)
        let reactions = permitsReactions(line)
        guard expected.indices.contains(expectedOrdinal),
              !(reactions && nonverbal(expected[expectedOrdinal])),
              let ordinals = match(expected: expected, heard: recognizedTokens(transcript, line: line, locale: locale), reactions: reactions) else { return nil }
        return ordinals[expectedOrdinal]
    }

    private static func permitsReactions(_ line: String) -> Bool {
        line.range(of: #"\[(?:laugh[^\]]*|chuckle[^\]]*|sigh[^\]]*|gasp[^\]]*|cough[^\]]*|clears throat|hesitates|pause)\]"#,
                   options: [.regularExpression, .caseInsensitive]) != nil
    }

    private static func match(expected: [String], heard: [String], reactions: Bool, complete: Bool = true) -> [Int]? {
        var next = 0
        var ordinals: [Int] = []
        for (ordinal, word) in heard.enumerated() {
            if next < expected.count, word == expected[next] { next += 1; ordinals.append(ordinal) }
            else if reactions && nonverbal(word) { continue }
            else { return nil }
        }
        return !expected.isEmpty && (!complete || next == expected.count) ? ordinals : nil
    }

    private static func removingReactionAnnotations(_ transcript: String, for line: String) -> String {
        var text = transcript
        for annotation in englishForms.reaction_annotations
        where line.range(of: annotation.cue_pattern, options: [.regularExpression, .caseInsensitive]) != nil {
            let label = "(?:\(annotation.label_pattern))"
            let pattern = "\\(\\s*\(label)\\s*\\)|\\[\\s*\(label)\\s*\\]|\\*\\s*\(label)\\s*\\*"
            text = text.replacingOccurrences(of: pattern, with: " ", options: [.regularExpression, .caseInsensitive])
        }
        return text
    }

    private static func nonverbal(_ word: String) -> Bool {
        englishForms.reaction_words.contains(word)
            || word.range(of: englishForms.reaction_pattern, options: .regularExpression) != nil
    }
}

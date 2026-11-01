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
        if locale.language.languageCode?.identifier == "en" {
            text = text.folding(options: .diacriticInsensitive, locale: locale)
                .replacingOccurrences(of: #"\bcannot\b"#, with: "can not", options: .regularExpression)
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
        let number = NumberFormatter()
        number.locale = locale
        number.numberStyle = .spellOut
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
        tokens(removingReactionAnnotations(transcript, for: line), locale: locale)
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

    private static func match(expected: [String], heard: [String], reactions: Bool) -> [Int]? {
        var next = 0
        var ordinals: [Int] = []
        for (ordinal, word) in heard.enumerated() {
            if next < expected.count, word == expected[next] { next += 1; ordinals.append(ordinal) }
            else if reactions && nonverbal(word) { continue }
            else { return nil }
        }
        return !expected.isEmpty && next == expected.count ? ordinals : nil
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

// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation
import NaturalLanguage
import Synchronization

/// Compares an audio-only transcript with the requested words. No aggregate
/// error threshold: even one missing word, repetition or negation fails.
package enum WordCoverage {
    /// How a recognizer writes the numbers it hears. The script has no profile.
    package enum RecognizerProfile: Sendable {
        /// Keeps what was said: "one thirty" is 1.30 and "one hundred thirty" is 130.
        case exact
        /// Apple's SpeechTranscriber writes both as 130, and "two point five
        /// million" as 25 million, so its numbers also get those readings.
        case apple
    }

    struct ReactionAnnotation: Decodable {
        var cue_pattern: String
        var label_pattern: String
    }
    struct EnglishForms: Decodable {
        var titles: [String: String] = [:]
        var contractions: [String: String] = [:]
        var reaction_pattern = "(?!)"
        var reaction_words: Set<String> = []
        var reaction_annotations: [ReactionAnnotation] = []
        var spellings: [String: String] = [:]
        /// What a hyphen between two numbers may be read as.
        var range_words: [String] = []
        /// A unit right after a number, abbreviated or spelled out, singular or
        /// plural, as one singular name: "kg" and "kilograms" are "kilogram".
        var units: [String: String] = [:]
        /// Unit symbols that are not words, as the unit words they stand for.
        var unit_symbols: [String: String] = [:]
    }
    private static let forms = Result { () throws -> EnglishForms in
        guard let url = EngineResources.bundle().url(forResource: "EnglishSpokenForms", withExtension: "json") else {
            throw SpeechVerificationError.unavailable("the word-normalization resource is missing")
        }
        return try JSONDecoder().decode(EnglishForms.self, from: Data(contentsOf: url))
    }
    private static var englishForms: EnglishForms { (try? forms.get()) ?? EnglishForms() }
    package static func prepare() throws { _ = try forms.get() }

    private static func alternation(_ words: some Collection<String>) -> String {
        words.sorted { $0.count > $1.count }.map(NSRegularExpression.escapedPattern(for:)).joined(separator: "|")
    }
    private static let titlePattern = try! NSRegularExpression(
        pattern: "\\b(" + alternation(englishForms.titles.keys) + ")\\.?(?=\\s+\\p{L})")
    private static let contractionPattern = try! NSRegularExpression(
        pattern: "\\b(" + alternation(englishForms.contractions.keys) + ")\\b")
    private static let cannot = try! NSRegularExpression(pattern: #"\bcannot\b"#)
    /// The apostrophe of a possessive is silent: neighbor's, neighbors, neighbors'.
    private static let possessive = try! NSRegularExpression(pattern: #"(?<=\p{L})'(?=s(?!\p{L}))|(?<=\p{L}s)'(?!\p{L})"#)
    /// A sign before a number, or between two numbers with a space each side.
    private static let signs = [("-", "minus"), ("−", "minus"), ("+", "plus")].map { symbol, spoken in
        let symbol = NSRegularExpression.escapedPattern(for: symbol)
        return (try! NSRegularExpression(pattern: #"(?<![\p{L}\p{N}.])"# + symbol + #"(?=\d|\.\d)|(?<=\d )"# + symbol + #"(?= \d)"#),
                " \(spoken) ")
    }
    private static let unitSymbol = try! NSRegularExpression(pattern: "(" + alternation(englishForms.unit_symbols.keys) + ")")
    private static let unitAfterDigits = try! NSRegularExpression(
        pattern: #"(?<=[0-9])(?=(?:"# + alternation(englishForms.units.keys.filter { $0.count > 1 }) + #")(?!\p{L}))"#)
    private static let digitsAfterLetters = try! NSRegularExpression(pattern: #"(?<=\p{L})(?=[0-9])"#)
    private static let digits = try! NSRegularExpression(pattern: #"(?<![\p{L}\p{N}.])[−-]?(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)(?![\p{L}\p{N}]|\.[0-9])"#)
    private static let writtenNumber = try! NSRegularExpression(
        pattern: #"(?<![\p{L}\p{N}_.:])(?:([0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:([.:])([0-9]+))?|\.([0-9]+))('?s)?(?![\p{L}\p{N}_]|[.:][0-9]|,[0-9]{3}(?![0-9]))"#)
    private static let wordHyphen = try! NSRegularExpression(pattern: #"-(?![0-9])|(?<![0-9])-"#)
    private static let spellOutFormatters = Mutex<[String: NumberFormatter]>([:])

    private static func spellOut(_ value: NSNumber, locale: Locale) -> String? {
        spellOutFormatters.withLock { cache in
            let formatter = cache[locale.identifier] ?? {
                let formatter = NumberFormatter()
                formatter.locale = locale
                formatter.numberStyle = .spellOut
                cache[locale.identifier] = formatter
                return formatter
            }()
            return formatter.string(from: value)
        }
    }

    private static func replacing(_ regex: NSRegularExpression, in text: String, with replacement: (String) -> String?) -> String {
        var result = text
        for match in regex.matches(in: text, range: NSRange(text.startIndex..., in: text)).reversed() {
            guard let range = Range(match.range(at: match.numberOfRanges > 1 ? 1 : 0), in: text),
                  let whole = Range(match.range, in: result),
                  let spoken = replacement(String(text[range])) else { continue }
            result.replaceSubrange(whole, with: spoken)
        }
        return result
    }

    package static func tokens(_ text: String, locale: Locale) -> [String] {
        tokens(text, locale: locale, profile: nil)
    }

    package static func tokens(_ text: String, locale: Locale, profile: RecognizerProfile) -> [String] {
        tokens(text, locale: locale, profile: .some(profile))
    }

    private static func tokens(_ text: String, locale: Locale, profile: RecognizerProfile?) -> [String] {
        let english = locale.language.languageCode?.identifier == "en"
        var original = text
        if english {
            original = replacing(titlePattern, in: original) { englishForms.titles[$0] }
        }
        var text = original.precomposedStringWithCompatibilityMapping
            .replacingOccurrences(of: "’", with: "'")
            .folding(options: [.caseInsensitive, .widthInsensitive], locale: locale)
        if english {
            text = text.folding(options: .diacriticInsensitive, locale: locale)
            text = cannot.stringByReplacingMatches(in: text, range: NSRange(text.startIndex..., in: text), withTemplate: "can not")
            text = expandingOrdinals(text, locale: locale)
            text = replacing(contractionPattern, in: text) { englishForms.contractions[$0] }
            text = possessive.stringByReplacingMatches(in: text, range: NSRange(text.startIndex..., in: text), withTemplate: "")
            for (symbol, spoken) in signs {
                text = symbol.stringByReplacingMatches(in: text, range: NSRange(text.startIndex..., in: text),
                                                       withTemplate: NSRegularExpression.escapedTemplate(for: spoken))
            }
            text = digitsAfterLetters.stringByReplacingMatches(in: text, range: NSRange(text.startIndex..., in: text), withTemplate: " ")
        }
        if english {
            text = text.replacingOccurrences(of: "%", with: " percent ")
            text = replacing(unitSymbol, in: text) { englishForms.unit_symbols[$0].map { " \($0) " } }
            text = unitAfterDigits.stringByReplacingMatches(in: text, range: NSRange(text.startIndex..., in: text), withTemplate: " ")
            text = wordHyphen.stringByReplacingMatches(in: text, range: NSRange(text.startIndex..., in: text), withTemplate: " ")
            return EnglishNumbers.tokens(englishPieces(text, profile: profile), forms: englishForms, profile: profile)
        }
        text = text.replacingOccurrences(of: "%", with: " percent_symbol ")
        for match in digits.matches(in: text, range: NSRange(text.startIndex..., in: text)).reversed() {
            guard let range = Range(match.range, in: text) else { continue }
            let numeric = String(text[range]).replacingOccurrences(of: "−", with: "-")
            let value: NSNumber
            if let integer = Int64(numeric) { value = NSNumber(value: integer) }
            else {
                guard numeric.filter(\.isNumber).count <= 18 else { continue }
                value = NSDecimalNumber(string: numeric, locale: Locale(identifier: "en_US_POSIX"))
            }
            guard value.doubleValue.isFinite, let spoken = spellOut(value, locale: locale) else { continue }
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

    private static func englishPieces(_ text: String, profile: RecognizerProfile?) -> [(piece: EnglishNumbers.Piece, gap: EnglishNumbers.Gap)] {
        var numbers: [(range: Range<String.Index>, piece: EnglishNumbers.Piece)] = []
        for match in writtenNumber.matches(in: text, range: NSRange(text.startIndex..., in: text)) {
            func group(_ index: Int) -> String? { Range(match.range(at: index), in: text).map { String(text[$0]) } }
            guard let range = Range(match.range, in: text) else { continue }
            let currency = text[..<range.lowerBound].last.flatMap { EnglishNumbers.currencySymbols[$0] }
                ?? text[range.upperBound...].first.flatMap { EnglishNumbers.currencySymbols[$0] }
            guard let piece = EnglishNumbers.written(integer: group(1), separator: group(2) ?? group(4).map { _ in "." },
                                                     fraction: group(3) ?? group(4), decade: group(5) != nil,
                                                     currency: currency, profile: profile) else { continue }
            numbers.append((range, piece))
        }
        var found = numbers
        let tokenizer = NLTokenizer(unit: .word)
        tokenizer.setLanguage(.english)
        tokenizer.string = text
        tokenizer.enumerateTokens(in: text.startIndex..<text.endIndex) { range, _ in
            let word = text[range]
            if word.unicodeScalars.contains(where: CharacterSet.alphanumerics.contains),
               !numbers.contains(where: { $0.range.overlaps(range) }) {
                found.append((range, .word(String(word))))
            }
            return true
        }
        found.sort { $0.range.lowerBound < $1.range.lowerBound }
        return found.indices.map { index in
            let between = index > 0 ? text[found[index - 1].range.upperBound..<found[index].range.lowerBound] : "."
            let gap: EnglishNumbers.Gap = between.allSatisfy(\.isWhitespace) ? .space
                : between.allSatisfy({ $0.isWhitespace || $0 == "," }) ? .comma
                : between == "-" || between == "–" ? .hyphen : .other
            return (found[index].piece, gap)
        }
    }

    package static func isNumber(_ token: String) -> Bool { EnglishNumbers.isNumber(token) }

    package static func matches(line: String, transcript: String, locale: Locale, profile: RecognizerProfile,
                                uncertainSpans: [String] = []) -> Bool {
        let expected = tokens(DirectorsNote.spokenWords(line), locale: locale)
        let heard = recognizedTokens(transcript, line: line, locale: locale, profile: profile)
        let reactions = permitsReactions(line)
        for span in uncertainSpans {
            let words = recognizedTokens(span, line: line, locale: locale, profile: profile)
            let insertedReaction = reactions && words.allSatisfy(nonverbal)
                && words.allSatisfy { !expected.contains($0) }
            if !words.isEmpty && !insertedReaction { return false }
        }
        return match(expected: expected, heard: heard, reactions: reactions) != nil
    }

    package static func recognizedTokens(_ transcript: String, line: String, locale: Locale, profile: RecognizerProfile) -> [String] {
        tokens(removingReactionAnnotations(transcript, for: line), locale: locale, profile: profile)
    }

    private static let ordinalNumber = try! NSRegularExpression(
        pattern: #"(?<![\p{L}\p{N}_.+−-])([1-9][0-9]{0,3})(st|nd|rd|th)(?![\p{L}\p{N}_]|\.[0-9])"#)

    private static func expandingOrdinals(_ source: String, locale: Locale) -> String {
        var text = source
        for match in ordinalNumber.matches(in: text, range: NSRange(text.startIndex..., in: text)).reversed() {
            let ns = text as NSString
            guard let value = Int(ns.substring(with: match.range(at: 1))) else { continue }
            let suffix = (11...13).contains(value % 100) ? "th" : [1: "st", 2: "nd", 3: "rd"][value % 10] ?? "th"
            guard ns.substring(with: match.range(at: 2)) == suffix,
                  let spoken = spellOut(NSNumber(value: value), locale: locale),
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

    package static func unambiguousHeardOrdinal(line: String, transcript: String, expectedOrdinal: Int, locale: Locale,
                                                profile: RecognizerProfile) -> Int? {
        let expected = tokens(DirectorsNote.spokenWords(line), locale: locale)
        let reactions = permitsReactions(line)
        guard expected.indices.contains(expectedOrdinal),
              !(reactions && nonverbal(expected[expectedOrdinal])),
              let ordinals = match(expected: expected, heard: recognizedTokens(transcript, line: line, locale: locale, profile: profile),
                                   reactions: reactions) else { return nil }
        return ordinals[expectedOrdinal]
    }

    private static func permitsReactions(_ line: String) -> Bool {
        line.range(of: #"\[(?:laugh[^\]]*|chuckle[^\]]*|sigh[^\]]*|gasp[^\]]*|cough[^\]]*|clears throat|hesitates|pause)\]"#,
                   options: [.regularExpression, .caseInsensitive]) != nil
    }

    /// Returns, for each expected token, the ordinal of the heard token that
    /// matched it. Spacing may differ inside a compound: "web site" == "website",
    /// and punctuation inside a run of digits: "three two one" == "3, 2, 1".
    /// A currency only a symbol implies may be absent from the other side, but
    /// never replaced by a different currency.
    private static func match(expected: [String], heard: [String], reactions: Bool) -> [Int]? {
        func joinable(_ tokens: [String], _ index: Int) -> Bool {
            index + 1 < tokens.count && !isNumber(tokens[index]) && !isNumber(tokens[index + 1])
        }
        func optional(_ tokens: [String], _ index: Int, against other: String?) -> Bool {
            index < tokens.count && EnglishNumbers.isImpliedCurrency(tokens[index]) && !(other.map(EnglishNumbers.isCurrency) ?? false)
        }
        var next = 0, ordinal = 0
        var ordinals: [Int] = []
        while ordinal < heard.count {
            let word = heard[ordinal]
            if next < expected.count, EnglishNumbers.same(word, expected[next]) {
                ordinals.append(ordinal); next += 1; ordinal += 1
            } else if next < expected.count, joinable(expected, next), word == expected[next] + expected[next + 1] {
                ordinals += [ordinal, ordinal]; next += 2; ordinal += 1
            } else if next < expected.count, joinable(heard, ordinal), word + heard[ordinal + 1] == expected[next] {
                ordinals.append(ordinal); next += 1; ordinal += 2
            } else if next < expected.count, let count = EnglishNumbers.digitRun(word, spelledBy: expected[next...]) {
                ordinals += Array(repeating: ordinal, count: count); next += count; ordinal += 1
            } else if next < expected.count, let count = EnglishNumbers.digitRun(expected[next], spelledBy: heard[ordinal...]) {
                ordinals.append(ordinal); next += 1; ordinal += count
            } else if optional(expected, next, against: word) {
                ordinals.append(ordinal); next += 1
            } else if optional(heard, ordinal, against: next < expected.count ? expected[next] : nil) {
                ordinal += 1
            } else if reactions && nonverbal(word) {
                ordinal += 1
            } else {
                return nil
            }
        }
        while optional(expected, next, against: nil) {
            ordinals.append(ordinal); next += 1
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

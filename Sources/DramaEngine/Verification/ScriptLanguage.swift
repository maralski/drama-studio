// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation
import NaturalLanguage

package enum ScriptLanguage {
    package static func identify(_ text: String) throws -> String {
        let spoken = DirectorsNote.spokenWords(text)
        let segments = DirectorsNote.sentences(spoken).filter { text in
            let tokenizer = NLTokenizer(unit: .word)
            tokenizer.string = text
            var count = 0
            tokenizer.enumerateTokens(in: text.startIndex..<text.endIndex) { _, _ in count += 1; return count < 4 }
            return count >= 4
        }
        return try choose(overall: evidence(spoken), segments: segments.map(evidence))
    }

    private static func evidence(_ text: String) -> [String: Double] {
        let detector = NLLanguageRecognizer()
        detector.processString(text)
        return Dictionary(uniqueKeysWithValues: detector.languageHypotheses(withMaximum: 3).map { ($0.key.rawValue, $0.value) })
    }

    package static func preferredLocale(language: String, installed: [Locale], preferences: [String]) -> Locale? {
        let requested = Locale(identifier: language).language
        let candidates = installed.filter {
            $0.language.languageCode == requested.languageCode
                && (requested.script == nil || Locale.Language(identifier: $0.language.maximalIdentifier).script == requested.script)
        }
        for preference in preferences {
            let id = Locale(identifier: preference).identifier(.bcp47)
            if let exact = candidates.first(where: { $0.identifier(.bcp47) == id }) { return exact }
        }
        return candidates.sorted { $0.identifier(.bcp47) < $1.identifier(.bcp47) }.first
    }

    package static func choose(overall: [String: Double], segments: [[String: Double]]) throws -> String {
        let strong = Set(segments.compactMap { scores -> String? in
            guard let best = scores.max(by: { $0.value < $1.value }), best.value >= 0.8 else { return nil }
            return best.key
        })
        guard strong.count <= 1 else {
            throw SpeechVerificationError.unavailable("this script mixes languages; generate each language separately")
        }
        guard let best = overall.max(by: { $0.value < $1.value }), best.value >= 0.75, best.key != "und",
              strong.isEmpty || strong.contains(best.key) else {
            throw SpeechVerificationError.unavailable("the script's language is uncertain; use a complete sentence")
        }
        return best.key
    }

    package static func requireLocale(language: String, recognizerAvailable: Bool, locale: Locale?, installed: [Locale]) throws -> Locale {
        guard recognizerAvailable else { throw SpeechVerificationError.unavailable("on-device recognition is unavailable on this Mac") }
        guard let locale else {
            throw SpeechVerificationError.unavailable("on-device recognition does not support this script's language (\(language))")
        }
        let exact = installed.first { $0.identifier(.bcp47) == locale.identifier(.bcp47) }
        let sameLanguage = installed.first { $0.language.languageCode == locale.language.languageCode }
        guard let installedLocale = exact ?? sameLanguage else {
            throw SpeechVerificationError.unavailable("the Mac has no installed on-device speech model for \(locale.identifier)")
        }
        return installedLocale
    }
}

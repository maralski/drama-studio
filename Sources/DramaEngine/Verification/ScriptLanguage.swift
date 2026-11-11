// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation
import NaturalLanguage

package enum ScriptLanguage {
    /// The language of the user's macOS, which a line too short to tell is read in.
    package static var userLanguage: String {
        let language = Locale.Language(identifier: Locale.preferredLanguages.first ?? "en")
        let code = language.languageCode?.identifier ?? "en"
        return language.script.map { "\(code)-\($0.identifier)" } ?? code
    }

    /// The language a request is spoken and checked in: the one it names,
    /// else the one its text is written in.
    package static func language(of request: SpeechRequest) throws -> String {
        try request.language ?? identify(request.text, fallback: userLanguage)
    }

    /// The language `text` is written in. `fallback` answers for text the
    /// recognizer cannot tell in a script that language is written in: an
    /// uncertain result, or a line of under four words.
    package static func identify(_ text: String, fallback: String? = nil) throws -> String {
        let spoken = DirectorsNote.spokenWords(text)
        let enough = 4
        func words(_ text: String) -> Int {
            let tokenizer = NLTokenizer(unit: .word)
            tokenizer.string = text
            var count = 0
            tokenizer.enumerateTokens(in: text.startIndex..<text.endIndex) { _, _ in count += 1; return count < enough }
            return count
        }
        let segments = DirectorsNote.sentences(spoken).filter { words($0) >= enough }
        return try choose(overall: evidence(spoken), segments: segments.map(evidence), short: words(spoken) < enough, fallback: fallback)
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

    /// On 164 everyday English lines of one to three words the recognizer
    /// named another language at 0.75 or more once ("Hi." as Catalan at 0.80)
    /// and never at 0.9; on 60 French, Spanish and German ones it named a
    /// wrong language at up to 0.93 and never at 0.95.
    static let shortLineConfidence = 0.95

    package static func choose(overall: [String: Double], segments: [[String: Double]], short: Bool = false, fallback: String? = nil) throws -> String {
        let strong = Set(segments.compactMap { scores -> String? in
            guard let best = scores.max(by: { $0.value < $1.value }), best.value >= 0.8 else { return nil }
            return best.key
        })
        guard strong.count <= 1 else {
            throw SpeechVerificationError.language("this script mixes languages; generate each language separately")
        }
        let best = overall.max(by: { $0.value < $1.value })
        let confident = best.map { $0.value >= 0.75 && $0.key != "und" && (strong.isEmpty || strong.contains($0.key)) } ?? false
        if let fallback, strong.isEmpty {
            let sameScript = best.map { $0.key == "und" || script($0.key) == script(fallback) } ?? true
            let sure = confident && !(short && best!.value < shortLineConfidence)
            if sameScript && !sure { return fallback }
        }
        guard let best, confident else {
            throw SpeechVerificationError.language("the script's language is uncertain; use a complete sentence")
        }
        return best.key
    }

    private static func script(_ language: String) -> String? {
        Locale.Language(identifier: Locale.Language(identifier: language).maximalIdentifier).script?.identifier
    }

    /// The installed Apple speech model for `locale`'s language, or what this
    /// Mac lacks: the recognizer itself, or a model for the language, which
    /// covers one Apple has none for and one it has not downloaded.
    package static func appleModel(recognizerAvailable: Bool, locale: Locale?, installed: [Locale]) -> AppleRecognizer {
        guard recognizerAvailable else { return .missing(.recognizer) }
        guard let locale, let model = installed.first(where: { $0.identifier(.bcp47) == locale.identifier(.bcp47) })
                ?? installed.first(where: { $0.language.languageCode == locale.language.languageCode }) else {
            return .missing(.model)
        }
        return .model(model)
    }
}

/// Apple's half of the word check: the model it uses, or what this Mac lacks.
package enum AppleRecognizer: Equatable, Sendable {
    case model(Locale)
    case missing(SpeechNotice.AppleSpeech)
}

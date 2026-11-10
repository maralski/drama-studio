// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation
internal import MingTTS

/// The one table from Drama Studio's cues to what Ming performs. Ming reads
/// 14 fixed tags as plain text in the prompt and takes one style per
/// request, so a reaction cue becomes a tag, an opening style becomes an
/// emotion, and everything else is removed from the text, its words kept,
/// and reported as a ``SpeechNotice``.
package enum MingCues {
    /// Ming's tags, exact spelling. Ming's text frontend passes these through untouched.
    package static let tags = ["[Breathing]", "[Cough]", "[Sigh]", "[Laughter]", "[Surprise-ah]", "[Surprise-oh]", "[Dissatisfaction-hnn]",
                               "[Uhm]", "[Confirmation-en]", "[Question-ei]", "[Question-ah]", "[Question-en]", "[Question-oh]", "[Question-huh]"]
    static let builtInVoices = MingVoice.catalog.map(\.ip)
    static let languages = ["zh", "en", "ja", "ko", "de", "fr", "es"]

    /// The code in ``languages`` for a language ``ScriptLanguage`` identified,
    /// without its script or region: `zh` for `zh-Hant`. Nil for a language
    /// Ming does not speak.
    package static func languageCode(_ identified: String) -> String? {
        let code = Locale.Language(identifier: identified).languageCode?.identifier ?? identified
        return languages.contains(code) ? code : nil
    }

    /// Throws for a script in a language Ming does not speak, naming the ones it does.
    /// A script whose language is uncertain passes; Ming reads it as English.
    static func checkLanguage(of text: String) throws {
        guard let identified = try? ScriptLanguage.identify(text), languageCode(identified) == nil else { return }
        let english = Locale(identifier: "en")
        let name = { (code: String) in english.localizedString(forLanguageCode: code) ?? code }
        throw PromptError.engineUnsupported(.ming, "speak \(name(identified)); it speaks \(languages.map(name).formatted(.list(type: .and)))")
    }

    /// Ming's emotion control, named for people and for ``CueSupport/style(_:)``.
    package enum Emotion: String, Sendable, Equatable, CaseIterable {
        case happy, sad, angry, fearful, surprised
    }

    package enum Meaning: Sendable, Equatable {
        case tag(String)
        case emotion(Emotion)
    }

    /// Keys are a cue's text without brackets, lowercased, white space collapsed.
    package static let table: [String: Meaning] = [
        "laughs": .tag("[Laughter]"), "laughing": .tag("[Laughter]"), "laughter": .tag("[Laughter]"), "laugh": .tag("[Laughter]"),
        "chuckles": .tag("[Laughter]"), "chuckle": .tag("[Laughter]"), "giggles": .tag("[Laughter]"),
        "sighs": .tag("[Sigh]"), "sigh": .tag("[Sigh]"),
        "coughs": .tag("[Cough]"), "cough": .tag("[Cough]"), "clears throat": .tag("[Cough]"),
        "breathes": .tag("[Breathing]"), "breathing": .tag("[Breathing]"), "breath": .tag("[Breathing]"), "breathes heavily": .tag("[Breathing]"),
        "panting": .tag("[Breathing]"), "pants": .tag("[Breathing]"),
        "gasps": .tag("[Surprise-ah]"), "gasp": .tag("[Surprise-ah]"),
        "hesitates": .tag("[Uhm]"), "um": .tag("[Uhm]"), "uhm": .tag("[Uhm]"), "hmm": .tag("[Dissatisfaction-hnn]"),
        "with bright eager excitement": .emotion(.happy), "heavy, deflated sadness": .emotion(.sad),
        "furious, barely holding it together": .emotion(.angry), "breathless and terrified": .emotion(.fearful),
    ]

    /// Words in a director's note that name one of Ming's emotions.
    static let noteWords: [String: Emotion] = [
        "happy": .happy, "happier": .happy, "cheerful": .happy, "excited": .happy, "joyful": .happy,
        "sad": .sad, "sadder": .sad, "sorrowful": .sad, "mournful": .sad,
        "angry": .angry, "angrier": .angry, "furious": .angry, "enraged": .angry,
        "scared": .fearful, "afraid": .fearful, "terrified": .fearful, "fearful": .fearful, "frightened": .fearful,
        "surprised": .surprised, "astonished": .surprised, "shocked": .surprised,
    ]

    static let bracketCue = try! NSRegularExpression(pattern: #"\[[^\[\]]*\]"#)
    static let passageTag = try! NSRegularExpression(pattern: #"</?([A-Za-z][^<>\s]*)[^<>]*>"#)
    private static let tagInners = Set(tags.map { String($0.dropFirst().dropLast()) })

    /// How Ming performs `cue`, written as typed: `[laughs]`, `[laughs nervously]`, `<whisper>`.
    package static func support(_ cue: String) -> CueSupport {
        let cue = UnicodeWhitespace.strip(cue)
        if cue.hasPrefix("["), cue.hasSuffix("]") {
            guard let (meaning, _) = lookup(String(cue.dropFirst().dropLast())) else { return .unsupported }
            switch meaning {
            case .tag(let tag): return .tag(tag)
            case .emotion(let emotion): return .style(emotion.rawValue)
            }
        }
        return .unsupported
    }

    /// The meaning of a cue's inner text, and whether only its leading words matched.
    static func lookup(_ inner: String) -> (Meaning, approximated: Bool)? {
        let trimmed = UnicodeWhitespace.strip(inner)
        if tagInners.contains(trimmed) { return (.tag("[\(trimmed)]"), false) }
        let words = trimmed.lowercased().split(whereSeparator: { $0.isWhitespace }).map(String.init)
        guard !words.isEmpty else { return nil }
        if let meaning = table[words.joined(separator: " ")] { return (meaning, false) }
        for count in stride(from: words.count - 1, through: 1, by: -1) {
            if let meaning = table[words.prefix(count).joined(separator: " ")] { return (meaning, true) }
        }
        return nil
    }

    /// Where a line's emotion came from, so a line whose voice cannot take a style can name what it lost.
    package enum Style: Sendable, Equatable {
        case note(String, Emotion)
        case cue(String, Emotion)

        package var emotion: Emotion {
            switch self {
            case .note(_, let emotion), .cue(_, let emotion): emotion
            }
        }
    }

    /// One sentence as Ming speaks it.
    package struct Line: Sendable, Equatable {
        /// The text with Ming tags in place of cues, no passage or speaker tags.
        package var text: String
        /// The style in force for this line: the note's, or the last style cue's at or before it.
        package var style: Style?
        /// The `<|speaker:N|>` the line led with, or 0.
        package var speaker: Int
    }

    package struct Translation: Sendable, Equatable {
        package var lines: [Line]
        package var notices: [SpeechNotice]
    }

    /// Rewrites each sentence for Ming. Pure. A style cue holds for the
    /// sentences after it, as an opening style does in a script; a note's
    /// emotion wins over cues.
    package static func translate(lines: [String], note: String?) -> Translation {
        var notices: [SpeechNotice] = []
        func report(_ notice: SpeechNotice) { if !notices.contains(notice) { notices.append(notice) } }
        var style: Style?
        if let note = note.flatMap(DirectorsNote.clean) {
            if let emotion = note.lowercased().split(whereSeparator: { !$0.isLetter }).lazy.compactMap({ noteWords[String($0)] }).first {
                style = .note(note, emotion)
            } else {
                report(.unsupportedNote(note))
            }
        }
        let noteStyle = style
        var out: [Line] = []
        for line in lines {
            var body = line
            var speaker = 0
            let source = body as NSString
            if let tag = Drama3Prompt.speakerTag.firstMatch(in: body, options: .anchored, range: NSRange(location: 0, length: source.length)) {
                speaker = Int(source.substring(with: tag.range(at: 1))) ?? 0
                body = source.substring(from: NSMaxRange(tag.range))
            }
            var text = ""
            var cursor = 0
            var styledHere = false
            let ns = body as NSString
            for match in bracketCue.matches(in: body, range: NSRange(location: 0, length: ns.length)) {
                text += ns.substring(with: NSRange(location: cursor, length: match.range.location - cursor))
                cursor = NSMaxRange(match.range)
                let cue = ns.substring(with: match.range)
                guard let (meaning, approximated) = lookup(String(cue.dropFirst().dropLast())) else {
                    report(.unsupportedCue(cue))
                    text += " "
                    continue
                }
                switch meaning {
                case .tag(let tag):
                    if approximated { report(.approximatedCue(cue, as: tag)) }
                    text += " \(tag) "
                case .emotion(let next):
                    if noteStyle != nil || styledHere {
                        report(.unsupportedCue(cue))
                    } else {
                        if approximated { report(.approximatedCue(cue, as: next.rawValue)) }
                        style = .cue(cue, next)
                        styledHere = true
                    }
                    text += " "
                }
            }
            text += ns.substring(from: cursor)
            let tagged = text as NSString
            for match in passageTag.matches(in: text, range: NSRange(location: 0, length: tagged.length)).reversed() {
                if !tagged.substring(with: match.range).hasPrefix("</") { report(.unsupportedCue("<\(tagged.substring(with: match.range(at: 1)))>")) }
                text = (text as NSString).replacingCharacters(in: match.range, with: " ")
            }
            out.append(Line(text: tidy(text), style: style, speaker: speaker))
        }
        return Translation(lines: out, notices: notices)
    }

    static func tidy(_ text: String) -> String {
        DirectorsNote.collapseWhitespace(text).replacingOccurrences(of: #"\s+([,!?;:…]|\.(?!\d))"#, with: "$1", options: .regularExpression)
    }
}

// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation
internal import MingTTS

/// Ming's built-in voices, from the MingTTS catalog. A name is the voice's
/// `ip` in the model's own instruction, so it is passed to Ming as written.
package enum MingVoices {
    package static let names = MingVoice.catalog.map(\.ip)
    /// The voice Ming speaks in when a request names none.
    package static let defaultName = MingVoice.emma.ip

    /// The emotion a `<voice>_<emotion>` name ends in, in English.
    static let emotions = ["中性": "neutral", "高兴": "happy", "悲伤": "sad", "生气": "angry", "恐惧": "fearful", "惊讶": "surprised", "厌恶": "disgusted"]

    /// Where a name sits in a voice list. The catalog writes English voices
    /// as `US_Emma`, a Chinese voice recorded in one emotion as
    /// `<voice>_<emotion>`, and a character from a show as `<show>_<character>`.
    static func group(of name: String) -> String {
        let parts = name.split(separator: "_", maxSplits: 1)
        guard parts.count == 2 else { return "Chinese" }
        if parts[0].allSatisfy(\.isASCII) { return "English" }
        return emotions[String(parts[1])] == nil ? "Chinese characters" : "Chinese, in one emotion"
    }

    /// What a reader of English can't tell from the name: the emotion of a `<voice>_<emotion>` voice.
    static func note(on name: String) -> String? {
        name.split(separator: "_", maxSplits: 1).last.flatMap { emotions[String($0)] }
    }
}

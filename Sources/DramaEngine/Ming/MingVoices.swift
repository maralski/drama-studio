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

    /// The longest clip Ming clones well, in seconds.
    package static let maxCloneSeconds = MingReferenceVoice.recommendedMaxSeconds

    /// Where a name sits in a voice list.
    static func group(of name: String) -> String {
        switch MingVoice(ip: name).kind {
        case .english: "English"
        case .chinese: "Chinese"
        case .chineseEmotion: "Chinese, in one emotion"
        case .chineseCharacter: "Chinese characters"
        }
    }

    /// What a reader of English can't tell from the name: the emotion of a `<voice>_<emotion>` voice.
    static func note(on name: String) -> String? {
        MingVoice(ip: name).emotion.map { "\($0)" }
    }
}

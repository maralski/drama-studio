// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import DramaEngine
import Foundation

/// What speaks for one speaker.
enum VoiceChoice: Hashable {
    case modelDefault
    /// A voice the model picks; its first take becomes the session voice.
    case surprise
    case library(ReferenceVoice.ID)
    /// A voice cast from a description; its first take becomes the session voice.
    case designed(VoiceDesign)

    var design: VoiceDesign? {
        if case .designed(let design) = self { design } else { nil }
    }
}

/// A voice picker row. Designing a voice is an action, so it is a tag the
/// session never stores.
enum VoicePickerTag: Hashable {
    case choice(VoiceChoice)
    case designNew
}

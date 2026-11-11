// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import DramaEngine
import Foundation

/// What speaks for one speaker.
enum VoiceChoice: Hashable, Codable, Sendable {
    case modelDefault
    /// A voice the model picks; its first checked take becomes the session voice.
    case surprise
    case library(ReferenceVoice.ID)
    /// A voice cast from a description; its first checked read becomes the session voice.
    case designed(VoiceDesign)
    /// One of an engine's own voices; its first checked take becomes the session voice.
    case builtIn(BuiltInVoice)

    var design: VoiceDesign? {
        if case .designed(let design) = self { design } else { nil }
    }

    var builtIn: BuiltInVoice? {
        if case .builtIn(let voice) = self { voice } else { nil }
    }
}

/// A voice picker row. Designing a voice and browsing the built-in voices
/// are actions, so they are tags the session never stores.
enum VoicePickerTag: Hashable {
    case choice(VoiceChoice)
    case designNew
    case chooseBuiltIn
}

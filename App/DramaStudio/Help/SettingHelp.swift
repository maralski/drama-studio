import Foundation

/// The help text for every control. The popovers beside each control and the
/// Help window both read from here; case order is the Help window's order.
enum SettingHelp: String, CaseIterable, Identifiable {
    case gettingStarted
    case screenplayMode
    case speakerVoice
    case performanceCues
    case referenceClip
    case voiceTranscript
    case voiceConsent
    case newTake
    case quickPreview
    case expressiveness
    case modelLocation
    case license

    enum Group: String, CaseIterable {
        case start = "Start Here"
        case script = "Script"
        case voices = "Voices"
        case takes = "Takes"
        case model = "Model"
        case about = "About"
    }

    var id: String { rawValue }

    var group: Group {
        switch self {
        case .gettingStarted: .start
        case .screenplayMode, .speakerVoice, .performanceCues: .script
        case .referenceClip, .voiceTranscript, .voiceConsent: .voices
        case .newTake, .quickPreview, .expressiveness: .takes
        case .modelLocation: .model
        case .license: .about
        }
    }

    /// The `GenerationConfig` property this entry explains, by its declared name.
    var configField: String? {
        switch self {
        case .newTake: "seed"
        case .quickPreview: "frameCap"
        default: nil
        }
    }

    var title: String {
        switch self {
        case .gettingStarted: "Getting Started"
        case .screenplayMode: "Screenplay Mode"
        case .speakerVoice: "Speaker Voice"
        case .performanceCues: "Performance Cues"
        case .referenceClip: "Reference Clip"
        case .voiceTranscript: "Transcript"
        case .voiceConsent: "Consent"
        case .newTake: "New Take"
        case .quickPreview: "Quick Preview"
        case .expressiveness: "Expressiveness"
        case .modelLocation: "Model Location"
        case .license: "License and Consent"
        }
    }

    var text: String {
        switch self {
        case .gettingStarted:
            "Type or paste a script in the editor, or click Import Script to open a text, Word, or subtitle file. Add a voice from the sidebar by choosing a clip or recording one, then pick it for a speaker under Cast. Click Generate to hear the take while it is made. When you like it, click Export to save it as WAV or M4A."
        case .screenplayMode:
            "Treats lines written as \u{201C}Name: line\u{201D} as dialogue and gives each name its own speaker, up to 16. Each speaker then gets a voice picker under Cast. It turns on by itself when an imported script looks like a screenplay. Turn it off to have the names read aloud as plain text."
        case .speakerVoice:
            "Picks which library voice speaks for this speaker. Model default lets the model choose a voice of its own. Voices apply in speaker order, so a speaker's voice is used only when every earlier speaker has one too."
        case .performanceCues:
            "Cues tell the model how to deliver a line, such as [laughs] or [sighs], or wrap a passage in tags such as <whisper>\u{2026}</whisper>. Insert Cue puts the cue at the cursor, or around the selected text for a passage cue. Cues are plain script text, so you can also type them by hand."
        case .referenceClip:
            "A short recording of the voice you want, from a file or recorded here. Use 5 to 20 seconds of clean speech with no music or background noise. In testing, cloned voices scored 0.87 speaker similarity to their clip, against about 0.55 for the model's own voices."
        case .voiceTranscript:
            "The exact words spoken in the reference clip. The model pairs the clip with this text to learn the voice, so a wrong or missing word weakens the match. Include filler words and repeats if the speaker said them."
        case .voiceConsent:
            "Confirms that the voice is yours or that the speaker agreed to have it cloned. Drama Studio won't add a voice until you check it. The Fish Audio Community License forbids imitating an identifiable person without their consent."
        case .newTake:
            "Generates the script again with a fresh random seed, so you hear a different read of the same text, voices, and settings. Every generation draws its own seed, so Generate after an edit also gives a new read. Export a take you like before you try another, because a new take replaces it."
        case .quickPreview:
            "Makes faster rough takes for checking casting, cues, and pacing. It trades a possible dropped sentence for speed, because a sentence the model cuts short is kept instead of generated again, and each sentence gets less time to run. Turn it off for the take you export."
        case .expressiveness:
            "Sets how much the delivery varies from one take to the next. The default, in the middle, is the model vendor's own sampling, temperature 0.7 and top-p 0.7. Lower is steadier and flatter, and higher is more varied, with more risk of slurred or skipped words. No setting was measured to beat the default on word error rate, so move it for taste rather than accuracy."
        case .modelLocation:
            "Where Drama Studio looks for the Drama 3 weights, about 41.5 GB. By default it uses the Hugging Face cache, where \u{201C}hf download fishaudio/drama-3\u{201D} puts them. Choose a folder only if you keep the model somewhere else, such as an external drive. Restart Drama Studio after changing it."
        case .license:
            "Drama Studio's code is MIT-licensed, and it runs Fish Audio's Drama 3 model, whose weights are under the Fish Audio Community License. Everything runs on this Mac, and scripts and voices are not uploaded. Clone a real person's voice only with their consent, and read the license for what else it allows."
        }
    }
}

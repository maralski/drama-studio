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
    case temperature
    case topP
    case topK
    case seed
    case chunking
    case chunkBytes
    case maxFrames
    case stopAfterSilence
    case retryThreshold
    case modelLocation
    case license

    enum Group: String, CaseIterable {
        case start = "Start Here"
        case script = "Script"
        case voices = "Voices"
        case sampling = "Sampling"
        case chunking = "Chunking"
        case model = "Model"
        case about = "About"
    }

    var id: String { rawValue }

    var group: Group {
        switch self {
        case .gettingStarted: .start
        case .screenplayMode, .speakerVoice, .performanceCues: .script
        case .referenceClip, .voiceTranscript, .voiceConsent: .voices
        case .temperature, .topP, .topK, .seed: .sampling
        case .chunking, .chunkBytes, .maxFrames, .stopAfterSilence, .retryThreshold: .chunking
        case .modelLocation: .model
        case .license: .about
        }
    }

    /// The `GenerationConfig` property this entry explains, by its declared name.
    var configField: String? {
        switch self {
        case .temperature: "temperature"
        case .topP: "topP"
        case .topK: "topK"
        case .seed: "seed"
        case .chunking: "chunking"
        case .chunkBytes: "chunkBytes"
        case .maxFrames: "maxNewFrames"
        case .stopAfterSilence: "stopAfterSilenceSeconds"
        case .retryThreshold: "minSpokenRatio"
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
        case .temperature: "Temperature"
        case .topP: "Top-p"
        case .topK: "Top-k"
        case .seed: "Seed"
        case .chunking: "Chunking"
        case .chunkBytes: "Chunk Size"
        case .maxFrames: "Max Frames"
        case .stopAfterSilence: "Stop After Silence"
        case .retryThreshold: "Retry Below"
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
        case .temperature:
            "Controls how much the delivery varies from one take to the next. The default is 0.7. Lower it toward 0.5 for steadier, flatter reads, or raise it for more expressive takes at a higher risk of slurred or skipped words."
        case .topP:
            "Limits each step to the most likely choices whose probabilities add up to this value. The default is 0.7. Lower values make the read more predictable, and higher values allow rarer choices and more variety."
        case .topK:
            "Limits each step to this many of the most likely choices. The default is 30. It works together with top-p and the stricter of the two wins, so most people leave it alone and change temperature instead."
        case .seed:
            "With Fixed seed off, every take draws a new random seed, so each Generate gives a different read. Turn it on to repeat a take with the same text, voice, and settings. Change the number or click the dice to try another variation you can come back to."
        case .chunking:
            "Decides how the script is split into turns for the model. One sentence per turn is the default. With the other defaults it scored a mean word error rate of 0.016 in testing, against 0.141 for the reference's 300-byte chunks, which often stopped after the first sentence or trailed into silence. Choose the reference policy only to compare with Fish Audio's own output."
        case .chunkBytes:
            "With the reference chunking policy, the script is cut into pieces of about this many bytes and each piece is generated in turn. The default is 300 bytes, the same as Fish Audio's hosted server. It has no effect when chunking is one sentence per turn."
        case .maxFrames:
            "The longest one chunk of audio may run before it is cut off, counted in frames. The default is 600 frames, about 27.9 seconds. Raise it only if long sentences end abruptly, since a lower cap stops runaway takes sooner."
        case .stopAfterSilence:
            "Ends a chunk once its audio has been silent for this long, instead of waiting for the model to finish on its own. The default is 2 seconds. Turn it off only when the script needs longer pauses, because the model can otherwise trail on in silence."
        case .retryThreshold:
            "Generates a chunk again when its spoken audio falls below this fraction of the time the text should take, which usually means the model skipped words. The default is 0.6, with up to two retries. Set it to 0 to turn retries off and keep every first take."
        case .modelLocation:
            "Where Drama Studio looks for the Drama 3 weights, about 41.5 GB. By default it uses the Hugging Face cache, where \u{201C}hf download fishaudio/drama-3\u{201D} puts them. Choose a folder only if you keep the model somewhere else, such as an external drive. Restart Drama Studio after changing it."
        case .license:
            "Drama Studio's code is MIT-licensed, and it runs Fish Audio's Drama 3 model, whose weights are under the Fish Audio Community License. Everything runs on this Mac, and scripts and voices are not uploaded. Clone a real person's voice only with their consent, and read the license for what else it allows."
        }
    }
}

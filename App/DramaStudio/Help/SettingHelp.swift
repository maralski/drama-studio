// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation

/// The help text for every control. The popovers beside each control and the
/// Help window both read from here; case order is the Help window's order.
enum SettingHelp: String, CaseIterable, Identifiable {
    case gettingStarted
    case screenplayMode
    case speakerVoice
    case performanceCues
    case examples
    case sections
    case pronunciation
    case projects
    case surpriseMe
    case voiceDesign
    case referenceClip
    case voiceTranscript
    case voiceConsent
    case newTake
    case directorsNote
    case keepTake
    case quickPreview
    case expressiveness
    case playbackSpeed
    case streaming
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
        case .screenplayMode, .speakerVoice, .performanceCues, .examples, .sections, .pronunciation, .projects: .script
        case .surpriseMe, .voiceDesign, .referenceClip, .voiceTranscript, .voiceConsent: .voices
        case .newTake, .directorsNote, .keepTake, .quickPreview, .expressiveness, .playbackSpeed: .takes
        case .modelLocation, .streaming: .model
        case .license: .about
        }
    }

    /// The `GenerationConfig` property this entry explains, by its declared name.
    var configField: String? {
        switch self {
        case .newTake: "seed"
        case .quickPreview: "frameCap"
        case .streaming: "streaming"
        default: nil
        }
    }

    var title: String {
        switch self {
        case .gettingStarted: "Getting Started"
        case .screenplayMode: "Screenplay Mode"
        case .speakerVoice: "Speaker Voice"
        case .performanceCues: "Performance Cues"
        case .examples: "Examples"
        case .sections: "Sections"
        case .pronunciation: "Pronunciation"
        case .projects: "Projects"
        case .surpriseMe: "Surprise Me"
        case .voiceDesign: "Voice Design"
        case .referenceClip: "Reference Clip"
        case .voiceTranscript: "Transcript"
        case .voiceConsent: "Consent"
        case .newTake: "Generating Takes"
        case .directorsNote: "Director's Note"
        case .keepTake: "Use for Export"
        case .quickPreview: "Quick Preview"
        case .expressiveness: "Expressiveness"
        case .playbackSpeed: "Playback Speed"
        case .streaming: "Streaming"
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
            "Picks who speaks for this speaker: a library voice, the model's default, a surprise voice, or a designed one. Model default lets the model choose a voice of its own for every take. Voices apply in speaker order, so a speaker's voice is used only when every earlier speaker has one too, and a script with several speakers uses library voices only."
        case .performanceCues:
            "Cues direct delivery with markers such as [laughs] or passage tags such as <whisper>\u{2026}</whisper>. Insert Cue puts a marker at the cursor or wraps the selected passage. Normal generation requires macOS 26 and an installed local speech model, checks words before playback, and stops if retries cannot verify them. Quick preview is unchecked; listen before export to judge the acting and cue sounds."
        case .examples:
            "The Examples menu has a line for each opening style, which goes in at the cursor, and a cast of ready-made characters. Choosing a character replaces the script with its line and casts its designed voice. Use them to hear what cues and voice design can do before you write your own."
        case .sections:
            "Import a PowerPoint, PDF or EPUB to narrate it section by section. Generate All narrates included sections; Generate Changed Sections updates missing or outdated audio, and Narrate This Section regenerates the selected section. Click a take to audition it, then Use for Export to choose it. Export the whole narration, individual sections, captions, or a slide narration package."
        case .pronunciation:
            "Rules in Settings rewrite words in the script before the model hears them, so a name or an abbreviation is said the way you want, such as Nguyen said as win. Whole word keeps a rule from changing part of a longer word, and Case sensitive makes US and us different rules. Rules never touch speaker tags, cues in brackets, or markup, and they apply to every take and section you generate. Edit the script instead when only one line should change."
        case .projects:
            "Save Project preserves all takes, their original text and voice assignments, and your export selections. Save updates the current project; Save Project As creates a copy, and a background recovery draft protects edits between saves. Older projects open with an audio verification warning because they did not store generation history. Library voice assignments still refer to your local voice library."
        case .surpriseMe:
            "Lets the model invent a voice for this speaker. The first checked take sets that voice, and later takes and retakes keep it. Pick Surprise me again for a different voice, or click Save Voice to Library to save it to the library."
        case .voiceDesign:
            "Casts a voice from a description instead of a recording: a name, how the voice sounds, and an optional scene. Try a Line speaks a sample in that voice, and if you then click Use Voice without changing the design, the voice you heard is the voice you get. Try a Line checks the spoken words even when Quick preview is enabled on the main screen. Designed voices cast scripts with one speaker, so click Save Voice to Library to save one to the library for a screenplay."
        case .referenceClip:
            "A short recording of the voice you want, from a file or recorded here. Use 5 to 20 seconds of clean speech with no music or background noise. In testing, cloned voices scored 0.87 speaker similarity to their clip, against about 0.55 for the model's own voices."
        case .voiceTranscript:
            "The exact words spoken in the reference clip. The model pairs the clip with this text to learn the voice, so a wrong or missing word weakens the match. Include filler words and repeats if the speaker said them."
        case .voiceConsent:
            "Confirms that the voice is yours or that the speaker agreed to have it cloned. Drama Studio won't add a voice until you check it. The Fish Audio Community License forbids imitating an identifiable person without their consent."
        case .newTake:
            "Click Generate or Generate All again for a fresh take of the same text, voices, and settings. Scroll the grid to browse takes, click a card to audition it, and choose Use for Export to change the exported take. The trash button deletes a take after confirmation, replacing the export take with the newest remaining take when needed. Generating after an edit preserves earlier takes and marks them as outdated."
        case .directorsNote:
            "Tells the performer how to read the line again, such as softer and warmer or slower, with more pauses. Retake reads the whole script once more with the note written in, after the takes so far, so the model hears what to change. Retakes work on scripts with one speaker."
        case .keepTake:
            "Click a take to audition it, then Use for Export to choose the audio and captions to export. The Export badge and status identify the output. Auditioning another take does not change it."
        case .quickPreview:
            "Makes faster rough takes for checking casting, cues, and pacing. It skips word verification and retries, uses a shorter time limit, and stops after two seconds of silence, so it can drop words. Only checked takes become automatic voice references or retake history, and voice auditions always use checked generation. Turn Quick Preview off for the take you export."
        case .expressiveness:
            "Sets how much the delivery varies from one take to the next. The default, in the middle, uses temperature 0.7 and top-p 0.7, with steadier delivery below and more variation above. Normal generation checks the spoken words and may try a broader delivery when a cue attempt fails that check. Listen before export to judge the performance."
        case .playbackSpeed:
            "Plays the audio faster or slower without changing its pitch, from 0.75x to 1.5x. It changes playback only: exports, captions, and projects keep the original timing. The scrubber and the sentence highlight follow the audio at every speed, and the setting is remembered."
        case .streaming:
            "Plays completed chunks while the rest of the script generates. Normal generation checks each chunk and can retry an incomplete attempt before you hear it. Quick Preview skips retries and can start playback before a chunk finishes. Turn streaming off to wait for completed chunks even in Quick Preview."
        case .modelLocation:
            "Where Drama Studio looks for the Drama 3 weights, about 41.5 GB. By default it uses the Hugging Face cache, where \u{201C}hf download fishaudio/drama-3\u{201D} puts them. Choose a folder only if you keep the model somewhere else, such as an external drive. Restart Drama Studio after changing it."
        case .license:
            "Drama Studio is written by Peter Marelas and its code is under the Apache License 2.0. It runs Fish Audio's Drama 3 model, whose weights are under the Fish Audio Community License, and the parts of Drama Studio derived from Fish Audio's code stay under that license too. Everything runs on this Mac, and scripts and voices are not uploaded. Clone a real person's voice only with their consent, and read the license for what else it allows."
        }
    }
}

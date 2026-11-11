// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import DramaEngine
import Foundation

/// The help text for every control. The popovers beside each control and the
/// Help window both read from here; case order is the Help window's order.
enum SettingHelp: String, CaseIterable, Identifiable {
    case gettingStarted
    case requirements
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
    case expressiveness
    case playbackSpeed
    case streaming
    case modelLocation
    case mingLocation
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
        case .requirements, .gettingStarted: .start
        case .screenplayMode, .speakerVoice, .performanceCues, .examples, .sections, .pronunciation, .projects: .script
        case .surpriseMe, .voiceDesign, .referenceClip, .voiceTranscript, .voiceConsent: .voices
        case .newTake, .directorsNote, .keepTake, .expressiveness, .playbackSpeed: .takes
        case .modelLocation, .mingLocation, .streaming: .model
        case .license: .about
        }
    }

    /// The `GenerationConfig` property this entry explains, by its declared name.
    var configField: String? {
        switch self {
        case .newTake: "seed"
        case .streaming: "streaming"
        default: nil
        }
    }

    var title: String {
        switch self {
        case .requirements: "System requirements"
        case .gettingStarted: "Getting started"
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
        case .expressiveness: "Expressiveness"
        case .playbackSpeed: "Playback Speed"
        case .streaming: "Streaming"
        case .modelLocation: "Model Location"
        case .mingLocation: "Ming Location"
        case .license: "License and Consent"
        }
    }

    var text: String {
        switch self {
        case .requirements:
            "Requires an Apple silicon Mac and macOS 26 or later. Ming is a 3.0 GB download and is the engine to use on a Mac with less than 64 GB of unified memory. Drama 3 needs about 49 GB of free disk for setup and 64 GB or more of memory: it maps about 41.5 GB of weights, and a smaller Mac may swap heavily or fail to load. Both engines have been verified only on an M3 Max with 128 GB, so neither figure is a tested minimum. A network connection is needed for downloads and update checks; generation runs locally. The studio window needs at least 900 by 640 points."
        case .gettingStarted:
            "Type or paste a script in the editor, or click Import Script to open a text, Word, or subtitle file. Add a voice from the sidebar by choosing a clip or recording one, then pick it for a speaker under Cast. Click Generate to hear the take while it is made. When you like it, click Export to save it as WAV or M4A."
        case .screenplayMode:
            "Write each speaker's name, a colon, a space, then the spoken words. For example:\n\nAlex: Hello there.\nSam: Good morning.\n\nUse up to 16 speakers. Names must start with a letter and can contain letters, numbers, spaces, apostrophes, periods and hyphens, up to 40 characters. A line without a name continues the previous speaker; text before the first named speaker belongs to Narrator. Names are case-insensitive. Each speaker gets a voice picker under Cast. Imported dialogue can turn this mode on automatically. Turn it off to read ordinary prose or have the names spoken aloud."
        case .speakerVoice:
            "Picks who speaks for this speaker: a library voice, the model's default, a surprise voice, a designed one, or one of an engine's built-in voices. Model default lets the model choose a voice of its own for every take. Choose a Built-in Voice lists the voices that come with an installed engine, such as Ming's \(SpeechEngineID.ming.builtInVoices.count), and such a voice speaks only on its own engine. Voices apply in speaker order, so a speaker's voice is used only when every earlier speaker has one too, and a script with several speakers uses library voices only."
        case .performanceCues:
            "Cues direct delivery with markers such as [laughs] or passage tags such as <whisper>\u{2026}</whisper>. Insert Cue puts a marker at the cursor or wraps the selected passage. Normal generation requires macOS 26 and an installed local speech model, checks words before playback, and stops if retries cannot verify them. Listen before export to judge the acting and cue sounds."
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
            "Casts a voice from a description instead of a recording: a name, how the voice sounds, and an optional scene. Try a Line speaks a sample in that voice, and if you then click Use Voice without changing the design, the voice you heard is the voice you get. Try a Line checks the spoken words before playback. Designed voices cast scripts with one speaker, so click Save Voice to Library to save one to the library for a screenplay."
        case .referenceClip:
            "A short recording of the voice you want, from a file or recorded here. Use 5 to 20 seconds of clean speech with no music or background noise. "
                + SpeechEngineID.allCases.compactMap { engine in
                    engine.capabilities.maxCloneSeconds.map { "\(engine.capabilities.name) uses the whole clip, and one over \(Int($0)) seconds leaves it less room to speak each line. " }
                }.joined()
                + "In testing with Drama 3, cloned voices scored 0.87 speaker similarity to their clip, against about 0.55 for the model's own voices."
        case .voiceTranscript:
            "The exact words spoken in the reference clip. The model pairs the clip with this text to learn the voice, so a wrong or missing word weakens the match. Include filler words and repeats if the speaker said them."
        case .voiceConsent:
            "Confirms that the voice is yours or that the speaker agreed to have it cloned. Drama Studio won't add a voice until you check it. The Fish Audio Community License forbids imitating an identifiable person without their consent."
        case .newTake:
            "Click Generate or Generate All again for a fresh take of the same text, voices, and settings. Scroll the grid to browse takes or drag its top edge up to show more rows, click a card to audition it, and choose Use for Export to change the exported take. The trash button deletes a take after confirmation, replacing the export take with the newest remaining take when needed. Generating after an edit preserves earlier takes and marks them as outdated."
        case .directorsNote:
            "Tells the performer how to read the line again, such as softer and warmer or slower, with more pauses. Retake reads the whole script once more with the note written in, after the takes so far, so the model hears what to change. Retakes work on scripts with one speaker."
        case .keepTake:
            "Click a take to audition it, then Use for Export to choose the audio and captions to export. The Export badge and status identify the output. Auditioning another take does not change it."
        case .expressiveness:
            "Sets how much the delivery varies from one take to the next. The default, in the middle, uses temperature 0.7 and top-p 0.7, with steadier delivery below and more variation above. Normal generation checks the spoken words and may try a broader delivery when a cue attempt fails that check. Listen before export to judge the performance."
        case .playbackSpeed:
            "Plays the audio faster or slower without changing its pitch, from 0.75x to 1.5x. It changes playback only: exports, captions, and projects keep the original timing. The scrubber and the sentence highlight follow the audio at every speed, and the setting is remembered."
        case .streaming:
            "Plays completed chunks while the rest of the script generates. Generation checks each chunk and can retry an incomplete attempt before you hear it. Each chunk finishes generation and word checks before playback."
        case .modelLocation:
            "Where Drama Studio looks for the Drama 3 weights, about 41.5 GB. It checks your chosen folder first, then its own download, then the newest Hugging Face cache snapshot. Choose Folder selects an existing model; Use Default Location returns to automatic discovery and download. Setup checks the new location and attaches the engine without requiring a restart."
        case .mingLocation:
            "Where Drama Studio looks for Ming, a \(ModelSetup.gigabytes(SpeechEngineID.ming.install.plan.totalBytes)) engine for Macs with less memory than Drama 3 needs. The MING_WEIGHTS environment variable comes first when it is set; then your chosen folder, Drama Studio's own download, and the newest Hugging Face cache snapshot. Choose Folder selects an existing checkpoint; Use Default Location returns to automatic discovery. Download fetches Ming into Drama Studio's own folder, and Ming is ready without a restart."
        case .license:
            "Drama Studio is written by Peter Marelas and its code is under the Apache License 2.0. It runs Fish Audio's Drama 3 model, whose weights are under the Fish Audio Community License, as are the parts of Drama Studio derived from Fish Audio's code, and inclusionAI's Ming model, whose weights are under the MIT License. Everything runs on this Mac, and scripts and voices are not uploaded. Clone a real person's voice only with their consent, and read the license for what else it allows."
        }
    }
}

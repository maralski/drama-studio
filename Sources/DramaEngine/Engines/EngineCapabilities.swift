// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation

/// What one engine can do, as the app, the estimate and the API read it.
/// ``SpeechEngineID/capabilities`` is the one table; adding an engine fails
/// to compile until its row exists.
public struct EngineCapabilities: Sendable, Equatable {
    /// Which inline cues the engine performs.
    public enum CueVocabulary: Sendable, Equatable {
        /// Any `[bracket cue]` and any `<tag>` passage, as written.
        case freeForm
        /// Only these exact tags. Other cues go through ``SpeechEngineID/cueSupport(_:)``.
        case fixed([String])
    }

    public enum Languages: Sendable, Equatable {
        case unrestricted
        /// BCP-47 language codes.
        case only([String])
    }

    /// The engine's name for people.
    public let name: String
    /// Clones a voice from a clip and a transcript.
    public let cloning: Bool
    /// The longest clip that clones well; nil for no engine limit.
    public let maxCloneSeconds: Double?
    /// Built-in voices by name; empty for an engine without them.
    public let builtInVoices: [String]
    /// Casts a voice from a description.
    public let voiceDesign: Bool
    public let cues: CueVocabulary
    /// Paired `<whisper>…</whisper>` passages.
    public let passageTags: Bool
    /// `<|speaker:N|>` turns, one speaker per sentence.
    public let speakerTurns: Bool
    /// Earlier takes feed back as history, not only as the voice.
    public let history: Bool
    public let languages: Languages
    /// Samples one ``SpeechEvent/frame(index:maxFrames:)`` step stands for.
    public let samplesPerStep: Int
    /// Steps per second an M3 Max generates, the estimate's starting rate.
    public let typicalStepsPerSecond: Double

    /// Steps per second of audio.
    public var stepsPerSecondOfAudio: Double { Double(ModelConstants.sampleRate) / Double(samplesPerStep) }
}

/// How an engine performs one cue as typed, such as `[laughs]` or `<whisper>`.
public enum CueSupport: Sendable, Equatable {
    /// Performed as written.
    case native
    /// Replaced by this engine tag.
    case tag(String)
    /// Lifted into a request-level style named here, such as "happy".
    case style(String)
    /// The markup is removed, enclosed words are kept, and the result reports it.
    case unsupported
}

public extension SpeechEngineID {
    var capabilities: EngineCapabilities {
        switch self {
        case .drama3:
            EngineCapabilities(name: "Drama 3", cloning: true, maxCloneSeconds: nil, builtInVoices: [], voiceDesign: true,
                               cues: .freeForm, passageTags: true, speakerTurns: true, history: true, languages: .unrestricted,
                               samplesPerStep: ModelConstants.samplesPerFrame, typicalStepsPerSecond: GenerationEstimate.typicalFramesPerSecond)
        case .ming:
            EngineCapabilities(name: "Ming", cloning: true, maxCloneSeconds: 15, builtInVoices: MingCues.builtInVoices, voiceDesign: true,
                               cues: .fixed(MingCues.tags), passageTags: false, speakerTurns: true, history: false,
                               languages: .only(MingCues.languages), samplesPerStep: MingVoiceEncoding.samplesPerLatent, typicalStepsPerSecond: 60)
        }
    }

    /// How this engine performs `cue`, written as typed in a script.
    func cueSupport(_ cue: String) -> CueSupport {
        switch self {
        case .drama3: .native
        case .ming: MingCues.support(cue)
        }
    }
}

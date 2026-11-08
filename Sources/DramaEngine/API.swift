// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation

/// One generated or prompt frame: `[main token, c0, ..., c9]`.
/// Text rows carry the text token and ten zeros. Audio rows carry
/// `c0 + 248099` as the main token. One audio frame is 2048 samples at
/// 44.1 kHz, about 46 ms.
public typealias Frame = [Int32]

/// Fixed facts about the Drama 3 model and its codec.
public enum ModelConstants: Sendable {
    /// Codebooks per audio frame: one semantic codebook and nine residual ones.
    public static let numCodebooks = 10
    /// Entries in codebook 0, the semantic codebook.
    public static let codebookSize = 4096
    /// Entries in each of codebooks 1 to 9.
    public static let residualCodebookSize = 1024
    /// The main-token id of semantic code 0. An audio frame's main token is
    /// `c0 + semanticStart`.
    public static let semanticStart: Int32 = 248_099
    package static let semanticEnd: Int32 = 252_194
    package static let eos: Int32 = 248_046
    /// Output sample rate in Hz. Every sample buffer the engine returns is
    /// mono at this rate.
    public static let sampleRate = 44_100
    /// Samples per audio frame. 44,100 / 2048 is about 21.5 frames per second
    /// of audio.
    public static let samplesPerFrame = 2048
    /// The highest speaker id a `<|speaker:N|>` tag may name. Ids run from 0.
    public static let maxSpeakerID = 15
    package static let hiddenSize = 2048
}

/// How the engine samples and paces one request. By default, each sentence
/// gets a generation turn and short attempts are retried before audio leaves
/// the engine. Normal takes wait for the model's end of speech, subject to
/// the frame cap and loop guard. Silence stopping is opt-in because a quiet
/// pause can precede words the model has not generated yet.
public struct GenerationConfig: Sendable, Codable, Equatable {
    /// Normal generation checks decoded speech locally before releasing it.
    /// Preview deliberately permits unchecked, partially streamed audio.
    public var wordVerification: WordVerification = .required
    /// Sampling temperature. Default 0.7, the vendor's value.
    public var temperature: Double = 0.7
    /// Top-k candidates kept before sampling. Default 30. 4097, the number
    /// of candidates, keeps them all and leaves the choice to top-p.
    public var topK: Int = 30
    /// Top-p nucleus kept before sampling. Default 0.7, the vendor's value.
    /// 1 keeps every candidate up to top-k.
    public var topP: Double = 0.7
    /// The most frames one chunk may run. Defaults to the text's expected
    /// duration times 1.5, plus 2 s. Slow reference voices extend the expected
    /// duration using their transcript length and audio duration.
    public var frameCap: FrameCap = .fromText(margin: 1.5, floorSeconds: 2.0)
    package var maxNewFrames: Int {
        get {
            switch frameCap {
            case .fixed(let frames): frames
            case .fromText: FrameCap.maximumFrames
            }
        }
        set { frameCap = .fixed(newValue) }
    }
    /// Seed for the sampler. `nil`, the default, draws a fresh seed per
    /// request. The same seed, text and voice give the same frames.
    public var seed: UInt64? = nil
    /// Under ``Chunking/reference``, cut long text into chunks of about this
    /// many UTF-8 bytes. Default 300, the hosted server's value. Earlier
    /// chunks feed back as history.
    public var chunkBytes: Int = 300
    /// How text is cut into generation turns. Default ``Chunking/sentences``.
    /// Required word verification always uses sentence turns.
    public var chunking: Chunking = .sentences
    /// Stop a chunk once its decoded audio has been silent for this many
    /// seconds. `nil`, the default, lets the model reach the end of speech,
    /// subject to the frame cap and loop guard. A silence stop can cut words
    /// after a pause.
    public var stopAfterSilenceSeconds: Double? = nil
    /// Generate a chunk again when its spoken audio is shorter than this
    /// fraction of the expected spoken duration (text length at 16 characters
    /// per second). Reference audio can include silence, so its duration only
    /// extends the frame allowance, never this retry threshold. Default 0.6. On the
    /// evaluation set every good run scored at
    /// least 0.72 and every failed run at most 0.65. 0 turns retries off.
    public var minSpokenRatio: Double = 0.6
    /// The most times one chunk is generated again. Default 5. Replacements
    /// must pass word recognition, including its uncertainty check.
    public var maxRetries: Int = 5
    /// Stop a chunk once its main tokens settle into a repeating cycle, the
    /// shape of a laughter loop. Default true. On the reliability study's two
    /// laughter loops the last 48 main tokens repeated the tokens 17 or 34
    /// frames earlier exactly, while no speech run among 267 matched above
    /// 0.75 at any lag from 17 to 40 frames.
    public var stopOnLoop: Bool = true
    /// Also generate a chunk again when the silence stop or the frame cap
    /// ended it before the end of speech and its spoken audio is shorter than
    /// this fraction of the expected duration. A chunk the loop guard stopped
    /// is always generated again. Default 1.0. In the performance-cue study
    /// every such attempt that lost a sentence spoke 0.64 to 0.90 of its
    /// expected duration, while those that spoke every word, often with a
    /// trailing `[sighs]` that never reaches the end of speech, spoke 1.19 to
    /// 1.5 of it. 0 leaves the ``minSpokenRatio`` rule alone.
    public var retryWithoutEOSBelow: Double = 1.0
    /// Seconds added to a text-derived ``frameCap`` for each bracket cue in a
    /// chunk, since a `[laugh]`, a `[pause]` or a `[sigh]` takes time the
    /// characters do not count. Default 1.5. Without it, slow cue chunks were
    /// cut at the cap with words missing; with it, the two lines that had been
    /// cut kept every sentence in 5 of 5 runs each, and loops cost no more,
    /// since the loop guard ends them.
    /// A ``FrameCap/fixed(_:)`` cap gets no allowance.
    public var capSecondsPerCue: Double = 1.5
    /// When decoded audio leaves the engine. Default ``StreamingMode/sentence``.
    /// Low latency takes effect only with ``WordVerification/uncheckedPreview``
    /// and `maxRetries` 0.
    public var streaming: StreamingMode = .sentence
    /// The measured defaults.
    public init() {}

    public enum WordVerification: String, Sendable, Codable {
        case required, uncheckedPreview
    }

    enum CodingKeys: String, CodingKey {
        case wordVerification, temperature, topK, topP, frameCap, seed, chunkBytes, chunking
        case stopAfterSilenceSeconds, minSpokenRatio, maxRetries, stopOnLoop, retryWithoutEOSBelow
        case capSecondsPerCue, streaming
    }

    public init(from decoder: Decoder) throws {
        self.init()
        let values = try decoder.container(keyedBy: CodingKeys.self)
        wordVerification = try values.decodeIfPresent(WordVerification.self, forKey: .wordVerification) ?? .required
        temperature = try values.decodeIfPresent(Double.self, forKey: .temperature) ?? temperature
        topK = try values.decodeIfPresent(Int.self, forKey: .topK) ?? topK
        topP = try values.decodeIfPresent(Double.self, forKey: .topP) ?? topP
        frameCap = try values.decodeIfPresent(FrameCap.self, forKey: .frameCap) ?? frameCap
        seed = try values.decodeIfPresent(UInt64.self, forKey: .seed)
        chunkBytes = try values.decodeIfPresent(Int.self, forKey: .chunkBytes) ?? chunkBytes
        chunking = try values.decodeIfPresent(Chunking.self, forKey: .chunking) ?? chunking
        stopAfterSilenceSeconds = try values.decodeIfPresent(Double.self, forKey: .stopAfterSilenceSeconds)
        minSpokenRatio = try values.decodeIfPresent(Double.self, forKey: .minSpokenRatio) ?? minSpokenRatio
        maxRetries = try values.decodeIfPresent(Int.self, forKey: .maxRetries) ?? maxRetries
        stopOnLoop = try values.decodeIfPresent(Bool.self, forKey: .stopOnLoop) ?? stopOnLoop
        retryWithoutEOSBelow = try values.decodeIfPresent(Double.self, forKey: .retryWithoutEOSBelow) ?? retryWithoutEOSBelow
        capSecondsPerCue = try values.decodeIfPresent(Double.self, forKey: .capSecondsPerCue) ?? capSecondsPerCue
        streaming = try values.decodeIfPresent(StreamingMode.self, forKey: .streaming) ?? streaming
    }

    /// How text is cut into generation turns.
    public enum Chunking: String, Sendable, Codable, CaseIterable {
        /// The reference's policy: about ``GenerationConfig/chunkBytes`` per
        /// turn, at most three speakers. It often stops after the first
        /// sentence of a turn; it is here for unchecked comparison. Required
        /// word verification overrides this choice with sentence turns.
        case reference
        /// One sentence per turn, with the previous turns as history. The
        /// default. ``Drama3Chunking/splitSentencesTagged(_:)`` shows the cut.
        case sentences
    }
}

public extension GenerationConfig {
    /// The middle of the expressiveness range: temperature 0.7 and top-p 0.7.
    static let defaultExpressiveness = 0.5

    /// The expressiveness control the app and the API expose over the defaults. Every
    /// other field keeps the measured value.
    ///
    /// - Parameters:
    ///   - expressiveness: 0 to 1. The middle, 0.5, is the vendor's sampling
    ///     (temperature 0.7, top-p 0.7, top-k 30) and the top is the Hugging
    ///     Face Space's (temperature 1.0, top-p 0.9, no top-k), with
    ///     temperature and top-p linear between the ends and the middle:
    ///     both `0.5 + 0.4 * expressiveness` up to the middle, then
    ///     temperature `0.7 + 0.6 * (expressiveness - 0.5)` and top-p
    ///     `0.7 + 0.4 * (expressiveness - 0.5)`, top-p alone bounding the
    ///     choice above the middle. On the reliability study's 130 prose
    ///     rows the Space's sampling lost no row, reached the end of speech
    ///     without a retry more often, kept every sentence of the cue lines,
    ///     and cost 0.007 of word error rate; the middle stays the default
    ///     for that cost.
    static func make(expressiveness: Double) -> GenerationConfig {
        var config = GenerationConfig()
        if expressiveness <= 0.5 {
            config.temperature = 0.5 + 0.4 * expressiveness
            config.topP = 0.5 + 0.4 * expressiveness
        } else {
            // Interpolated so that 1 lands on the Space's values exactly.
            let t = (expressiveness - 0.5) * 2
            config.temperature = 0.7 * (1 - t) + 1.0 * t
            config.topP = 0.7 * (1 - t) + 0.9 * t
            config.topK = 4097
        }
        return config
    }
}

/// When decoded audio leaves the engine.
public enum StreamingMode: Sendable, Codable, Equatable {
    /// Each chunk's audio arrives as one ``SpeechEvent/audio(samples:chunkIndex:)``
    /// once the chunk is generated. A chunk that comes out short is
    /// generated again before any of it leaves.
    case sentence
    /// Audio also arrives as ``SpeechEvent/partialAudio(samples:chunkIndex:frameRange:)``
    /// every `stepFrames` generated frames, decoded on the CPU while the GPU
    /// generates. A chunk is never generated again, because its first audio
    /// may already be playing.
    /// Requests use ``sentence`` while ``GenerationConfig/wordVerification``
    /// is ``GenerationConfig/WordVerification/required``, or while `maxRetries`
    /// and `minSpokenRatio` are positive, so failed attempts can be replaced
    /// before playback.
    case lowLatency(stepFrames: Int)

    /// Partials every 64 frames, about 3 s of audio. In measurements of an
    /// earlier streaming implementation, 32-frame steps started 0.7 s sooner
    /// but ran out of audio before the next partial in every measured trace.
    public static let defaultLowLatency = StreamingMode.lowLatency(stepFrames: 64)

    /// The partial step in frames, at least 1; nil for ``sentence``.
    public var stepFrames: Int? {
        if case .lowLatency(let step) = self { max(1, step) } else { nil }
    }
}

/// The most frames one chunk may run before the engine stops it.
public enum FrameCap: Sendable, Codable, Equatable {
    /// Exactly this many frames.
    case fixed(Int)
    /// The text's expected duration times `margin`, plus `floorSeconds`,
    /// clamped to ``minimumFrames`` through ``maximumFrames``. The expected
    /// duration uses 16 characters per second, or the reference voice's slower
    /// pace, leaving out cues and tags.
    case fromText(margin: Double, floorSeconds: Double)

    /// The smallest text-derived cap: 43 frames, about 2 s of audio.
    public static let minimumFrames = 43
    /// The largest text-derived cap: 600 frames, about 27.9 s of audio.
    public static let maximumFrames = 600

    /// The cap in frames for a chunk expected to last `expectedSeconds`.
    public func frames(expectedSeconds: Double) -> Int {
        switch self {
        case .fixed(let frames):
            return frames
        case .fromText(let margin, let floorSeconds):
            let seconds = expectedSeconds * margin + floorSeconds
            let frames = Int((seconds * Double(ModelConstants.sampleRate) / Double(ModelConstants.samplesPerFrame)).rounded(.up))
            return min(Self.maximumFrames, max(Self.minimumFrames, frames))
        }
    }
}

/// A text-to-speech engine that a voice encoding, a take or a synthesizer belongs to.
public enum SpeechEngineID: String, Codable, CodingKeyRepresentable, Sendable, Hashable, CaseIterable {
    /// Fish Audio's Drama 3.
    case drama3
    /// Ming-3.0-tts.
    case ming
}

/// What one engine needs to clone a voice, made from the voice's clip by
/// ``SpeechSynthesizer/encodeVoice(samples:transcript:)``.
public enum VoiceEncoding: Codable, Sendable, Hashable {
    /// `[10][T]` Drama 3 codec codes, one row per codebook.
    case drama3(codes: [[Int32]])
    /// Ming's clone conditioning for the clip.
    case ming(reference: MingVoiceEncoding)

    /// The engine that reads this encoding.
    public var engine: SpeechEngineID {
        switch self {
        case .drama3: .drama3
        case .ming: .ming
        }
    }

    /// The length of the audio the encoding stands for, at 44.1 kHz.
    public var sampleCount: Int {
        switch self {
        case .drama3(let codes): (codes.first?.count ?? 0) * ModelConstants.samplesPerFrame
        case .ming(let reference): reference.sampleCount
        }
    }
}

/// A cloned voice: a clip's transcript plus one encoding of the clip per
/// engine that has read it. The clip itself lives in the voice library,
/// beside the voice. A voice saved before encodings existed decodes with a
/// Drama 3 encoding and no clip.
public struct ReferenceVoice: Sendable, Codable, Identifiable, Equatable {
    /// A stable identity for a voice library.
    public var id: UUID
    /// A display name.
    public var name: String
    /// What the clip says, word for word. A wrong transcript weakens the clone.
    public var transcript: String
    /// The clip encoded for each engine, keyed by the encoding's own engine.
    public private(set) var encodings: [SpeechEngineID: VoiceEncoding]
    /// The stored clip's length in 44.1 kHz samples; nil when the library has
    /// no clip for this voice.
    public var clipSampleCount: Int?
    /// The clip's file name, for display.
    public var sourceFileName: String?

    /// Creates a voice from encodings you already have.
    public init(id: UUID = UUID(), name: String, transcript: String, encodings: [VoiceEncoding], clipSampleCount: Int? = nil, sourceFileName: String? = nil) {
        self.id = id
        self.name = name
        self.transcript = transcript
        self.encodings = Dictionary(encodings.map { ($0.engine, $0) }, uniquingKeysWith: { _, last in last })
        self.clipSampleCount = clipSampleCount
        self.sourceFileName = sourceFileName
    }

    /// Adds the encoding, replacing any earlier one for its engine.
    public mutating func store(_ encoding: VoiceEncoding) {
        encodings[encoding.engine] = encoding
    }

    /// The clip's length in seconds, or the longest encoding's when there is no clip.
    public var durationSeconds: Double {
        Double(clipSampleCount ?? encodings.values.map(\.sampleCount).max() ?? 0) / Double(ModelConstants.sampleRate)
    }

    private enum CodingKeys: String, CodingKey { case id, name, transcript, encodings, clipSampleCount, sourceFileName, codes }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(UUID.self, forKey: .id)
        name = try container.decode(String.self, forKey: .name)
        transcript = try container.decode(String.self, forKey: .transcript)
        clipSampleCount = try container.decodeIfPresent(Int.self, forKey: .clipSampleCount)
        sourceFileName = try container.decodeIfPresent(String.self, forKey: .sourceFileName)
        if let encodings = try container.decodeIfPresent([SpeechEngineID: VoiceEncoding].self, forKey: .encodings) {
            guard encodings.allSatisfy({ $0.key == $0.value.engine }) else {
                throw DecodingError.dataCorruptedError(forKey: .encodings, in: container, debugDescription: "an encoding is filed under another engine")
            }
            self.encodings = encodings
        } else {
            encodings = [.drama3: .drama3(codes: try container.decode([[Int32]].self, forKey: .codes))]
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(id, forKey: .id)
        try container.encode(name, forKey: .name)
        try container.encode(transcript, forKey: .transcript)
        try container.encode(encodings, forKey: .encodings)
        try container.encodeIfPresent(clipSampleCount, forKey: .clipSampleCount)
        try container.encodeIfPresent(sourceFileName, forKey: .sourceFileName)
    }
}

/// What the engine that made a take needs to continue from it: as history,
/// or as the voice reference of a later request.
public enum TakeContext: Codable, Sendable, Hashable {
    /// `[10][T]` Drama 3 codec codes, one row per codebook.
    case drama3(codes: [[Int32]])
    /// Ming has no history, so a later take continues only by cloning this
    /// one: `reference` encodes the take's opening sentences, up to about
    /// 10 s so a request still fits Ming's 512-position cache, and
    /// `sampleCount` is the whole take's length.
    case ming(reference: MingVoiceEncoding, sampleCount: Int)

    /// The engine that made the take.
    public var engine: SpeechEngineID {
        switch self {
        case .drama3: .drama3
        case .ming: .ming
        }
    }

    /// The length of the take's audio, at 44.1 kHz.
    public var sampleCount: Int {
        switch self {
        case .drama3(let codes): (codes.first?.count ?? 0) * ModelConstants.samplesPerFrame
        case .ming(_, let sampleCount): sampleCount
        }
    }

    /// The take as a voice encoding, so a voice saved from a take needs no encoder.
    public var voiceEncoding: VoiceEncoding {
        switch self {
        case .drama3(let codes): .drama3(codes: codes)
        case .ming(let reference, _): .ming(reference: reference)
        }
    }
}

/// One read of a line: the user turn text exactly as it went into the prompt,
/// and the engine's context for it. Fed back as history or as the voice
/// reference, a take reproduces that turn.
public struct Take: Sendable, Codable, Hashable {
    /// The user turn as the prompt held it, including a character's
    /// `Name: ` or a director's note.
    public var text: String
    /// What the engine that made the take needs to continue from it.
    public var context: TakeContext
    /// Creates a take from stored text and context.
    public init(text: String, context: TakeContext) {
        self.text = text
        self.context = context
    }

    /// The length of the take's audio, at 44.1 kHz.
    public var sampleCount: Int { context.sampleCount }
    /// The take's length in seconds.
    public var durationSeconds: Double { Double(sampleCount) / Double(ModelConstants.sampleRate) }

    private enum CodingKeys: String, CodingKey { case text, context, codes }

    /// Takes saved before contexts existed hold `codes` and decode as Drama 3 takes.
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        text = try container.decode(String.self, forKey: .text)
        context = try container.decodeIfPresent(TakeContext.self, forKey: .context)
            ?? .drama3(codes: try container.decode([[Int32]].self, forKey: .codes))
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(text, forKey: .text)
        try container.encode(context, forKey: .context)
    }
}

/// A character cast from a description instead of a reference clip.
public struct VoiceDesign: Sendable, Codable, Hashable {
    /// The character's display name, retained in recorded take metadata.
    public var name: String
    /// What the voice sounds like, for example "gravelly old male voice,
    /// weathered and slow".
    public var description: String
    /// An optional scene that frames the performance.
    public var scene: String?
    /// Creates a designed voice.
    public init(name: String, description: String, scene: String? = nil) {
        self.name = name
        self.description = description
        self.scene = scene
    }

    package var systemPrompt: String {
        var lines: [String] = []
        if let scene = scene.map(UnicodeWhitespace.strip), !scene.isEmpty { lines.append("Scene: \(scene)") }
        lines.append("Characters:")
        lines.append("- \(Drama3Prompt.designSpeakerTag): \(UnicodeWhitespace.strip(description))")
        lines.append("Voice every line of the script in order, with \(Drama3Prompt.audioSeparator) between consecutive lines.")
        return lines.joined(separator: "\n")
    }

    package func turn(_ text: String) -> String { "\(UnicodeWhitespace.strip(name)): \(text)" }

    package func removingName(from text: String) -> String {
        let prefix = turn("")
        return text.hasPrefix(prefix) ? String(text.dropFirst(prefix.count)) : text
    }
}

/// One thing to speak, in one voice, with one configuration.
///
/// The voice comes from at most one of `references`, `character`, and the
/// first of `previousTakes`. With none of them the model picks a voice.
public struct SpeechRequest: Sendable {
    /// The script. Performance cues such as `[laughs]`, paired tags such as
    /// `<whisper>…</whisper>`, and `<|speaker:N|>` tags for up to 16
    /// speakers work as written.
    public var text: String
    /// Cloned voices. One is the usual case; several voice a multi-speaker
    /// script in speaker order.
    public var references: [ReferenceVoice]
    /// A designed voice. Mutually exclusive with `references`: setting both
    /// fails with ``PromptError/voiceConflict``.
    public var character: VoiceDesign?
    /// A director's note, such as "softer and warmer". The line is generated
    /// again as one turn, with the note as a direction and as a cue before
    /// every sentence. The note is cut to 120 characters, without `[ ] < > |`.
    public var note: String?
    /// Earlier takes, oldest first; the last two become history. With no
    /// references the first one is the voice reference, which keeps a
    /// designed or model-default voice the same across takes.
    public var previousTakes: [Take]
    /// Sampling and pacing. Defaults to the measured configuration.
    public var config: GenerationConfig
    /// Creates a request.
    public init(text: String, references: [ReferenceVoice] = [], character: VoiceDesign? = nil, note: String? = nil,
                previousTakes: [Take] = [], config: GenerationConfig = GenerationConfig()) {
        self.text = text
        self.references = references
        self.character = character
        self.note = note
        self.previousTakes = previousTakes
        self.config = config
    }
}

/// Where a request's time went, summed over its chunks.
public struct SpeechTimings: Sendable, Equatable {
    /// Seconds spent reading prompts into the model. A 324-row prompt takes
    /// about 0.7 s on an M3 Max.
    public var prefillSeconds: Double = 0
    /// Seconds spent generating frames.
    public var decodeSeconds: Double = 0
    /// Seconds spent in the codec turning frames into audio.
    public var codecSeconds: Double = 0
    /// Seconds spent independently recognizing and comparing generated words.
    public var verificationSeconds: Double = 0
    /// Prompt rows read, summed over chunks.
    public var promptRows: Int = 0
    /// Generation steps, retries included. A step is the engine's
    /// ``EngineCapabilities/samplesPerStep``: a 2048-sample frame for
    /// Drama 3, a 3528-sample latent for Ming.
    public var framesGenerated: Int = 0
    /// Steps generated per second of decode time; 0 before any decode. An
    /// M3 Max generates 33 to 42 Drama 3 frames per second, 21.5 of which make a
    /// second of audio, or about 60 Ming steps, 12.5 of which make a second.
    public var framesPerSecond: Double { decodeSeconds > 0 ? Double(framesGenerated) / decodeSeconds : 0 }
    /// Zero timings.
    public init() {}
}

/// What a running request reports, in order. ``SpeechSynthesizer/synthesize(_:)``
/// streams these and ends with ``finished(_:)``.
public enum SpeechEvent: Sendable {
    /// The model is loading. `progress` runs from 0 to 1; `detail` names the
    /// phase, such as tuning the kernels on a Mac's first load.
    case loadingModel(progress: Double, detail: String)
    /// Chunk `index` of `count` starts. `text` is the script text it speaks.
    case chunkStarted(index: Int, count: Int, text: String)
    /// The chunk's prompt of `rows` rows has been read; 0 from an engine that
    /// does not count prompt rows.
    case prefill(rows: Int)
    /// Step `index` of the current chunk is generated, in the engine's
    /// ``EngineCapabilities/samplesPerStep``. `maxFrames` is the chunk's step cap.
    case frame(index: Int, maxFrames: Int)
    /// The chunk spoke too little of its text and is being generated again.
    case retry(chunkIndex: Int, attempt: Int)
    /// Decoded audio for frames `frameRange` of a chunk still generating,
    /// under ``StreamingMode/lowLatency(stepFrames:)``. A chunk's partials, in
    /// order, join into exactly the samples of its ``audio(samples:chunkIndex:)``.
    case partialAudio(samples: [Float], chunkIndex: Int, frameRange: Range<Int>)
    /// Decoded audio for one finished chunk, 44.1 kHz mono.
    case audio(samples: [Float], chunkIndex: Int)
    /// The request is done. Always the last event.
    case finished(SpeechResult)
}

/// One chunk of a request: the script text it spoke and the length of its
/// audio, so callers can time captions and highlights.
public struct SpokenSentence: Sendable, Codable, Equatable {
    /// The script text of the chunk.
    public var text: String
    /// The chunk's audio length, in 44.1 kHz samples.
    public var samples: Int
    /// Creates a sentence record.
    public init(text: String, samples: Int) {
        self.text = text
        self.samples = samples
    }

    private enum CodingKeys: String, CodingKey { case text, samples, frames }

    /// Records saved before sample timing hold Drama 3 `frames` of 2048 samples.
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        text = try container.decode(String.self, forKey: .text)
        samples = try container.decodeIfPresent(Int.self, forKey: .samples)
            ?? container.decode(Int.self, forKey: .frames) * ModelConstants.samplesPerFrame
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(text, forKey: .text)
        try container.encode(samples, forKey: .samples)
    }
}

/// A finished request.
public struct SpeechResult: Sendable {
    /// Required generation checks each chunk and every complete adjacent-chunk join.
    /// This is bounded context coverage, not a whole-file transcript for long takes.
    /// A recognition match is evidence of word coverage, not an acting-quality
    /// or zero-error guarantee. Mock and preview audio are unchecked.
    public enum WordCoverage: String, Sendable { case recognized, unchecked }
    public var wordCoverage: WordCoverage
    /// The whole request as one take: its turn text and the engine's context for every chunk in order.
    public var take: Take
    /// All of the request's audio, 44.1 kHz mono.
    public var samples: [Float]
    /// Where the time went.
    public var timings: SpeechTimings
    /// True when every chunk ended because the model finished speaking,
    /// rather than at the frame cap or the silence stop.
    public var finishedByEOS: Bool
    /// Every chunk in order; their samples add up to the take's.
    public var sentences: [SpokenSentence]
    /// What the engine could not perform as written. ``SpeechOutline`` reports
    /// the same notices before a run.
    public var notices: [SpeechNotice]
    /// Creates a result.
    public init(take: Take, samples: [Float], timings: SpeechTimings, finishedByEOS: Bool, sentences: [SpokenSentence] = [],
                wordCoverage: WordCoverage = .unchecked, notices: [SpeechNotice] = []) {
        self.wordCoverage = wordCoverage
        self.take = take
        self.samples = samples
        self.timings = timings
        self.finishedByEOS = finishedByEOS
        self.sentences = sentences
        self.notices = notices
    }

    /// Seconds of audio per second of prefill, decode and codec time;
    /// comparable across engines. 0 before any generation.
    public var speed: Double {
        let busy = timings.prefillSeconds + timings.decodeSeconds + timings.codecSeconds
        return busy > 0 ? Double(samples.count) / Double(ModelConstants.sampleRate) / busy : 0
    }
}

/// Something an engine changed about a request because it cannot perform it
/// as written. The words are always kept; only the direction is lost.
public enum SpeechNotice: Sendable, Hashable, Codable {
    /// A cue the engine has no equivalent for. Its markup was removed.
    case unsupportedCue(String)
    /// A cue performed as the nearest thing the engine has.
    case approximatedCue(String, as: String)
    /// A director's note the engine could not turn into a direction.
    case unsupportedNote(String)
    /// Earlier takes from another engine could not carry the voice or history into this request.
    case otherEngineTakes(SpeechEngineID, count: Int)

    /// One sentence for a person.
    public var message: String {
        switch self {
        case .unsupportedCue(let cue): "\(cue) isn't performed by this engine; its words are kept."
        case .approximatedCue(let cue, let tag): "\(cue) is performed as \(tag)."
        case .unsupportedNote(let note): "The note \"\(note)\" names no emotion this engine performs."
        case .otherEngineTakes(let engine, let count): "\(count) earlier \(count == 1 ? "take" : "takes") from \(engine.capabilities.name) can't carry the voice into this one."
        }
    }
}

/// What a synthesizer reports about itself, for status and model listings.
public struct EngineDescription: Sendable, Equatable {
    /// The engine whose voice encodings and takes the synthesizer reads and writes.
    public var engine: SpeechEngineID
    /// The model's public id, such as `fishaudio/drama-3`.
    public var modelID: String
    /// The loaded snapshot's folder name, or `mock`.
    public var snapshot: String
    /// True for a stand-in that runs without the model.
    public var isMock: Bool
    /// Creates a description.
    public init(engine: SpeechEngineID, modelID: String, snapshot: String, isMock: Bool) {
        self.engine = engine
        self.modelID = modelID
        self.snapshot = snapshot
        self.isMock = isMock
    }
}

/// A text-to-speech engine. ``DramaSynthesizer`` is the Metal implementation;
/// ``MockSynthesizer`` stands in for previews and tests.
public protocol SpeechSynthesizer: Sendable {
    /// Which engine this is, and which model it runs.
    var engineDescription: EngineDescription { get }
    /// Nil until the engine loads; then whether its kernels run a schedule
    /// tuned for this Mac. Engines without tuning report false.
    var isTuned: Bool? { get }
    /// Loads weights and compiles kernels, reporting each phase that takes
    /// long enough to name. Safe to call more than once.
    func load(onPhase: @escaping @Sendable (String) -> Void) async throws
    /// Whether ``load(onPhase:)`` has finished.
    var isLoaded: Bool { get async }
    /// Encodes a voice clip, mono 44.1 kHz samples, for this engine.
    /// `transcript` must say what the clip says.
    func encodeVoice(samples: [Float], transcript: String) async throws -> VoiceEncoding
    /// Streams events for one request. Cancel by cancelling the consuming task.
    func synthesize(_ request: SpeechRequest) -> AsyncThrowingStream<SpeechEvent, Error>
}

public extension SpeechSynthesizer {
    /// The engine whose voice encodings and takes this synthesizer reads and writes.
    var engine: SpeechEngineID { engineDescription.engine }

    /// Loads without reporting phases.
    func load() async throws { try await load { _ in } }

    /// Clones a voice from a clip in any format AVFoundation reads. Returns
    /// the voice, encoded for this engine, and the clip as mono 44.1 kHz
    /// samples for the voice library to keep.
    func cloneVoice(audioURL: URL, transcript: String, name: String) async throws -> (voice: ReferenceVoice, clip: [Float]) {
        let clip = try ReferenceAudio.load(url: audioURL)
        let encoding = try await encodeVoice(samples: clip, transcript: transcript)
        return (ReferenceVoice(name: name, transcript: transcript, encodings: [encoding], clipSampleCount: clip.count,
                               sourceFileName: audioURL.lastPathComponent), clip)
    }
}

/// Where the Drama 3 weights live: a snapshot folder holding `config.json`,
/// `tokenizer.json`, `model.safetensors.index.json`, the shards and
/// `codec.pth`. ``DramaSetup/check(location:codecDirectory:tuningDirectory:)``
/// reports what a location is missing.
public struct ModelLocation: Sendable, Equatable {
    /// The snapshot folder.
    public var directory: URL
    /// A location at `directory`. Nothing is read until the engine loads.
    public init(directory: URL) { self.directory = directory }

    /// The newest complete snapshot of `repo` in the ``HuggingFaceCache``,
    /// where `hf download` puts it, or the newest one when none is complete;
    /// nil when there is none.
    public static func huggingFaceCache(repo: String = SpeechEngineID.drama3.install.repo) -> ModelLocation? {
        HuggingFaceCache.latestSnapshot(of: repo) { ModelLocation(directory: $0).isComplete }.map(ModelLocation.init(directory:))
    }

    /// Whether `config.json`, `tokenizer.json`, `model.safetensors.index.json`
    /// and `codec.pth` exist. It does not look at the shards;
    /// ``DramaSetup/check(location:codecDirectory:tuningDirectory:)`` does.
    public var isComplete: Bool {
        let fm = FileManager.default
        let required = ["config.json", "tokenizer.json", "model.safetensors.index.json", "codec.pth"]
        return required.allSatisfy { fm.fileExists(atPath: directory.appendingPathComponent($0).path) }
    }
}

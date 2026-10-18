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

/// How the engine samples and paces one request. The defaults are the
/// configuration the evaluation measured.
///
/// Under the defaults (one sentence per turn, a 2 s silence stop, and
/// retries of short chunks) every one of 130 plain-prose sentences came out
/// as speech in the 330-run reliability study, and the 36-run speech suite
/// scored a mean word error rate of 0.016 with every run reaching the end of
/// speech. The two controls show what the defaults buy: with retries off, 1
/// of 5 runs spoke every sentence; with the reference's 300-byte chunking,
/// 7 of 10 did.
public struct GenerationConfig: Sendable, Codable, Equatable {
    /// Sampling temperature. Default 0.7, the vendor's value.
    public var temperature: Double = 0.7
    /// Top-k candidates kept before sampling. Default 30.
    public var topK: Int = 30
    /// Top-p nucleus kept before sampling. Default 0.7, the vendor's value.
    public var topP: Double = 0.7
    /// The most frames one chunk may run. Defaults to the text's expected
    /// duration times 1.5, plus 2 s.
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
    public var chunking: Chunking = .sentences
    /// Stop a chunk once its decoded audio has been silent for this many
    /// seconds. Default 2. `nil` keeps generating until the end of speech or
    /// the frame cap, like the reference.
    public var stopAfterSilenceSeconds: Double? = 2.0
    /// Generate a chunk again when its spoken audio is shorter than this
    /// fraction of the expected duration (text length at 16 characters per
    /// second). Default 0.6. On the evaluation set every good run scored at
    /// least 0.72 and every failed run at most 0.65. 0 turns retries off.
    public var minSpokenRatio: Double = 0.6
    /// The most times one chunk is generated again. Default 2.
    public var maxRetries: Int = 2
    /// Stop a chunk once its main tokens settle into a repeating cycle, the
    /// shape of a laughter loop. Default true. On the reliability study's two
    /// laughter loops the last 48 main tokens repeated the tokens 17 or 34
    /// frames earlier exactly, while no speech run among 267 matched above
    /// 0.75 at any lag from 17 to 40 frames.
    public var stopOnLoop: Bool = true
    // Experiment knobs of the performance-cue investigation (eval/cues); package-only until one ships.
    /// Also regenerate a chunk that stopped without eos (the silence stop, the
    /// frame cap or the loop guard ended it) when its spoken audio is shorter
    /// than this fraction of the expected duration. 0 leaves the duration rule
    /// alone to decide; `.infinity` regenerates every chunk that misses eos.
    package var retryWithoutEOSBelow: Double = 0
    /// Where a sentence-mode chunk boundary falls around a bracket cue.
    package var cueFold: Drama3Chunking.CueFold = .startsChunk
    /// A text-derived frame cap is never shorter than this many seconds.
    package var capMinimumSeconds: Double = 0
    /// Seconds added to a text-derived frame cap for every `[cue]` in the chunk,
    /// since a laugh, a sigh or a pause takes time that the words do not count.
    package var capSecondsPerCue: Double = 0
    /// The expected duration of a chunk with a slow direction (`slow`, `whisper`,
    /// `drawn-out`, `long pause`) is multiplied by this before the cap is set.
    package var slowCapMultiplier: Double = 1
    /// When decoded audio leaves the engine. Default ``StreamingMode/sentence``.
    public var streaming: StreamingMode = .sentence
    /// The measured defaults.
    public init() {}

    /// How text is cut into generation turns.
    public enum Chunking: String, Sendable, Codable, CaseIterable {
        /// The reference's policy: about ``GenerationConfig/chunkBytes`` per
        /// turn, at most three speakers. It often stops after the first
        /// sentence of a turn; it is here for comparison.
        case reference
        /// One sentence per turn, with the previous turns as history. The
        /// default. ``Drama3Chunking/splitSentencesTagged(_:)`` shows the cut.
        case sentences
    }
}

public extension GenerationConfig {
    /// The middle of the expressiveness range: temperature 0.7 and top-p 0.7.
    static let defaultExpressiveness = 0.5

    /// The two controls the app and the API expose over the defaults. Every
    /// other field keeps the measured value.
    ///
    /// - Parameters:
    ///   - expressiveness: 0 to 1. Temperature and top-p are both
    ///     `0.5 + 0.4 * expressiveness`, so 0.5 is the vendor's sampling.
    ///   - quickPreview: Turns retries off and tightens the frame cap to the
    ///     expected duration times 1.2, plus 1 s, for faster rough takes.
    static func make(expressiveness: Double, quickPreview: Bool) -> GenerationConfig {
        var config = GenerationConfig()
        config.temperature = 0.5 + 0.4 * expressiveness
        config.topP = 0.5 + 0.4 * expressiveness
        if quickPreview {
            config.minSpokenRatio = 0
            config.frameCap = .fromText(margin: 1.2, floorSeconds: 1.0)
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
    case lowLatency(stepFrames: Int)

    /// Partials every 64 frames, about 3 s of audio. Through the HTTP API on
    /// an M3 Max the first audio byte came after 4.49 s instead of 12.95 s.
    /// 32-frame steps started 0.7 s sooner but ran out of audio before the
    /// next partial in every measured trace.
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
    /// duration counts 16 characters per second, leaving out cues and tags.
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

/// A cloned voice: a clip's transcript plus the codec codes of the clip.
/// ``SpeechSynthesizer/encodeReference(audioURL:transcript:name:)`` makes one.
/// It is `Codable`, so a voice library can store it and skip the encoder
/// next time.
public struct ReferenceVoice: Sendable, Codable, Identifiable, Equatable {
    /// A stable identity for a voice library.
    public var id: UUID
    /// A display name.
    public var name: String
    /// What the clip says, word for word. A wrong transcript weakens the clone.
    public var transcript: String
    /// `[10][T]` codec codes, one row per codebook.
    public var codes: [[Int32]]
    /// The clip's file name, for display.
    public var sourceFileName: String?
    /// The clip's length in seconds: frames times 2048 samples at 44.1 kHz.
    public var durationSeconds: Double { Double(codes.first?.count ?? 0) * Double(ModelConstants.samplesPerFrame) / Double(ModelConstants.sampleRate) }

    /// Creates a voice from codes you already have.
    public init(id: UUID = UUID(), name: String, transcript: String, codes: [[Int32]], sourceFileName: String? = nil) {
        self.id = id
        self.name = name
        self.transcript = transcript
        self.codes = codes
        self.sourceFileName = sourceFileName
    }
}

/// One read of a line: the user turn text exactly as it went into the prompt,
/// and the codes the model answered with. Fed back as history or as the voice
/// reference, a take reproduces that turn.
public struct Take: Sendable, Codable, Hashable {
    /// The user turn as the prompt held it, including a character's
    /// `Name: ` or a director's note.
    public var text: String
    /// `[10][T]` codec codes, one row per codebook.
    public var codes: [[Int32]]
    /// Creates a take from stored text and codes.
    public init(text: String, codes: [[Int32]]) {
        self.text = text
        self.codes = codes
    }

    /// Audio frames in the take.
    public var frameCount: Int { codes.first?.count ?? 0 }
    /// The take's length in seconds: frames times 2048 samples at 44.1 kHz.
    public var durationSeconds: Double { Double(frameCount) * Double(ModelConstants.samplesPerFrame) / Double(ModelConstants.sampleRate) }
    /// The codes as model frames, for ``SpeechSynthesizer/decode(frames:)``.
    public var frames: [Frame] {
        (0..<frameCount).map { t in [codes[0][t] + ModelConstants.semanticStart] + codes.map { $0[t] } }
    }
}

/// A character cast from a description instead of a reference clip.
public struct VoiceDesign: Sendable, Codable, Hashable {
    /// The character's name, which the prompt writes before every line.
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
        lines.append("- \(UnicodeWhitespace.strip(name)): \(UnicodeWhitespace.strip(description))")
        lines.append("Voice every line of the script in order, with \(Drama3Prompt.audioSeparator) between consecutive lines.")
        return lines.joined(separator: "\n")
    }

    package func turn(_ text: String) -> String { "\(UnicodeWhitespace.strip(name)): \(text)" }
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
    /// Prompt rows read, summed over chunks.
    public var promptRows: Int = 0
    /// Frames generated, retries included.
    public var framesGenerated: Int = 0
    /// Frames generated per second of decode time; 0 before any decode. An
    /// M3 Max generates 33 to 42 frames per second, 21.5 of which make a
    /// second of audio.
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
    /// The chunk's prompt of `rows` rows has been read.
    case prefill(rows: Int)
    /// Frame `index` of the current chunk is generated. `maxFrames` is the
    /// chunk's frame cap.
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

/// One chunk of a request: the script text it spoke and its frame count, so
/// callers can time captions and highlights.
public struct SpokenSentence: Sendable, Codable, Equatable {
    /// The script text of the chunk.
    public var text: String
    /// Frames the chunk took. Each is 2048 samples at 44.1 kHz.
    public var frames: Int
    /// Creates a sentence record.
    public init(text: String, frames: Int) {
        self.text = text
        self.frames = frames
    }
}

/// A finished request.
public struct SpeechResult: Sendable {
    /// The whole request as one take: its turn text and every chunk's codes in order.
    public var take: Take
    /// All of the request's audio, 44.1 kHz mono.
    public var samples: [Float]
    /// Where the time went.
    public var timings: SpeechTimings
    /// True when every chunk ended because the model finished speaking,
    /// rather than at the frame cap or the silence stop.
    public var finishedByEOS: Bool
    /// Every chunk in order; their frames add up to the take's.
    public var sentences: [SpokenSentence]
    /// Creates a result.
    public init(take: Take, samples: [Float], timings: SpeechTimings, finishedByEOS: Bool, sentences: [SpokenSentence] = []) {
        self.take = take
        self.samples = samples
        self.timings = timings
        self.finishedByEOS = finishedByEOS
        self.sentences = sentences
    }

    /// The take's codes as model frames.
    public var frames: [Frame] { take.frames }
}

/// A text-to-speech engine. ``DramaSynthesizer`` is the Metal implementation;
/// ``MockSynthesizer`` stands in for previews and tests.
public protocol SpeechSynthesizer: Sendable {
    /// Loads weights and compiles kernels, reporting each phase that takes
    /// long enough to name. Safe to call more than once.
    func load(onPhase: @escaping @Sendable (String) -> Void) async throws
    /// Whether ``load(onPhase:)`` has finished.
    var isLoaded: Bool { get async }
    /// Encodes a reference clip (any format AVFoundation reads) to codec codes.
    func encodeReference(audioURL: URL, transcript: String, name: String) async throws -> ReferenceVoice
    /// Decodes codes to 44.1 kHz mono samples, 2048 per frame.
    func decode(frames: [Frame]) async throws -> [Float]
    /// Streams events for one request. Cancel by cancelling the consuming task.
    func synthesize(_ request: SpeechRequest) -> AsyncThrowingStream<SpeechEvent, Error>
}

public extension SpeechSynthesizer {
    /// Loads without reporting phases.
    func load() async throws { try await load { _ in } }
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

    /// The most recently modified snapshot of `repo` under
    /// `~/.cache/huggingface/hub`, where `hf download` puts it; nil when
    /// there is none.
    public static func huggingFaceCache(repo: String = "fishaudio/drama-3") -> ModelLocation? {
        let home = FileManager.default.homeDirectoryForCurrentUser
        let repoDir = home.appendingPathComponent(".cache/huggingface/hub/models--" + repo.replacingOccurrences(of: "/", with: "--") + "/snapshots")
        guard let snapshots = try? FileManager.default.contentsOfDirectory(at: repoDir, includingPropertiesForKeys: [.contentModificationDateKey]),
              let latest = snapshots.max(by: { a, b in
                  let da = (try? a.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
                  let db = (try? b.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
                  return da < db
              }) else { return nil }
        return ModelLocation(directory: latest)
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

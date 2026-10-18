import Foundation

/// One generated or prompt frame: `[main token, c0, ..., c9]`.
/// Text rows carry the text token and ten zeros. Audio rows carry
/// `c0 + 248099` as the main token.
public typealias Frame = [Int32]

public enum ModelConstants {
    public static let numCodebooks = 10
    public static let codebookSize = 4096
    public static let residualCodebookSize = 1024
    public static let semanticStart: Int32 = 248_099
    public static let semanticEnd: Int32 = 252_194
    public static let eos: Int32 = 248_046
    public static let sampleRate = 44_100
    public static let samplesPerFrame = 2048
    public static let maxSpeakerID = 15
    public static let hiddenSize = 2048
}

public struct GenerationConfig: Sendable, Codable, Equatable {
    public var temperature: Double = 0.7
    public var topK: Int = 30
    public var topP: Double = 0.7
    /// The most frames one chunk may run.
    public var frameCap: FrameCap = .fromText(margin: 1.5, floorSeconds: 2.0)
    /// The CLI's `--max-frames`: setting it pins `frameCap` to `.fixed`.
    public var maxNewFrames: Int {
        get {
            switch frameCap {
            case .fixed(let frames): frames
            case .fromText: FrameCap.maximumFrames
            }
        }
        set { frameCap = .fixed(newValue) }
    }
    /// Seed for the sampler. `nil` draws a fresh seed per request.
    public var seed: UInt64? = nil
    /// Cut long text into chunks of about this many UTF-8 bytes (the hosted
    /// server uses 300). Earlier chunks feed back as history.
    public var chunkBytes: Int = 300
    /// How text is cut into generation turns.
    public var chunking: Chunking = .sentences
    /// Stop a chunk once its decoded audio has been silent for this long; nil
    /// keeps generating until eos or the frame cap, like the reference.
    public var stopAfterSilenceSeconds: Double? = 2.0
    /// Regenerate a chunk whose spoken audio is shorter than this fraction of
    /// the expected duration (text length / 16 characters per second). On the
    /// evaluation set every good run scored at least 0.72 and every failed run
    /// at most 0.65. 0 disables retries.
    public var minSpokenRatio: Double = 0.6
    public var maxRetries: Int = 2
    /// Also regenerate a chunk that stopped without eos (the silence stop, the
    /// frame cap or the loop guard ended it) when its spoken audio is shorter
    /// than this fraction of the expected duration. 0 leaves the duration rule
    /// alone to decide; `.infinity` regenerates every chunk that misses eos.
    public var retryWithoutEOSBelow: Double = 0
    /// Stop a chunk whose main tokens have settled into a cycle (a laughter loop).
    public var stopOnLoop: Bool = false
    /// Where a sentence-mode chunk boundary falls around a bracket cue.
    public var cueFold: Drama3Chunking.CueFold = .startsChunk
    /// A text-derived frame cap is never shorter than this many seconds.
    public var capMinimumSeconds: Double = 0
    /// Seconds added to a text-derived frame cap for every `[cue]` in the chunk,
    /// since a laugh, a sigh or a pause takes time that the words do not count.
    public var capSecondsPerCue: Double = 0
    /// The expected duration of a chunk with a slow direction (`slow`, `whisper`,
    /// `drawn-out`, `long pause`) is multiplied by this before the cap is set.
    public var slowCapMultiplier: Double = 1
    /// When decoded audio leaves the engine.
    public var streaming: StreamingMode = .sentence
    public init() {}

    public enum Chunking: String, Sendable, Codable, CaseIterable {
        /// The reference's policy: about `chunkBytes` per turn, at most three speakers.
        case reference
        /// One sentence per turn, with the previous turns as history.
        case sentences
    }
}

public extension GenerationConfig {
    static let defaultExpressiveness = 0.5

    /// The two controls the app and the API expose over the engine defaults;
    /// every other field keeps the value the evaluation measured.
    /// Expressiveness 0.5 is the vendor's sampling, temperature 0.7 and top-p 0.7.
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

public enum StreamingMode: Sendable, Codable, Equatable {
    /// Each chunk's audio arrives once the chunk is generated.
    case sentence
    /// Audio arrives as `.partialAudio` every `stepFrames` generated frames,
    /// decoded on the CPU while the GPU generates. A chunk is never generated
    /// again, because its first audio may already be playing.
    case lowLatency(stepFrames: Int)

    /// About 3 s of audio.
    public static let defaultLowLatency = StreamingMode.lowLatency(stepFrames: 64)

    public var stepFrames: Int? {
        if case .lowLatency(let step) = self { max(1, step) } else { nil }
    }
}

public enum FrameCap: Sendable, Codable, Equatable {
    case fixed(Int)
    /// The text's expected duration times `margin`, plus `floorSeconds`.
    case fromText(margin: Double, floorSeconds: Double)

    /// About 2 s and 27.9 s of audio.
    public static let minimumFrames = 43
    public static let maximumFrames = 600

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

/// A reference voice: transcript plus the codec codes of the clip.
public struct ReferenceVoice: Sendable, Codable, Identifiable, Equatable {
    public var id: UUID
    public var name: String
    public var transcript: String
    /// `[10][T]` codec codes.
    public var codes: [[Int32]]
    public var sourceFileName: String?
    public var durationSeconds: Double { Double(codes.first?.count ?? 0) * Double(ModelConstants.samplesPerFrame) / Double(ModelConstants.sampleRate) }

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
    public var text: String
    /// `[10][T]` codec codes.
    public var codes: [[Int32]]
    public init(text: String, codes: [[Int32]]) {
        self.text = text
        self.codes = codes
    }

    public var frameCount: Int { codes.first?.count ?? 0 }
    public var durationSeconds: Double { Double(frameCount) * Double(ModelConstants.samplesPerFrame) / Double(ModelConstants.sampleRate) }
    public var frames: [Frame] {
        (0..<frameCount).map { t in [codes[0][t] + ModelConstants.semanticStart] + codes.map { $0[t] } }
    }
}

/// A character cast from a description instead of a reference clip.
public struct VoiceDesign: Sendable, Codable, Hashable {
    public var name: String
    public var description: String
    public var scene: String?
    public init(name: String, description: String, scene: String? = nil) {
        self.name = name
        self.description = description
        self.scene = scene
    }

    /// The system message text that replaces the usual instruction.
    public var systemPrompt: String {
        var lines: [String] = []
        if let scene = scene.map(UnicodeWhitespace.strip), !scene.isEmpty { lines.append("Scene: \(scene)") }
        lines.append("Characters:")
        lines.append("- \(UnicodeWhitespace.strip(name)): \(UnicodeWhitespace.strip(description))")
        lines.append("Voice every line of the script in order, with \(Drama3Prompt.audioSeparator) between consecutive lines.")
        return lines.joined(separator: "\n")
    }

    /// A user turn spoken by this character.
    public func turn(_ text: String) -> String { "\(UnicodeWhitespace.strip(name)): \(text)" }
}

public struct SpeechRequest: Sendable {
    public var text: String
    public var references: [ReferenceVoice]
    /// Mutually exclusive with `references`.
    public var character: VoiceDesign?
    /// A director's note: the line is generated again as one turn with the
    /// note written into it.
    public var note: String?
    /// Earlier takes, oldest first; the last two become history. With no
    /// references the first one is the voice reference, which keeps a
    /// designed or model-default voice the same across takes.
    public var previousTakes: [Take]
    public var config: GenerationConfig
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

public struct SpeechTimings: Sendable, Equatable {
    public var prefillSeconds: Double = 0
    public var decodeSeconds: Double = 0
    public var codecSeconds: Double = 0
    public var promptRows: Int = 0
    public var framesGenerated: Int = 0
    public var framesPerSecond: Double { decodeSeconds > 0 ? Double(framesGenerated) / decodeSeconds : 0 }
    public init() {}
}

public enum SpeechEvent: Sendable {
    case loadingModel(progress: Double, detail: String)
    case chunkStarted(index: Int, count: Int, text: String)
    case prefill(rows: Int)
    case frame(index: Int, maxFrames: Int)
    /// The chunk spoke too little of its text and is being generated again.
    case retry(chunkIndex: Int, attempt: Int)
    /// Decoded audio for frames `frameRange` of a chunk still generating,
    /// under `.lowLatency`. A chunk's partials, in order, join into exactly the
    /// samples of its `.audio`.
    case partialAudio(samples: [Float], chunkIndex: Int, frameRange: Range<Int>)
    /// Decoded audio for one finished chunk, 44.1 kHz mono.
    case audio(samples: [Float], chunkIndex: Int)
    case finished(SpeechResult)
}

/// One chunk of a request: the script text it spoke and its frame count, so
/// callers can time captions and highlights.
public struct SpokenSentence: Sendable, Codable, Equatable {
    public var text: String
    public var frames: Int
    public init(text: String, frames: Int) {
        self.text = text
        self.frames = frames
    }
}

public struct SpeechResult: Sendable {
    /// The whole request as one take: its turn text and every chunk's codes in order.
    public var take: Take
    /// 44.1 kHz mono.
    public var samples: [Float]
    public var timings: SpeechTimings
    public var finishedByEOS: Bool
    /// Every chunk in order; their frames add up to the take's.
    public var sentences: [SpokenSentence]
    public init(take: Take, samples: [Float], timings: SpeechTimings, finishedByEOS: Bool, sentences: [SpokenSentence] = []) {
        self.take = take
        self.samples = samples
        self.timings = timings
        self.finishedByEOS = finishedByEOS
        self.sentences = sentences
    }

    public var frames: [Frame] { take.frames }
}

/// What the app talks to. `DramaEngine` is the Metal implementation; tests and
/// previews use a mock.
public protocol SpeechSynthesizer: Sendable {
    /// Loads weights and compiles kernels, reporting each phase that takes
    /// long enough to name. Safe to call more than once.
    func load(onPhase: @escaping @Sendable (String) -> Void) async throws
    var isLoaded: Bool { get async }
    /// Encodes a reference clip (any format AVFoundation reads) to codec codes.
    func encodeReference(audioURL: URL, transcript: String, name: String) async throws -> ReferenceVoice
    /// Decodes codes to 44.1 kHz samples (used for previewing a reference).
    func decode(frames: [Frame]) async throws -> [Float]
    /// Streams events for one request. Cancel by cancelling the consuming task.
    func synthesize(_ request: SpeechRequest) -> AsyncThrowingStream<SpeechEvent, Error>
}

public extension SpeechSynthesizer {
    func load() async throws { try await load { _ in } }
}

/// Where the weights live. Defaults to the Hugging Face cache snapshot.
public struct ModelLocation: Sendable, Equatable {
    public var directory: URL
    public init(directory: URL) { self.directory = directory }

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

    public var isComplete: Bool {
        let fm = FileManager.default
        let required = ["config.json", "tokenizer.json", "model.safetensors.index.json", "codec.pth"]
        return required.allSatisfy { fm.fileExists(atPath: directory.appendingPathComponent($0).path) }
    }
}

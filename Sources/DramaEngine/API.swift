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
    /// Frames per chunk. 600 frames is about 27.9 s of audio.
    public var maxNewFrames: Int = 600
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
    public init() {}

    public enum Chunking: String, Sendable, Codable, CaseIterable {
        /// The reference's policy: about `chunkBytes` per turn, at most three speakers.
        case reference
        /// One sentence per turn, with the previous turns as history.
        case sentences
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

public struct SpeechRequest: Sendable {
    public var text: String
    public var references: [ReferenceVoice]
    public var config: GenerationConfig
    public init(text: String, references: [ReferenceVoice] = [], config: GenerationConfig = GenerationConfig()) {
        self.text = text
        self.references = references
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
    /// Decoded audio for one finished chunk, 44.1 kHz mono.
    case audio(samples: [Float], chunkIndex: Int)
    case finished(SpeechResult)
}

public struct SpeechResult: Sendable {
    /// All generated frames across chunks, in order.
    public var frames: [Frame]
    /// 44.1 kHz mono.
    public var samples: [Float]
    public var timings: SpeechTimings
    public var finishedByEOS: Bool
    public init(frames: [Frame], samples: [Float], timings: SpeechTimings, finishedByEOS: Bool) {
        self.frames = frames
        self.samples = samples
        self.timings = timings
        self.finishedByEOS = finishedByEOS
    }
}

/// What the app talks to. `DramaEngine` is the Metal implementation; tests and
/// previews use a mock.
public protocol SpeechSynthesizer: Sendable {
    /// Loads weights and compiles kernels. Safe to call more than once.
    func load() async throws
    var isLoaded: Bool { get async }
    /// Encodes a reference clip (any format AVFoundation reads) to codec codes.
    func encodeReference(audioURL: URL, transcript: String, name: String) async throws -> ReferenceVoice
    /// Decodes codes to 44.1 kHz samples (used for previewing a reference).
    func decode(frames: [Frame]) async throws -> [Float]
    /// Streams events for one request. Cancel by cancelling the consuming task.
    func synthesize(_ request: SpeechRequest) -> AsyncThrowingStream<SpeechEvent, Error>
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

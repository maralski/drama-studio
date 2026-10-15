import Foundation

/// Text in, codec frames out, chunk by chunk, following the reference's
/// long-text loop: each chunk is generated with the pinned first chunk and the
/// latest chunk as history.
public final class FrameGenerator {
    public let model: DramaModel
    public let tokenizer: Tokenizer

    public struct ChunkResult: Sendable {
        public var index: Int
        public var count: Int
        public var text: String
        public var frames: [Frame]
        public var promptRows: Int
        public var finishedByEOS: Bool
        public var prefillSeconds: Double
        public var decodeSeconds: Double
        /// `[10][T]` codes, the shape the codec and history take.
        public var codes: [[Int32]] { (0..<ModelConstants.numCodebooks).map { c in frames.map { $0[1 + c] } } }
    }

    public enum Event: Sendable {
        case chunkStarted(index: Int, count: Int, text: String)
        case prefilled(rows: Int)
        case frame(index: Int, maxFrames: Int)
        case chunkFinished(ChunkResult)
    }

    /// Decides whether the tail of a chunk is silence. The synthesizer decodes
    /// the frames with the codec and measures the last seconds; nil disables
    /// the early stop.
    public var silenceProbe: ((_ frames: [Frame], _ tailSeconds: Double) throws -> Bool)?
    static let silenceProbeEveryFrames = 16

    public init(model: DramaModel, tokenizer: Tokenizer) {
        self.model = model
        self.tokenizer = tokenizer
    }

    /// The prompt for one chunk with as much history as fits.
    func chunkPrompt(text: String, references: [Drama3Prompt.Reference], history: Drama3History) throws -> (rows: [Frame], prefixRows: Int) {
        var lastError: Error = PromptError.tooLong(0)
        for turns in history.candidates {
            do { return try Drama3Prompt.buildWithPrefix(text: text, tokenizer: tokenizer, references: references, history: turns) }
            catch PromptError.tooLong(let rows) { lastError = PromptError.tooLong(rows) }
        }
        throw lastError
    }

    /// Generates every chunk. `onEvent` runs on the calling thread; return
    /// false from `shouldContinue` to stop early.
    public func generate(text: String, references: [Drama3Prompt.Reference], config: GenerationConfig,
                         shouldContinue: () -> Bool = { true }, onEvent: (Event) -> Void) throws -> [ChunkResult] {
        var chunks: [String]
        switch config.chunking {
        case .reference: chunks = Drama3Chunking.split(text, maxBytes: config.chunkBytes)
        case .sentences: chunks = Drama3Chunking.splitSentencesTagged(text)
        }
        if chunks.isEmpty { chunks = [text] }
        for chunk in chunks {
            let rows = try Drama3Prompt.build(text: chunk, tokenizer: tokenizer, references: references).count
            guard rows < model.capacity else { throw PromptError.tooLong(rows) }
        }
        var sampler = Sampler(seed: config.seed ?? UInt64.random(in: 0...UInt64.max))
        var history = Drama3History()
        var results: [ChunkResult] = []
        for (index, chunk) in chunks.enumerated() {
            guard shouldContinue() else { break }
            onEvent(.chunkStarted(index: index, count: chunks.count, text: chunk))
            let (prompt, prefixRows) = try chunkPrompt(text: chunk, references: references, history: history)
            let budget = min(config.maxNewFrames, model.capacity - prompt.count - 1)
            guard budget >= 1 else { throw PromptError.tooLong(prompt.count) }
            let t0 = Date()
            let session = try GenerationSession(model: model, prompt: prompt, prefixRows: prefixRows)
            let prefillSeconds = Date().timeIntervalSince(t0)
            onEvent(.prefilled(rows: prompt.count))
            var frames: [Frame] = []
            var eos = false
            let t1 = Date()
            let silenceFrames = config.stopAfterSilenceSeconds.map { Int(($0 * Double(ModelConstants.sampleRate) / Double(ModelConstants.samplesPerFrame)).rounded(.up)) }
            for step in 0..<budget {
                guard shouldContinue() else { break }
                let r = try session.step(temperature: config.temperature, topK: config.topK, topP: config.topP, sampler: &sampler)
                guard let frame = r.frame else { eos = true; break }
                frames.append(frame)
                onEvent(.frame(index: step, maxFrames: budget))
                if let probe = silenceProbe, let silenceFrames, frames.count > silenceFrames, frames.count % Self.silenceProbeEveryFrames == 0,
                   try probe(frames, config.stopAfterSilenceSeconds!) {
                    break
                }
            }
            let result = ChunkResult(index: index, count: chunks.count, text: chunk, frames: frames, promptRows: prompt.count,
                                     finishedByEOS: eos, prefillSeconds: prefillSeconds, decodeSeconds: Date().timeIntervalSince(t1))
            if !frames.isEmpty { history.push(text: chunk, codes: result.codes) }
            results.append(result)
            onEvent(.chunkFinished(result))
        }
        return results
    }
}

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
        /// GPU time of the decode command buffers.
        public var gpuSeconds: Double
        public var spokenSeconds: Double
        public var attempts: Int
        /// `[10][T]` codes, the shape the codec and history take.
        public var codes: [[Int32]] { (0..<ModelConstants.numCodebooks).map { c in frames.map { $0[1 + c] } } }
    }

    public enum Event: Sendable {
        case chunkStarted(index: Int, count: Int, text: String)
        case prefilled(rows: Int)
        case frame(index: Int, maxFrames: Int)
        case retry(index: Int, attempt: Int, spokenSeconds: Double, expectedSeconds: Double)
        case chunkFinished(ChunkResult)
    }

    /// Seconds of silence at the end of the chunk so far. The synthesizer
    /// decodes the frames with the codec and measures; nil disables the early
    /// stop and the spoken-duration check falls back to the frame count.
    public var silenceProbe: ((_ frames: [Frame]) throws -> Double)?
    static let silenceProbeEveryFrames = 16

    /// Runs the probe off the generation thread so the CPU codec overlaps the GPU.
    final class ProbeRunner {
        private let probe: ([Frame]) throws -> Double
        private let lock = NSLock()
        private var inFlight = false
        private var latest: Double = 0
        private let group = DispatchGroup()
        init(_ probe: @escaping ([Frame]) throws -> Double) { self.probe = probe }

        /// Starts a probe on a snapshot unless one is already running.
        func offer(_ frames: [Frame]) {
            lock.lock()
            if inFlight { lock.unlock(); return }
            inFlight = true
            lock.unlock()
            group.enter()
            DispatchQueue.global(qos: .userInitiated).async { [self] in
                let r = (try? probe(frames)) ?? 0
                lock.lock(); latest = r; inFlight = false; lock.unlock()
                group.leave()
            }
        }

        var latestSilence: Double { lock.lock(); defer { lock.unlock() }; return latest }
        func drain() { group.wait() }
    }
    static let charactersPerSecond = 16.0

    public static func expectedSeconds(for text: String) -> Double {
        let plain = text.replacingOccurrences(of: #"<\|[^|]*\|>|\[[^\]]*\]"#, with: "", options: .regularExpression)
        let chars = plain.split(whereSeparator: { $0.isWhitespace }).joined(separator: " ").count
        return Double(chars) / charactersPerSecond
    }

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
            let expected = Self.expectedSeconds(for: chunk)
            let budget = min(config.frameCap.frames(expectedSeconds: expected), model.capacity - prompt.count - 1)
            guard budget >= 1 else { throw PromptError.tooLong(prompt.count) }
            var best: ChunkResult?
            for attempt in 0...max(0, config.maxRetries) {
                guard shouldContinue() else { break }
                let t0 = Date()
                let session = try GenerationSession(model: model, prompt: prompt, prefixRows: prefixRows)
                let prefillSeconds = Date().timeIntervalSince(t0)
                if attempt == 0 { onEvent(.prefilled(rows: prompt.count)) }
                var frames: [Frame] = []
                var eos = false
                var trailingSilence = 0.0
                let t1 = Date()
                let silenceFrames = config.stopAfterSilenceSeconds.map { Int(($0 * Double(ModelConstants.sampleRate) / Double(ModelConstants.samplesPerFrame)).rounded(.up)) }
                let runner = silenceProbe.map(ProbeRunner.init)
                for step in 0..<budget {
                    guard shouldContinue() else { break }
                    let r = try session.step(temperature: config.temperature, topK: config.topK, topP: config.topP, sampler: &sampler)
                    guard let frame = r.frame else { eos = true; break }
                    frames.append(frame)
                    onEvent(.frame(index: step, maxFrames: budget))
                    if let runner, let silenceFrames, let limit = config.stopAfterSilenceSeconds, frames.count > silenceFrames {
                        if frames.count % Self.silenceProbeEveryFrames == 0 { runner.offer(frames) }
                        if runner.latestSilence >= limit { break }
                    }
                }
                runner?.drain()
                if let probe = silenceProbe, !frames.isEmpty { trailingSilence = try probe(frames) }
                let total = Double(frames.count) * Double(ModelConstants.samplesPerFrame) / Double(ModelConstants.sampleRate)
                let spoken = max(0, total - trailingSilence)
                let result = ChunkResult(index: index, count: chunks.count, text: chunk, frames: frames, promptRows: prompt.count,
                                         finishedByEOS: eos, prefillSeconds: prefillSeconds, decodeSeconds: Date().timeIntervalSince(t1),
                                         gpuSeconds: session.gpuSeconds, spokenSeconds: spoken, attempts: attempt + 1)
                if best == nil || spoken > best!.spokenSeconds { best = result }
                if config.minSpokenRatio <= 0 || spoken >= config.minSpokenRatio * expected || attempt == config.maxRetries { break }
                onEvent(.retry(index: index, attempt: attempt + 1, spokenSeconds: spoken, expectedSeconds: expected))
            }
            guard let result = best else { break }
            let frames = result.frames
            if !frames.isEmpty { history.push(text: chunk, codes: result.codes) }
            results.append(result)
            onEvent(.chunkFinished(result))
        }
        return results
    }
}

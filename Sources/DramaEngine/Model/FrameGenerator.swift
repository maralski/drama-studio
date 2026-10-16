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
        /// The chunk's decoded audio; empty without a `codec`.
        public var samples: [Float]
        public var codecSeconds: Double
        /// `[10][T]` codes, the shape the codec and history take.
        public var codes: [[Int32]] { (0..<ModelConstants.numCodebooks).map { c in frames.map { $0[1 + c] } } }
    }

    public enum Event: Sendable {
        case chunkStarted(index: Int, count: Int, text: String)
        case prefilled(rows: Int)
        case frame(index: Int, maxFrames: Int)
        case retry(index: Int, attempt: Int, spokenSeconds: Double, expectedSeconds: Double)
        /// Under `.lowLatency`, audio of frames `frames` of the chunk generating.
        case partialAudio(samples: [Float], index: Int, frames: Range<Int>)
        case chunkFinished(ChunkResult)
    }

    /// Decodes one window of a chunk's codes. Set, the generator decodes each
    /// chunk while it generates: the early stop measures the silence of that
    /// audio, `.lowLatency` streams it, and `ChunkResult.samples` carries it.
    public var codec: ((_ codes: [[Int32]], _ window: DecodeWindow) throws -> [Float])?
    /// Seconds of silence at the end of the chunk so far, for callers without
    /// a `codec`. With neither, the early stop is off and the spoken-duration
    /// check falls back to the frame count.
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
        @discardableResult
        func offer(_ frames: [Frame]) -> Bool {
            lock.lock()
            if inFlight { lock.unlock(); return false }
            inFlight = true
            lock.unlock()
            group.enter()
            DispatchQueue.global(qos: .userInitiated).async { [self] in
                let r = (try? probe(frames)) ?? 0
                lock.lock(); latest = r; inFlight = false; lock.unlock()
                group.leave()
            }
            return true
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

    /// The prompt for one chunk with as much history as fits: the plan's
    /// earlier takes go first and are the first thing dropped.
    func chunkPrompt(text: String, plan: GenerationPlan, history: Drama3History) throws -> (rows: [Frame], prefixRows: Int) {
        var lastError: Error = PromptError.tooLong(0)
        let candidates = history.candidates.map { plan.history + $0 } + (plan.history.isEmpty ? [] : history.candidates)
        for turns in candidates {
            do { return try Drama3Prompt.buildWithPrefix(text: text, tokenizer: tokenizer, references: plan.references, character: plan.character, history: turns) }
            catch PromptError.tooLong(let rows) { lastError = PromptError.tooLong(rows) }
        }
        throw lastError
    }

    /// Generates every chunk of the plan. `onEvent` runs on the calling
    /// thread; return false from `shouldContinue` to stop early.
    public func generate(text: String, references: [Drama3Prompt.Reference], character: VoiceDesign? = nil, previousTakes: [Take] = [],
                         note: String? = nil, config: GenerationConfig,
                         shouldContinue: () -> Bool = { true }, onEvent: (Event) -> Void) throws -> [ChunkResult] {
        try generate(plan: GenerationPlan(text: text, references: references, character: character, previousTakes: previousTakes, note: note, config: config),
                     config: config, shouldContinue: shouldContinue, onEvent: onEvent)
    }

    public func generate(plan: GenerationPlan, config: GenerationConfig,
                         shouldContinue: () -> Bool = { true }, onEvent: (Event) -> Void) throws -> [ChunkResult] {
        let chunks = plan.chunks
        for chunk in chunks {
            let rows = try Drama3Prompt.build(text: chunk.turn, tokenizer: tokenizer, references: plan.references, character: plan.character).count
            guard rows < model.capacity else { throw PromptError.tooLong(rows) }
        }
        var sampler = Sampler(seed: config.seed ?? UInt64.random(in: 0...UInt64.max))
        var history = Drama3History()
        var results: [ChunkResult] = []
        for (index, chunk) in chunks.enumerated() {
            guard shouldContinue() else { break }
            onEvent(.chunkStarted(index: index, count: chunks.count, text: chunk.line))
            let (prompt, prefixRows) = try chunkPrompt(text: chunk.turn, plan: plan, history: history)
            let expected = Self.expectedSeconds(for: chunk.line)
            let budget = min(config.frameCap.frames(expectedSeconds: expected), model.capacity - prompt.count - 1)
            guard budget >= 1 else { throw PromptError.tooLong(prompt.count) }
            var best: ChunkResult?
            let streamStep = config.streaming.stepFrames
            let maxRetries = streamStep == nil ? max(0, config.maxRetries) : 0
            for attempt in 0...maxRetries {
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
                let decoder = codec.map(ChunkDecoder.init)
                let probe = decoder.map { decoder in { try decoder.advance(to: $0) } } ?? silenceProbe
                let runner = probe.map(ProbeRunner.init)
                func deliverPartials() {
                    guard streamStep != nil, let decoder else { return }
                    for piece in decoder.takePieces() { onEvent(.partialAudio(samples: piece.samples, index: index, frames: piece.frames)) }
                }
                var offered = 0
                for step in 0..<budget {
                    guard shouldContinue() else { break }
                    let r = try session.step(temperature: config.temperature, topK: config.topK, topP: config.topP, sampler: &sampler)
                    guard let frame = r.frame else { eos = true; break }
                    frames.append(frame)
                    onEvent(.frame(index: step, maxFrames: budget))
                    deliverPartials()
                    guard let runner else { continue }
                    let probing = silenceFrames.map { frames.count > $0 } ?? false
                    let due = if let streamStep, decoder != nil { frames.count - offered >= streamStep }
                              else { probing && frames.count % Self.silenceProbeEveryFrames == 0 }
                    if due, runner.offer(frames) { offered = frames.count }
                    if probing, let limit = config.stopAfterSilenceSeconds, runner.latestSilence >= limit { break }
                }
                runner?.drain()
                if let probe, !frames.isEmpty { trailingSilence = try probe(frames) }
                deliverPartials()
                let total = Double(frames.count) * Double(ModelConstants.samplesPerFrame) / Double(ModelConstants.sampleRate)
                let spoken = max(0, total - trailingSilence)
                let result = ChunkResult(index: index, count: chunks.count, text: chunk.turn, frames: frames, promptRows: prompt.count,
                                         finishedByEOS: eos, prefillSeconds: prefillSeconds, decodeSeconds: Date().timeIntervalSince(t1),
                                         gpuSeconds: session.gpuSeconds, spokenSeconds: spoken, attempts: attempt + 1,
                                         samples: decoder?.samples ?? [], codecSeconds: decoder?.codecSeconds ?? 0)
                if best == nil || spoken > best!.spokenSeconds { best = result }
                if config.minSpokenRatio <= 0 || spoken >= config.minSpokenRatio * expected || attempt == maxRetries { break }
                onEvent(.retry(index: index, attempt: attempt + 1, spokenSeconds: spoken, expectedSeconds: expected))
            }
            guard let result = best else { break }
            let frames = result.frames
            if !frames.isEmpty { history.push(text: chunk.turn, codes: result.codes) }
            results.append(result)
            onEvent(.chunkFinished(result))
        }
        return results
    }
}

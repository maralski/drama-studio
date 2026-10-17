import Foundation

/// Text in, codec frames out, chunk by chunk, following the reference's
/// long-text loop: each chunk is generated with the pinned first chunk and the
/// latest chunk as history.
package final class FrameGenerator {
    package let model: DramaModel
    package let tokenizer: Tokenizer

    package struct ChunkResult: Sendable {
        package var index: Int
        package var count: Int
        package var text: String
        package var frames: [Frame]
        package var promptRows: Int
        package var finishedByEOS: Bool
        package var prefillSeconds: Double
        package var decodeSeconds: Double
        /// GPU time of the decode command buffers.
        package var gpuSeconds: Double
        package var spokenSeconds: Double
        package var attempts: Int
        /// The chunk's decoded audio; empty without a `codec`.
        package var samples: [Float]
        package var codecSeconds: Double
        package var durationSeconds: Double { Double(frames.count) * Double(ModelConstants.samplesPerFrame) / Double(ModelConstants.sampleRate) }
        /// `[10][T]` codes, the shape the codec and history take.
        package var codes: [[Int32]] { (0..<ModelConstants.numCodebooks).map { c in frames.map { $0[1 + c] } } }
    }

    package enum Event: Sendable {
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
    /// Without it, the early stop is off and the spoken-duration check falls
    /// back to the frame count.
    package var codec: ((_ codes: [[Int32]], _ window: DecodeWindow) throws -> [Float])?
    static let silenceProbeEveryFrames = 16

    static let charactersPerSecond = 16.0

    package static func expectedSeconds(for text: String) -> Double {
        let plain = text.replacingOccurrences(of: #"<\|[^|]*\|>|\[[^\]]*\]"#, with: "", options: .regularExpression)
        let chars = plain.split(whereSeparator: { $0.isWhitespace }).joined(separator: " ").count
        return Double(chars) / charactersPerSecond
    }

    package init(model: DramaModel, tokenizer: Tokenizer) {
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
    package func generate(text: String, references: [Drama3Prompt.Reference], character: VoiceDesign? = nil, previousTakes: [Take] = [],
                         note: String? = nil, config: GenerationConfig,
                         shouldContinue: () -> Bool = { true }, onEvent: (Event) -> Void) throws -> [ChunkResult] {
        try generate(plan: GenerationPlan(text: text, references: references, character: character, previousTakes: previousTakes, note: note, config: config),
                     config: config, shouldContinue: shouldContinue, onEvent: onEvent)
    }

    package func generate(plan: GenerationPlan, config: GenerationConfig,
                         shouldContinue: () -> Bool = { true }, onEvent: (Event) -> Void) throws -> [ChunkResult] {
        let chunks = plan.chunks
        for chunk in chunks {
            let rows = try Drama3Prompt.build(text: chunk.turn, tokenizer: tokenizer, references: plan.references, character: plan.character).count
            guard rows < model.capacity else { throw PromptError.tooLong(rows) }
        }
        var sampler = Sampler(seed: config.seed ?? UInt64.random(in: 0...UInt64.max))
        var history = Drama3History()
        var results: [ChunkResult] = []
        let runner = DecodeRunner()
        let streamStep = config.streaming.stepFrames
        let maxRetries = streamStep == nil ? max(0, config.maxRetries) : 0
        /// Streamed chunks whose last frames are still decoding, oldest first.
        var finishing: [(result: ChunkResult, decoder: ChunkDecoder, probe: SilenceProbe)] = []

        func finish(_ result: ChunkResult) {
            results.append(result)
            onEvent(.chunkFinished(result))
        }
        /// Emits decoded audio in order: the finishing chunks first, then `current`.
        func deliver(current: (index: Int, decoder: ChunkDecoder?)? = nil) throws {
            while let (result, decoder, probe) = finishing.first {
                for piece in decoder.takePieces() { onEvent(.partialAudio(samples: piece.samples, index: result.index, frames: piece.frames)) }
                guard let silence = try probe.final() else { return }
                finishing.removeFirst()
                var finished = result
                finished.spokenSeconds = max(0, finished.durationSeconds - silence)
                finished.samples = decoder.samples
                finished.codecSeconds = decoder.codecSeconds
                finish(finished)
            }
            guard streamStep != nil, let current, let decoder = current.decoder else { return }
            for piece in decoder.takePieces() { onEvent(.partialAudio(samples: piece.samples, index: current.index, frames: piece.frames)) }
        }

        for (index, chunk) in chunks.enumerated() {
            guard shouldContinue() else { break }
            onEvent(.chunkStarted(index: index, count: chunks.count, text: chunk.line))
            let (prompt, prefixRows) = try chunkPrompt(text: chunk.turn, plan: plan, history: history)
            let expected = Self.expectedSeconds(for: chunk.line)
            let budget = min(config.frameCap.frames(expectedSeconds: expected), model.capacity - prompt.count - 1)
            guard budget >= 1 else { throw PromptError.tooLong(prompt.count) }
            var best: ChunkResult?
            for attempt in 0...maxRetries {
                guard shouldContinue() else { break }
                let t0 = Date()
                let session = try GenerationSession(model: model, prompt: prompt, prefixRows: prefixRows)
                let prefillSeconds = Date().timeIntervalSince(t0)
                if attempt == 0 { onEvent(.prefilled(rows: prompt.count)) }
                var frames: [Frame] = []
                var eos = false
                let t1 = Date()
                let silenceFrames = config.stopAfterSilenceSeconds.map { Int(($0 * Double(ModelConstants.sampleRate) / Double(ModelConstants.samplesPerFrame)).rounded(.up)) }
                let decoder = codec.map(ChunkDecoder.init)
                let probe = decoder.map { decoder in SilenceProbe { try decoder.advance(to: $0) } }
                var offered = 0
                for step in 0..<budget {
                    guard shouldContinue() else { break }
                    let r = try session.step(temperature: config.temperature, topK: config.topK, topP: config.topP, sampler: &sampler)
                    guard let frame = r.frame else { eos = true; break }
                    frames.append(frame)
                    onEvent(.frame(index: step, maxFrames: budget))
                    try deliver(current: (index, decoder))
                    guard let probe else { continue }
                    let probing = silenceFrames.map { frames.count > $0 } ?? false
                    let due = if let streamStep, decoder != nil { frames.count - offered >= streamStep }
                              else { probing && frames.count % Self.silenceProbeEveryFrames == 0 }
                    let snapshot = frames
                    if due, runner.offer({ probe.measure(snapshot) }) { offered = frames.count }
                    if probing, let limit = config.stopAfterSilenceSeconds, probe.latest >= limit { break }
                }
                var result = ChunkResult(index: index, count: chunks.count, text: chunk.turn, frames: frames, promptRows: prompt.count,
                                         finishedByEOS: eos, prefillSeconds: prefillSeconds, decodeSeconds: Date().timeIntervalSince(t1),
                                         gpuSeconds: session.gpuSeconds, spokenSeconds: 0, attempts: attempt + 1, samples: [], codecSeconds: 0)
                if streamStep != nil, let decoder, let probe, !frames.isEmpty {
                    // No retry can follow, so the GPU moves on while the CPU decodes the last frames.
                    let snapshot = frames
                    runner.enqueue { probe.measureFinal(snapshot) }
                    finishing.append((result, decoder, probe))
                    best = result
                    break
                }
                runner.drain()
                try deliver(current: (index, decoder))
                let trailingSilence = try probe.flatMap { probe in frames.isEmpty ? nil : try probe.measureNow(frames) } ?? 0
                try deliver(current: (index, decoder))
                result.spokenSeconds = max(0, result.durationSeconds - trailingSilence)
                result.samples = decoder?.samples ?? []
                result.codecSeconds = decoder?.codecSeconds ?? 0
                if best == nil || result.spokenSeconds > best!.spokenSeconds { best = result }
                if config.minSpokenRatio <= 0 || result.spokenSeconds >= config.minSpokenRatio * expected || attempt == maxRetries { break }
                onEvent(.retry(index: index, attempt: attempt + 1, spokenSeconds: result.spokenSeconds, expectedSeconds: expected))
            }
            guard let result = best else { break }
            if !result.frames.isEmpty { history.push(text: chunk.turn, codes: result.codes) }
            if !finishing.contains(where: { $0.result.index == index }) { finish(result) }
        }
        while let next = finishing.first {
            next.probe.waitForFinal()
            try deliver()
        }
        runner.drain()
        return results
    }
}

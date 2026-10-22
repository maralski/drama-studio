import Foundation

/// Text in, codec frames out, chunk by chunk, following the reference's
/// long-text loop: each chunk is generated with the pinned first chunk and the
/// latest chunk as history.
package final class FrameGenerator {
    package let model: DramaModel
    package let tokenizer: Tokenizer

    /// Why an attempt at a chunk stopped.
    package enum ChunkEnd: Sendable {
        case eos, loopGuard, silence, frameCap, cancelled
    }

    package struct ChunkResult: Sendable {
        package var index: Int
        package var count: Int
        package var text: String
        package var frames: [Frame]
        package var promptRows: Int
        package var end: ChunkEnd
        package var finishedByEOS: Bool { end == .eos }
        /// The most frames this chunk could run.
        package var frameBudget: Int
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
        /// The frame count at which this attempt's audio first ended in
        /// `silenceSeconds` of silence, where the silence stop should have
        /// fired; nil when it never did or `samples` is empty. The codec is
        /// causal, so the audio of the first k frames is the first k frames
        /// of `samples`.
        package func silenceStopFrame(silenceSeconds: Double) -> Int? {
            FrameGenerator.silenceStopFrame(samples: samples, silenceFrames: FrameGenerator.silenceFrames(for: silenceSeconds))
        }
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

    /// Decodes one window of a chunk's codes, through the smallest decoder
    /// shape the window fits. Set, the silence stop reads the newest frames
    /// of each chunk through the probe shape every `silenceProbeEveryFrames`
    /// frames, `.lowLatency` streams the chunk's audio as it generates, and
    /// `ChunkResult.samples` carries it. Without it, the early stop is off
    /// and the spoken-duration check falls back to the frame count.
    package var codec: ((_ codes: [[Int32]], _ window: DecodeWindow) throws -> [Float])?
    /// How often the silence probe reads the newest frames. With the probe
    /// shape of 64 frames, a reading costs well under the 0.4 s these frames
    /// take to generate, so the stop fires within about a reading of the
    /// audio going silent.
    package static let silenceProbeEveryFrames = 16
    /// Set, every decode session records its per-frame timestamps here.
    package var timeline: FrameTimeline?

    static let charactersPerSecond = 16.0

    /// The silent frames the stop needs: `seconds` rounded up to whole frames.
    package static func silenceFrames(for seconds: Double) -> Int {
        Int((seconds * Double(ModelConstants.sampleRate) / Double(ModelConstants.samplesPerFrame)).rounded(.up))
    }

    /// The first frame count at which `samples` end in `silenceFrames` silent
    /// frames (RMS under ``DramaSynthesizer/silenceThreshold``), nil if none.
    package static func silenceStopFrame(samples: [Float], silenceFrames: Int) -> Int? {
        let n = ModelConstants.samplesPerFrame
        var run = 0
        for k in 0 ..< samples.count / n {
            run = DramaSynthesizer.isSilent(samples[k * n ..< (k + 1) * n]) ? run + 1 : 0
            if run >= silenceFrames { return k + 1 }
        }
        return nil
    }

    package static func expectedSeconds(for text: String) -> Double {
        let plain = text.replacingOccurrences(of: #"<\|[^|]*\|>|\[[^\]]*\]"#, with: "", options: .regularExpression)
        let chars = plain.split(whereSeparator: { $0.isWhitespace }).joined(separator: " ").count
        return Double(chars) / charactersPerSecond
    }

    /// The most frames a chunk of `line` may run, before the model's capacity
    /// clamps it: the configured cap, plus ``GenerationConfig/capSecondsPerCue``
    /// for each bracket cue when the cap follows the text.
    package static func frameBudget(for line: String, config: GenerationConfig) -> Int {
        let frames = config.frameCap.frames(expectedSeconds: expectedSeconds(for: line))
        guard case .fromText = config.frameCap else { return frames }
        let cues = line.matches(of: #/\[[^\[\]]*\]/#).count
        let seconds = Double(cues) * config.capSecondsPerCue
        let allowance = Int((seconds * Double(ModelConstants.sampleRate) / Double(ModelConstants.samplesPerFrame)).rounded(.up))
        return min(FrameCap.maximumFrames, frames + allowance)
    }

    /// Whether an attempt that ended at `end` after `spokenSeconds` of speech
    /// should be generated again.
    package static func shouldRetry(end: ChunkEnd, spokenSeconds: Double, expectedSeconds: Double, config: GenerationConfig) -> Bool {
        guard config.minSpokenRatio > 0 else { return false }
        let short = spokenSeconds < config.minSpokenRatio * expectedSeconds
        switch end {
        case .cancelled: return false
        case .loopGuard: return true
        case .eos: return short
        case .silence, .frameCap: return short || spokenSeconds < config.retryWithoutEOSBelow * expectedSeconds
        }
    }

    /// Which of two attempts at a chunk to keep: one that reached eos over one
    /// that did not, then the one that spoke longer.
    package static func better(_ a: ChunkResult, than b: ChunkResult) -> Bool {
        if a.finishedByEOS != b.finishedByEOS { return a.finishedByEOS }
        return a.spokenSeconds > b.spokenSeconds
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
        /// Full-window decodes: streamed partials and each chunk's audio.
        let runner = DecodeRunner()
        /// Probe-shape decodes for the silence stop, on their own queue so a
        /// full-window decode never delays a reading.
        let probeRunner = DecodeRunner()
        let streamStep = config.streaming.stepFrames
        let maxRetries = streamStep == nil ? max(0, config.maxRetries) : 0
        /// Streamed chunks whose last frames are still decoding, oldest first.
        var finishing: [(result: ChunkResult, decoder: ChunkDecoder)] = []

        func finish(_ result: ChunkResult) {
            results.append(result)
            onEvent(.chunkFinished(result))
        }
        /// Emits decoded audio in order: the finishing chunks first, then `current`.
        func deliver(current: (index: Int, decoder: ChunkDecoder?)? = nil) throws {
            while let (result, decoder) = finishing.first {
                for piece in decoder.takePieces() { onEvent(.partialAudio(samples: piece.samples, index: result.index, frames: piece.frames)) }
                guard let silence = try decoder.finalSilence() else { return }
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
            let budget = min(Self.frameBudget(for: chunk.line, config: config), model.capacity - prompt.count - 1)
            guard budget >= 1 else { throw PromptError.tooLong(prompt.count) }
            var best: ChunkResult?
            for attempt in 0...maxRetries {
                guard shouldContinue() else { break }
                let t0 = Date()
                let session = try GenerationSession(model: model, prompt: prompt, prefixRows: prefixRows, timeline: timeline)
                let prefillSeconds = Date().timeIntervalSince(t0)
                if attempt == 0 { onEvent(.prefilled(rows: prompt.count)) }
                var frames: [Frame] = []
                var end = ChunkEnd.frameCap
                let t1 = Date()
                let silenceFrames = config.stopAfterSilenceSeconds.map(Self.silenceFrames)
                let decoder = codec.map(ChunkDecoder.init)
                let probe = codec.map { SilenceProbe($0) }
                var offered = 0
                for step in 0..<budget {
                    guard shouldContinue() else { end = .cancelled; break }
                    let r = try session.step(temperature: config.temperature, topK: config.topK, topP: config.topP, sampler: &sampler)
                    guard let frame = r.frame else { end = .eos; break }
                    frames.append(frame)
                    onEvent(.frame(index: step, maxFrames: budget))
                    try deliver(current: (index, decoder))
                    if config.stopOnLoop, frames.count % Self.silenceProbeEveryFrames == 0, MainTokenLoop.cycles(frames.map { $0[0] }) { end = .loopGuard; break }
                    guard let decoder, let probe else { continue }
                    let snapshot = frames
                    if let streamStep, frames.count - offered >= streamStep, runner.offer({ _ = try? decoder.advance(to: snapshot) }) { offered = frames.count }
                    guard let limit = config.stopAfterSilenceSeconds, let silenceFrames, frames.count > silenceFrames else { continue }
                    if frames.count % Self.silenceProbeEveryFrames == 0 { probeRunner.offer { probe.measure(snapshot) } }
                    if probe.latest >= limit { end = .silence; break }
                }
                var result = ChunkResult(index: index, count: chunks.count, text: chunk.turn, frames: frames, promptRows: prompt.count,
                                         end: end, frameBudget: budget, prefillSeconds: prefillSeconds, decodeSeconds: Date().timeIntervalSince(t1),
                                         gpuSeconds: session.gpuSeconds, spokenSeconds: 0, attempts: attempt + 1, samples: [], codecSeconds: 0)
                if streamStep != nil, let decoder, !frames.isEmpty {
                    // No retry can follow, so the GPU moves on while the CPU decodes the last frames.
                    let snapshot = frames
                    runner.enqueue { decoder.measureFinal(snapshot) }
                    finishing.append((result, decoder))
                    best = result
                    break
                }
                probeRunner.drain()
                runner.drain()
                try deliver(current: (index, decoder))
                let trailingSilence = try decoder.flatMap { decoder in frames.isEmpty ? nil : try decoder.advance(to: frames) } ?? 0
                try deliver(current: (index, decoder))
                result.spokenSeconds = max(0, result.durationSeconds - trailingSilence)
                result.samples = decoder?.samples ?? []
                result.codecSeconds = (decoder?.codecSeconds ?? 0) + (probe?.codecSeconds ?? 0)
                if best == nil || Self.better(result, than: best!) { best = result }
                if attempt == maxRetries || !Self.shouldRetry(end: end, spokenSeconds: result.spokenSeconds, expectedSeconds: expected, config: config) { break }
                onEvent(.retry(index: index, attempt: attempt + 1, spokenSeconds: result.spokenSeconds, expectedSeconds: expected))
            }
            guard let result = best else { break }
            if !result.frames.isEmpty { history.push(text: chunk.turn, codes: result.codes) }
            if !finishing.contains(where: { $0.result.index == index }) { finish(result) }
        }
        while let next = finishing.first {
            next.decoder.waitForFinal()
            try deliver()
        }
        probeRunner.drain()
        runner.drain()
        return results
    }
}

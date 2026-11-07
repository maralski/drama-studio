// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation

/// Text in, codec frames out, chunk by chunk, following the reference's
/// long-text loop: each chunk is generated with the pinned first chunk and the
/// latest chunk as history.
package final class FrameGenerator {
    package let model: DramaModel
    package let tokenizer: Tokenizer
    /// Work from all attempts, including replacements that never become audio.
    package private(set) var timings = SpeechTimings()

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
        /// Codec time of the attempt: the chunk's audio and the probe's readings.
        package var codecSeconds: Double
        /// The probe's share of `codecSeconds`.
        package var probeSeconds: Double = 0
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
    /// of each chunk through the probe shape as `silenceProbe` schedules,
    /// `.lowLatency` streams the chunk's audio as it generates, and
    /// `ChunkResult.samples` carries it. Without it, the early stop is off
    /// and the spoken-duration check falls back to the frame count.
    package var codec: ((_ codes: [[Int32]], _ window: DecodeWindow) throws -> [Float])?
    /// When the silence stop reads a chunk's newest frames. The shipped
    /// policy is ``SilenceProbePolicy/shipped``; the `drama` CLI's
    /// `--probe-every` and `--probe-after-expected` set others for measurement.
    package var silenceProbe = SilenceProbePolicy.shipped
    /// How often the loop guard checks the main tokens, independent of the
    /// probe's cadence so the frames a seed gives never change through it.
    package static let loopGuardEveryFrames = 16
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
        let chars = DirectorsNote.spokenWords(text).count
        return Double(chars) / charactersPerSecond
    }

    /// A failed cue turn can remain stuck on the same event/silence pattern
    /// under narrow sampling. Alternate broader replacements with the requested
    /// sampling so retries can also recover precise diction. Keep the cues,
    /// references and accepted history. Word checks
    /// still decide whether the replacement may leave the engine.
    package static func sampling(config: GenerationConfig, text: String, attempt: Int) -> GenerationConfig {
        guard attempt > 0, !attempt.isMultiple(of: 2), config.wordVerification == .required,
              text.range(of: #"\[[^\[\]]*\]|</?[A-Za-z][^>]*>"#, options: .regularExpression) != nil else { return config }
        var replacement = config
        replacement.temperature = max(config.temperature, 1)
        replacement.topP = max(config.topP, 0.9)
        replacement.topK = 4097
        return replacement
    }

    /// The most frames a chunk of `line` may run, before the model's capacity
    /// clamps it: the configured cap, plus ``GenerationConfig/capSecondsPerCue``
    /// for each bracket cue when the cap follows the text.
    package static func frameBudget(for line: String, config: GenerationConfig, expectedSeconds: Double? = nil) -> Int {
        let frames = config.frameCap.frames(expectedSeconds: expectedSeconds ?? Self.expectedSeconds(for: line))
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

    /// Prefer an accepted attempt, then eos, then the longer speech.
    package static func better(_ a: some ChunkAttempt, than b: some ChunkAttempt, expectedSeconds: Double, config: GenerationConfig) -> Bool {
        let retryA = shouldRetry(end: a.end, spokenSeconds: a.spokenSeconds, expectedSeconds: expectedSeconds, config: config)
        let retryB = shouldRetry(end: b.end, spokenSeconds: b.spokenSeconds, expectedSeconds: expectedSeconds, config: config)
        if retryA != retryB { return !retryA }
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
        let plan = try GenerationPlan(text: text, references: references, character: character, previousTakes: previousTakes, note: note, config: config)
        let verifier = try LocalWordVerifier.prepare(plan: plan, config: config, shouldContinue: shouldContinue)
        return try generate(plan: plan, config: config, verifier: verifier, shouldContinue: shouldContinue, onEvent: onEvent)
    }

    package func generate(plan: GenerationPlan, config: GenerationConfig, verifier: LocalWordVerifier?,
                         shouldContinue: () -> Bool = { true }, onEvent: (Event) -> Void) throws -> [ChunkResult] {
        let chunks = plan.chunks
        timings = SpeechTimings()
        if config.wordVerification == .required && verifier == nil {
            throw SpeechVerificationError.unavailable("the spoken-word verifier is missing")
        }
        if verifier != nil, codec == nil {
            throw SpeechVerificationError.unavailable("decoded audio is required for the spoken-word check")
        }
        for chunk in chunks {
            let rows = try Drama3Prompt.build(text: chunk.turn, tokenizer: tokenizer, references: plan.references, character: plan.character).count
            guard rows < model.capacity else { throw PromptError.tooLong(rows) }
        }
        var sampler = Sampler(seed: config.seed ?? UInt64.random(in: 0...UInt64.max))
        var history = Drama3History()
        var results: [ChunkResult] = []
        var chain = VerifiedChunkChain(verifier)
        /// Full-window decodes: streamed partials and each chunk's audio.
        let runner = DecodeRunner()
        defer { runner.drain() }
        let streamStep = plan.streaming.stepFrames
        let maxRetries = streamStep == nil ? max(0, config.maxRetries) : 0
        /// Streamed chunks whose last frames are still decoding, oldest first.
        var finishing: [(result: ChunkResult, decoder: ChunkDecoder)] = []

        func finish(_ result: ChunkResult) throws {
            guard shouldContinue() else { throw CancellationError() }
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
                timings.codecSeconds += finished.codecSeconds
                try finish(finished)
            }
            guard streamStep != nil, let current, let decoder = current.decoder else { return }
            for piece in decoder.takePieces() { onEvent(.partialAudio(samples: piece.samples, index: current.index, frames: piece.frames)) }
        }

        for (index, chunk) in chunks.enumerated() {
            guard shouldContinue() else { throw CancellationError() }
            onEvent(.chunkStarted(index: index, count: chunks.count, text: chunk.line))
            // Reference duration includes pauses and trailing silence. It may
            // extend generation time, but cannot measure how much speech is missing.
            let expected = Self.expectedSeconds(for: chunk.line)
            let allowance = plan.expectedSeconds(for: chunk.line)
            func attempt(_ attempt: Int) throws -> ChunkResult {
                guard shouldContinue() else { throw CancellationError() }
                var frames: [Frame] = []
                var recordedFailure = false
                do {
                    let turn = attempt > 0 && config.wordVerification == .required
                        ? DirectorsNote.literalRetry(line: chunk.line, direction: plan.direction) : chunk.turn
                    let (prompt, prefixRows) = try chunkPrompt(text: turn, plan: plan, history: history)
                    let budget = min(Self.frameBudget(for: chunk.line, config: config, expectedSeconds: allowance), model.capacity - prompt.count - 1)
                    guard budget >= 1 else { throw PromptError.tooLong(prompt.count) }
                    let sampling = Self.sampling(config: config, text: chunk.turn, attempt: attempt)
                    let t0 = Date()
                    let session = try GenerationSession(model: model, prompt: prompt, prefixRows: prefixRows, timeline: timeline)
                    let prefillSeconds = Date().timeIntervalSince(t0)
                    if attempt == 0 { onEvent(.prefilled(rows: prompt.count)) }
                    var end = ChunkEnd.frameCap
                    let t1 = Date()
                    let silenceFrames = config.stopAfterSilenceSeconds.map(Self.silenceFrames)
                    let decoder = codec.map(ChunkDecoder.init)
                    let probe = codec.map { SilenceProbe(silenceFrames: silenceFrames ?? .max, $0) }
                    let readings = silenceProbe.schedule(expectedSeconds: allowance, silenceFrames: silenceFrames ?? 0)
                    var offered = 0
                    for step in 0..<budget {
                        guard shouldContinue() else { end = .cancelled; break }
                        let r = try session.step(temperature: sampling.temperature, topK: sampling.topK, topP: sampling.topP, sampler: &sampler)
                        guard let frame = r.frame else { end = .eos; break }
                        frames.append(frame)
                        onEvent(.frame(index: step, maxFrames: budget))
                        try deliver(current: (index, decoder))
                        if config.stopOnLoop, frames.count % Self.loopGuardEveryFrames == 0, MainTokenLoop.cycles(frames.map { $0[0] }) { end = .loopGuard; break }
                        guard let decoder, let probe else { continue }
                        let snapshot = frames
                        if let streamStep, frames.count - offered >= streamStep, runner.offer({ _ = try? decoder.advance(to: snapshot) }) { offered = frames.count }
                        guard let limit = config.stopAfterSilenceSeconds else { continue }
                        if readings.reads(at: frames.count) {
                            // Resolve each fixed checkpoint before sampling again, so
                            // codec speed cannot change a seeded attempt's stop frame.
                            probe.measure(snapshot)
                            if probe.latest >= limit { end = .silence; break }
                        }
                    }
                    var result = ChunkResult(index: index, count: chunks.count, text: turn, frames: frames, promptRows: prompt.count,
                                             end: end, frameBudget: budget, prefillSeconds: prefillSeconds, decodeSeconds: Date().timeIntervalSince(t1),
                                             gpuSeconds: session.gpuSeconds, spokenSeconds: 0, attempts: attempt + 1, samples: [], codecSeconds: 0)
                    if streamStep != nil, let decoder, !frames.isEmpty {
                        // No retry can follow, so the GPU moves on while the CPU decodes the last frames.
                        let snapshot = frames
                        runner.enqueue { decoder.measureFinal(snapshot) }
                        finishing.append((result, decoder))
                        timings.include(result)
                        return result
                    }
                    runner.drain()
                    try deliver(current: (index, decoder))
                    let trailingSilence: Double
                    do {
                        trailingSilence = try decoder.flatMap { decoder in frames.isEmpty ? nil : try decoder.advance(to: frames) } ?? 0
                    } catch {
                        SpeechAttemptAudit.record(samples: decoder?.samples ?? [], line: chunk.line, reason: "codec_error", frames: frames,
                                                  error: error.localizedDescription)
                        recordedFailure = true
                        throw error
                    }
                    try deliver(current: (index, decoder))
                    result.spokenSeconds = max(0, result.durationSeconds - trailingSilence)
                    result.samples = decoder?.samples ?? []
                    result.probeSeconds = probe?.codecSeconds ?? 0
                    result.codecSeconds = (decoder?.codecSeconds ?? 0) + result.probeSeconds
                    timings.include(result)
                    if result.end == .loopGuard || result.end == .cancelled {
                        SpeechAttemptAudit.record(samples: result.samples, line: chunk.line, reason: String(describing: result.end), frames: frames)
                    }
                    return result
                } catch {
                    if !recordedFailure {
                        SpeechAttemptAudit.record(samples: [], line: chunk.line, reason: "generation_error", frames: frames, error: error.localizedDescription)
                    }
                    throw error
                }
            }
            var result = try ChunkAttempts.run(index: index, maxRetries: maxRetries,
                                               expectedSeconds: expected, config: config,
                                               shouldContinue: shouldContinue, attempt: attempt,
                                               verify: { result in
                                                   let start = Date()
                                                   defer { timings.verificationSeconds += Date().timeIntervalSince(start) }
                                                   return try chain.accept(result.samples, at: index, shouldContinue: shouldContinue)
                                               }) { attempt, spoken in
                onEvent(.retry(index: index, attempt: attempt, spokenSeconds: spoken, expectedSeconds: expected))
            }
            if let verified = try chain.commit() { result.samples = verified }
            if !finishing.contains(where: { $0.result.index == index }) { try finish(result) }
            if !result.frames.isEmpty { history.push(text: result.text, codes: result.codes) }
        }
        while let next = finishing.first {
            next.decoder.waitForFinal()
            try deliver()
        }
        runner.drain()
        return results
    }
}

extension SpeechTimings {
    package mutating func include(_ attempt: FrameGenerator.ChunkResult) {
        prefillSeconds += attempt.prefillSeconds
        decodeSeconds += attempt.decodeSeconds
        codecSeconds += attempt.codecSeconds
        promptRows += attempt.promptRows
        framesGenerated += attempt.frames.count
    }
}

/// What the acceptance loop judges an attempt by, from either engine.
package protocol ChunkAttempt: Sendable {
    var end: FrameGenerator.ChunkEnd { get }
    var spokenSeconds: Double { get }
}

extension ChunkAttempt {
    package var finishedByEOS: Bool { end == .eos }
}

extension FrameGenerator.ChunkResult: ChunkAttempt {}

/// The production acceptance boundary, independent of model sampling so its
/// retry, cancellation and exhaustion behavior can be exercised deterministically.
package enum ChunkAttempts {
    package static func run<Attempt: ChunkAttempt>(index: Int, maxRetries: Int, expectedSeconds: Double, config: GenerationConfig,
                                                   shouldContinue: () -> Bool,
                                                   attempt: (Int) throws -> Attempt,
                                                   verify: (Attempt) throws -> Bool,
                                                   retry: (Int, Double) -> Void) throws -> Attempt {
        var best: Attempt?
        let attempts = max(0, maxRetries) + 1
        for number in 0..<attempts {
            guard shouldContinue() else { throw CancellationError() }
            let result = try attempt(number)
            guard shouldContinue(), result.end != .cancelled else { throw CancellationError() }
            if config.wordVerification == .required {
                // Cue audio can satisfy the duration heuristic. Check actual
                // words even after EOS or a plausible amount of sound.
                if result.end != .loopGuard, try verify(result) {
                    guard shouldContinue() else { throw CancellationError() }
                    return result
                }
            } else {
                if best == nil || FrameGenerator.better(result, than: best!, expectedSeconds: expectedSeconds, config: config) { best = result }
                if !FrameGenerator.shouldRetry(end: result.end, spokenSeconds: result.spokenSeconds,
                                              expectedSeconds: expectedSeconds, config: config) { return result }
            }
            if number + 1 < attempts { retry(number + 1, result.spokenSeconds) }
        }
        // Only explicitly unchecked previews may retain a rejected attempt.
        if config.wordVerification == .uncheckedPreview, let best { return best }
        throw SpeechVerificationError.exhausted(sentence: index + 1, attempts: attempts)
    }
}

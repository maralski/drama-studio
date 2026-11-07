// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation
import Synchronization
internal import MingTTS

/// Where the Ming-3.0-tts-0.5B checkpoint lives: `model.safetensors`,
/// `config.json`, `tokenizer.json` and, for cloning, `campplus.onnx`.
public struct MingLocation: Sendable, Equatable {
    public var directory: URL
    public init(directory: URL) { self.directory = directory }

    /// `MING_WEIGHTS`, when it names a complete checkpoint.
    public static func fromEnvironment() -> MingLocation? {
        guard let path = ProcessInfo.processInfo.environment["MING_WEIGHTS"], !path.isEmpty else { return nil }
        let location = MingLocation(directory: URL(filePath: path, directoryHint: .isDirectory))
        return location.isComplete ? location : nil
    }

    public static let requiredFiles = ["model.safetensors", "config.json", "tokenizer.json", "campplus.onnx"]

    public var isComplete: Bool {
        Self.requiredFiles.allSatisfy { FileManager.default.fileExists(atPath: directory.appendingPathComponent($0).path) }
    }
}

/// Ming-3.0-tts-0.5B behind the same protocol as Drama 3. Requests run one
/// at a time on this actor. Loading maps 2.8 GB of weights; an M3 Max
/// generates about 60 steps of 80 ms a second, five times real time.
///
/// Each sentence is one Ming request, verified and retried like a Drama 3
/// chunk. A cloned voice conditions every sentence; a designed or built-in
/// voice speaks the first sentence, whose audio is then encoded and cloned
/// for the rest, as Ming's own style tasks do. The finished take's opening
/// sentences are encoded the same way so a later take can continue the voice.
public actor MingSynthesizer: SpeechSynthesizer {
    public static let modelID = "inclusionAI/Ming-3.0-tts-0.5b"
    /// The longest take excerpt a later take clones. Longer references leave
    /// Ming's 512-position cache too little room to generate.
    package static let excerptSeconds = 10.0
    public nonisolated let location: MingLocation
    private var model: MingTTSModel?
    private var loading: Task<MingTTSModel, Error>?
    private nonisolated let loaded = Mutex(false)

    public init(location: MingLocation) { self.location = location }

    public nonisolated var engineDescription: EngineDescription {
        EngineDescription(engine: .ming, modelID: Self.modelID, snapshot: location.directory.lastPathComponent, isMock: false)
    }

    /// Ming has no kernel tuning: false once loaded.
    public nonisolated var isTuned: Bool? { loaded.withLock { $0 } ? false : nil }
    public nonisolated var isLoaded: Bool { loaded.withLock { $0 } }

    public func load(onPhase: @escaping @Sendable (String) -> Void) async throws {
        _ = try await loadedModel(onPhase: onPhase)
    }

    private func loadedModel(onPhase: @escaping @Sendable (String) -> Void = { _ in }) async throws -> MingTTSModel {
        if let model { return model }
        if let loading { return try await loading.value }
        onPhase("Loading Ming")
        let task = Task.detached(priority: .userInitiated) { [location] in try MingTTSModel(directory: location.directory) }
        loading = task
        defer { loading = nil }
        let model = try await task.value
        self.model = model
        loaded.withLock { $0 = true }
        return model
    }

    /// Encodes a clip for cloning, loading the engine if needed. About 260 ms
    /// for a 12 s clip on an M3 Max. Trailing silence is cut first, as it is
    /// from a take's excerpt: Ming continues a reference from its last moment,
    /// and a clone of a clip that ends in silence stops early and clips its
    /// final word.
    public func encodeVoice(samples: [Float], transcript: String) async throws -> VoiceEncoding {
        let model = try await loadedModel()
        return .ming(reference: try Self.encode(Self.trimmed(samples), transcript: transcript, with: model))
    }

    static func encode(_ samples: [Float], transcript: String, with model: MingTTSModel) throws -> MingVoiceEncoding {
        let reference = try model.encodeReference(samples: samples, sampleRate: ModelConstants.sampleRate, transcript: transcript)
        return try MingVoiceEncoding(transcript: reference.transcript, sampleCount: samples.count,
                                     latents: reference.latents, speakerEmbedding: reference.speakerEmbedding)
    }

    public nonisolated func synthesize(_ request: SpeechRequest) -> AsyncThrowingStream<SpeechEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    try await self.run(request, continuation: continuation)
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    struct Attempt: ChunkAttempt {
        var samples: [Float]
        var end: FrameGenerator.ChunkEnd
        var spokenSeconds: Double
        var steps: Int
        var prefillSeconds: Double
        var decodeSeconds: Double
    }

    private func run(_ request: SpeechRequest, continuation: AsyncThrowingStream<SpeechEvent, Error>.Continuation) async throws {
        let plan = try MingPlan(request)
        let config = request.config
        let verifier = try LocalWordVerifier.prepare(lines: plan.chunks.map(\.spoken.text), config: config, shouldContinue: { !Task.isCancelled })
        let language = try Self.language(plan.language)
        continuation.yield(.loadingModel(progress: 0, detail: "Loading Ming"))
        let model = try await loadedModel { detail in continuation.yield(.loadingModel(progress: 0, detail: detail)) }
        continuation.yield(.loadingModel(progress: 1, detail: "Ready"))
        let shouldContinue = { !Task.isCancelled }

        var timings = SpeechTimings()
        var chain = VerifiedChunkChain(verifier)
        var sentences: [SpokenSentence] = []
        var samples: [Float] = []
        var chunkSamples: [[Float]] = []
        var finishedByEOS = true
        var conditioning = MingConditioning.zeroSpeaker
        var cloned = plan.voice.references
        let baseSeed = config.seed ?? UInt64.random(in: 0...UInt64.max)

        for (index, chunk) in plan.chunks.enumerated() {
            guard shouldContinue() else { throw CancellationError() }
            continuation.yield(.chunkStarted(index: index, count: plan.chunks.count, text: chunk.line))
            let text = chunk.spoken.text
            guard !UnicodeWhitespace.strip(text).isEmpty else {
                sentences.append(SpokenSentence(text: chunk.line, samples: 0))
                chunkSamples.append([])
                continue
            }
            let task: MingTask
            if let reference = plan.reference(for: chunk, among: cloned) {
                conditioning = .clone(try reference.mingReference())
                task = chunk.spoken.style.map { .emotion($0.emotion.ming) } ?? .plain
            } else if case .design(let description) = plan.voice {
                task = .voiceDesign(description)
            } else {
                task = .builtInVoice(.emma)
            }
            let mingRequest = try MingRequest(text: text, task: task, language: language, conditioning: conditioning)
            let expected = FrameGenerator.expectedSeconds(for: chunk.line)
            let maxSteps = Self.stepCap(for: chunk.line, config: config, expectedSeconds: plan.expectedSeconds[index])
            var promptReported = false
            func attempt(_ number: Int) throws -> Attempt {
                guard shouldContinue() else { throw CancellationError() }
                var options = SynthesisOptions()
                options.cfg = Self.guidance(attempt: number)
                options.seed = baseSeed &+ UInt64(number)
                options.maxSteps = maxSteps
                var pcm: [Float] = []
                var steps = 0
                let start = Date()
                let reports = try model.synthesize(mingRequest, options: options) { _, chunkPCM in
                    try Task.checkCancellation()
                    if !promptReported {
                        promptReported = true
                        continuation.yield(.prefill(rows: 0))
                    }
                    pcm += chunkPCM
                    steps = pcm.count / MingVoiceEncoding.samplesPerLatent
                    continuation.yield(.frame(index: max(0, steps - 1), maxFrames: maxSteps))
                }
                let wall = Date().timeIntervalSince(start)
                let prefill = reports.reduce(0) { $0 + $1.prefillSeconds }
                let generated = reports.reduce(0) { $0 + $1.steps }
                let end: FrameGenerator.ChunkEnd = reports.allSatisfy(\.stopped) ? .eos : .frameCap
                let spoken = max(0, Double(pcm.count) / Double(ModelConstants.sampleRate) - DramaSynthesizer.trailingSilenceSeconds(pcm))
                return Attempt(samples: pcm, end: end, spokenSeconds: spoken, steps: generated, prefillSeconds: prefill, decodeSeconds: max(0, wall - prefill))
            }
            var result = try ChunkAttempts.run(index: index, maxRetries: max(0, config.maxRetries), expectedSeconds: expected, config: config,
                                               shouldContinue: shouldContinue, attempt: { number in
                                                   let attempt = try attempt(number)
                                                   timings.prefillSeconds += attempt.prefillSeconds
                                                   timings.decodeSeconds += attempt.decodeSeconds
                                                   timings.framesGenerated += attempt.steps
                                                   return attempt
                                               }, verify: { attempt in
                                                   let start = Date()
                                                   defer { timings.verificationSeconds += Date().timeIntervalSince(start) }
                                                   return try chain.accept(attempt.samples, at: index, shouldContinue: shouldContinue)
                                               }) { number, spoken in
                continuation.yield(.retry(chunkIndex: index, attempt: number))
            }
            if let verified = try chain.commit() { result.samples = verified }
            finishedByEOS = finishedByEOS && result.end == .eos
            samples += result.samples
            chunkSamples.append(result.samples)
            sentences.append(SpokenSentence(text: chunk.line, samples: result.samples.count))
            if !result.samples.isEmpty { continuation.yield(.audio(samples: result.samples, chunkIndex: index)) }
            if cloned.isEmpty, !result.samples.isEmpty {
                cloned = [try Self.encode(Self.trimmed(result.samples), transcript: text, with: model)]
            }
        }
        try Task.checkCancellation()
        guard !samples.isEmpty else { throw PromptError.emptyText("spoken words") }
        let excerpt = Self.excerpt(chunks: plan.chunks.map(\.spoken.text), samples: chunkSamples)
        let encodeStart = Date()
        let reference = try Self.encode(Self.trimmed(excerpt.samples), transcript: excerpt.transcript, with: model)
        timings.codecSeconds += Date().timeIntervalSince(encodeStart)
        let take = Take(text: plan.turnText, context: .ming(reference: reference, sampleCount: samples.count))
        continuation.yield(.finished(SpeechResult(take: take, samples: samples, timings: timings, finishedByEOS: finishedByEOS, sentences: sentences,
                                                wordCoverage: config.wordVerification == .required ? .recognized : .unchecked, notices: plan.notices)))
    }

    /// The flow head's classifier-free guidance scale per attempt. Ming's
    /// reference uses 2.0. After a `[Laughter]` tag the model can keep
    /// laughing to the step cap instead of returning to the words; on the
    /// harness's cue lines 3.0 made that worse (21 of 36 first passes against
    /// 24), 1.5 best (29) and 1.0 close (27, no line exhausted). Which scale
    /// a voice derails at is the voice's own, and a new seed alone rarely
    /// rescues it, so the first attempt runs at 1.5 and retries cycle through
    /// the other two.
    package static let guidanceLadder: [Float] = [1.5, 1.0, 2.0]

    package static func guidance(attempt: Int) -> Float {
        guidanceLadder[attempt % guidanceLadder.count]
    }

    /// The most steps one sentence may run: the configured frame cap in
    /// seconds, plus ``GenerationConfig/capSecondsPerCue`` per cue as Drama 3
    /// allows, in 80 ms steps.
    package static func stepCap(for line: String, config: GenerationConfig, expectedSeconds: Double) -> Int {
        let frames = FrameGenerator.frameBudget(for: line, config: config, expectedSeconds: expectedSeconds)
        let seconds = Double(frames) * Double(ModelConstants.samplesPerFrame) / Double(ModelConstants.sampleRate)
        return max(1, Int((seconds * SpeechEngineID.ming.capabilities.stepsPerSecondOfAudio).rounded(.up)))
    }

    /// The opening sentences of a take, whole, up to ``excerptSeconds``; at
    /// least the first sentence that has audio.
    package static func excerpt(chunks: [String], samples: [[Float]]) -> (transcript: String, samples: [Float]) {
        var transcript: [String] = []
        var audio: [Float] = []
        let limit = Int(excerptSeconds * Double(ModelConstants.sampleRate))
        for (text, pcm) in zip(chunks, samples) where !pcm.isEmpty {
            if !audio.isEmpty, audio.count + pcm.count > limit { break }
            transcript.append(text)
            audio += pcm
        }
        return (transcript.joined(separator: " "), audio)
    }

    static func trimmed(_ samples: [Float]) -> [Float] {
        let silence = Int(DramaSynthesizer.trailingSilenceSeconds(samples) * Double(ModelConstants.sampleRate))
        return silence > 0 ? Array(samples.dropLast(silence)) : samples
    }

    static func language(_ identified: String?) throws -> MingLanguage {
        guard let identified else { return .english }
        guard let language = MingLanguage(id: identified) else { throw PromptError.engineUnsupported(.ming, "speak \(identified) text") }
        return language
    }
}

extension MingVoiceEncoding {
    func mingReference() throws -> MingReferenceVoice {
        try MingReferenceVoice(transcript: transcript, latents: latents, speakerEmbedding: speakerEmbedding)
    }
}

extension MingCues.Emotion {
    var ming: MingEmotion {
        switch self {
        case .happy: .happy
        case .sad: .sad
        case .angry: .angry
        case .fearful: .fearful
        case .surprised: .surprised
        }
    }
}

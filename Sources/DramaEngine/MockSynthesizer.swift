// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation

/// A stand-in engine for previews, tests, and working on an app without the
/// weights. It streams the same event sequence as ``DramaSynthesizer``, cut
/// into the same chunks, with a sine sweep per chunk as the audio. It needs
/// no model files, no Metal, and almost no memory.
public actor MockSynthesizer: SpeechSynthesizer {
    /// How fast the mock pretends to work.
    public struct Pace: Sendable {
        /// Seconds the pretend load takes.
        public var loadSeconds: Double
        /// Frames generated per second; 0 generates as fast as possible.
        public var framesPerSecond: Double
        /// Creates a pace.
        public init(loadSeconds: Double, framesPerSecond: Double) {
            self.loadSeconds = loadSeconds
            self.framesPerSecond = framesPerSecond
        }
        /// A 2 s load and 60 frames per second, close to the real engine's feel.
        public static let realistic = Pace(loadSeconds: 2, framesPerSecond: 60)
        /// No waiting, for tests.
        public static let instant = Pace(loadSeconds: 0, framesPerSecond: 0)
    }

    /// Why the mock refused a request.
    public enum MockError: Error, CustomStringConvertible {
        /// The request's text is empty or white space.
        case emptyText
        /// A message for a person.
        public var description: String { "There is no text to speak." }
    }

    private let pace: Pace
    /// The engine the mock stands in for; its encodings and takes carry that engine's context.
    public nonisolated let engine: SpeechEngineID
    private var loaded = false

    /// Creates a mock that works at `pace` and stands in for `engine`.
    public init(engine: SpeechEngineID = .drama3, pace: Pace = .realistic) {
        self.engine = engine
        self.pace = pace
    }

    /// Whether the pretend load has finished.
    public var isLoaded: Bool { loaded }

    /// Waits ``Pace/loadSeconds``, reporting ten pretend phases.
    public func load(onPhase: @escaping @Sendable (String) -> Void) async throws {
        try await load { _, detail in onPhase(detail) }
    }

    /// Stands in for the engine without its weights.
    public nonisolated var engineDescription: EngineDescription {
        switch engine {
        case .drama3: EngineDescription(engine: .drama3, modelID: "fishaudio/drama-3", snapshot: "mock", isMock: true)
        case .ming: EngineDescription(engine: .ming, modelID: MingSynthesizer.modelID, snapshot: "mock", isMock: true)
        }
    }

    public nonisolated var isTuned: Bool? { false }

    /// A made-up encoding as long as the clip, in the mock's engine.
    public func encodeVoice(samples: [Float], transcript: String) async throws -> VoiceEncoding {
        switch engine {
        case .drama3: .drama3(codes: Self.codes(frameCount: max(1, Int((Double(samples.count) / Double(ModelConstants.samplesPerFrame)).rounded()))))
        case .ming: .ming(reference: try Self.mingReference(transcript: transcript, sampleCount: max(1, samples.count)))
        }
    }

    package static func codes(frameCount: Int) -> [[Int32]] {
        (0..<ModelConstants.numCodebooks).map { book in
            let size = book == 0 ? ModelConstants.codebookSize : ModelConstants.residualCodebookSize
            return (0..<frameCount).map { Int32(($0 * 31 + book * 17) % size) }
        }
    }

    /// Made-up latents, one row per 3528 samples, and a unit speaker embedding.
    static func mingReference(transcript: String, sampleCount: Int) throws -> MingVoiceEncoding {
        let rows = max(1, Int((Double(sampleCount) / Double(MingVoiceEncoding.samplesPerLatent)).rounded(.up)))
        let latents = (0..<rows * MingVoiceEncoding.latentDim).map { Float($0 % 97) / 97 - 0.5 }
        let embedding = (0..<MingVoiceEncoding.speakerEmbeddingDim).map { _ in Float(1) / Float(MingVoiceEncoding.speakerEmbeddingDim).squareRoot() }
        return try MingVoiceEncoding(transcript: transcript, sampleCount: sampleCount, latents: latents, speakerEmbedding: embedding)
    }

    /// Streams the real engine's event sequence for `request`, with a sweep
    /// per chunk. Partials follow ``GenerationConfig/streaming`` as in the
    /// real engine. Throws ``MockError/emptyText`` for empty text.
    public nonisolated func synthesize(_ request: SpeechRequest) -> AsyncThrowingStream<SpeechEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    try await self.run(request) { continuation.yield($0) }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    private func load(report: @Sendable (Double, String) -> Void) async throws {
        guard !loaded else { return }
        let steps = 10
        for step in 1...steps {
            try await wait(pace.loadSeconds / Double(steps))
            let detail = step < steps ? "Mapping weights, \(step * 4) of 40 layers" : "Compiling kernels"
            report(Double(step) / Double(steps), detail)
        }
        loaded = true
    }

    private func run(_ request: SpeechRequest, emit: @Sendable (SpeechEvent) -> Void) async throws {
        try await load { emit(.loadingModel(progress: $0, detail: $1)) }
        guard !UnicodeWhitespace.strip(request.text).isEmpty else { throw MockError.emptyText }
        let outline = try SpeechOutline(request, engine: engine)
        let lines = outline.lines
        let streaming = request.config.effectiveStreaming

        var rng = SplitMix64(seed: request.config.seed ?? UInt64.random(in: 0...UInt64.max))
        var frames: [Frame] = []
        var samples: [Float] = []
        var timings = SpeechTimings()
        var sentences: [SpokenSentence] = []
        let framesPerSecondOfAudio = Double(ModelConstants.sampleRate) / Double(ModelConstants.samplesPerFrame)

        for (index, text) in lines.enumerated() {
            emit(.chunkStarted(index: index, count: lines.count, text: text))
            let promptRows = 24 + text.utf8.count / 3 + request.references.reduce(0) { $0 + Int($1.durationSeconds * framesPerSecondOfAudio) }
            timings.promptRows += promptRows
            emit(.prefill(rows: promptRows))

            let spokenSeconds = Double(text.count) / 15
            let budget = request.config.frameCap.frames(expectedSeconds: FrameGenerator.expectedSeconds(for: text))
            let target = min(budget, max(8, Int(spokenSeconds * framesPerSecondOfAudio)))
            let audio = Self.sweep(frameCount: target, pitch: 1 + Double(index % 4) * 0.25)
            var streamed = 0
            func stream(through frame: Int) {
                guard frame > streamed else { return }
                let n = ModelConstants.samplesPerFrame
                emit(.partialAudio(samples: Array(audio[streamed * n ..< frame * n]), chunkIndex: index, frameRange: streamed ..< frame))
                streamed = frame
            }
            let start = ContinuousClock.now
            for frameIndex in 0..<target {
                try await wait(pace.framesPerSecond > 0 ? 1 / pace.framesPerSecond : 0)
                let c0 = Int32(rng.next() % UInt64(ModelConstants.codebookSize))
                let residual = (1..<ModelConstants.numCodebooks).map { _ in Int32(rng.next() % UInt64(ModelConstants.residualCodebookSize)) }
                frames.append([c0 + ModelConstants.semanticStart, c0] + residual)
                emit(.frame(index: frameIndex, maxFrames: budget))
                if let step = streaming.stepFrames, (frameIndex + 1) % step == 0 { stream(through: frameIndex + 1) }
            }
            if streaming.stepFrames != nil { stream(through: target) }
            timings.decodeSeconds += (ContinuousClock.now - start).seconds
            timings.framesGenerated += target

            samples += audio
            sentences.append(SpokenSentence(text: text, samples: audio.count))
            emit(.audio(samples: audio, chunkIndex: index))
        }
        let context: TakeContext = switch engine {
        case .drama3: .drama3(codes: (0..<ModelConstants.numCodebooks).map { c in frames.map { $0[1 + c] } })
        case .ming: .ming(reference: try Self.mingReference(transcript: DirectorsNote.spokenWords(request.text), sampleCount: samples.count), sampleCount: samples.count)
        }
        let take = Take(text: GenerationPlan.turnText(text: request.text, note: request.note, character: request.character), context: context)
        emit(.finished(SpeechResult(take: take, samples: samples, timings: timings, finishedByEOS: true, sentences: sentences, notices: outline.notices)))
    }

    private func wait(_ seconds: Double) async throws {
        if seconds > 0 {
            try await Task.sleep(for: .seconds(seconds))
        } else {
            try Task.checkCancellation()
        }
    }

    static func sweep(frameCount: Int, pitch: Double) -> [Float] {
        let count = frameCount * ModelConstants.samplesPerFrame
        let rate = Double(ModelConstants.sampleRate)
        let fade = Int(rate * 0.01)
        var phase = 0.0
        return (0..<count).map { n in
            let t = Double(n) / Double(max(count, 1))
            phase += 2 * .pi * (220 + 220 * t) * pitch / rate
            let envelope = min(1, Double(min(n, count - 1 - n)) / Double(fade))
            return Float(0.2 * envelope * sin(phase))
        }
    }
}

private struct SplitMix64 {
    var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}

private extension Duration {
    var seconds: Double { Double(components.seconds) + Double(components.attoseconds) / 1e18 }
}

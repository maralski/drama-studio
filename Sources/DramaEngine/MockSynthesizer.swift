import AVFoundation
import Foundation

/// A stand-in engine for previews, tests, and the app until the Metal engine
/// lands. It streams the same event sequence as the real one, with a sine
/// sweep per chunk as the audio.
public actor MockSynthesizer: SpeechSynthesizer {
    public struct Pace: Sendable {
        public var loadSeconds: Double
        public var framesPerSecond: Double
        public init(loadSeconds: Double, framesPerSecond: Double) {
            self.loadSeconds = loadSeconds
            self.framesPerSecond = framesPerSecond
        }
        public static let realistic = Pace(loadSeconds: 2, framesPerSecond: 60)
        public static let instant = Pace(loadSeconds: 0, framesPerSecond: 0)
    }

    public enum MockError: Error, CustomStringConvertible {
        case emptyText
        public var description: String { "There is no text to speak." }
    }

    private let pace: Pace
    private var loaded = false

    public init(pace: Pace = .realistic) { self.pace = pace }

    public var isLoaded: Bool { loaded }

    public func load(onPhase: @escaping @Sendable (String) -> Void) async throws {
        try await load { _, detail in onPhase(detail) }
    }

    public func encodeReference(audioURL: URL, transcript: String, name: String) async throws -> ReferenceVoice {
        let file = try AVAudioFile(forReading: audioURL)
        let seconds = Double(file.length) / file.processingFormat.sampleRate
        let frameCount = max(1, Int((seconds * Double(ModelConstants.sampleRate) / Double(ModelConstants.samplesPerFrame)).rounded()))
        let codes = (0..<ModelConstants.numCodebooks).map { book in
            let size = book == 0 ? ModelConstants.codebookSize : ModelConstants.residualCodebookSize
            return (0..<frameCount).map { Int32(($0 * 31 + book * 17) % size) }
        }
        return ReferenceVoice(name: name, transcript: transcript, codes: codes, sourceFileName: audioURL.lastPathComponent)
    }

    public func decode(frames: [Frame]) async throws -> [Float] {
        Self.sweep(frameCount: frames.count, pitch: 1)
    }

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
        let plan = try GenerationPlan(request)
        let lines = plan.chunks.map(\.line)

        var rng = SplitMix64(seed: request.config.seed ?? UInt64.random(in: 0...UInt64.max))
        var frames: [Frame] = []
        var samples: [Float] = []
        var timings = SpeechTimings()
        var sentences: [SpokenSentence] = []
        let framesPerSecondOfAudio = Double(ModelConstants.sampleRate) / Double(ModelConstants.samplesPerFrame)

        for (index, text) in lines.enumerated() {
            emit(.chunkStarted(index: index, count: lines.count, text: text))
            let promptRows = 24 + text.utf8.count / 3 + request.references.reduce(0) { $0 + ($1.codes.first?.count ?? 0) }
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
                if let step = request.config.streaming.stepFrames, (frameIndex + 1) % step == 0 { stream(through: frameIndex + 1) }
            }
            if request.config.streaming.stepFrames != nil { stream(through: target) }
            timings.decodeSeconds += (ContinuousClock.now - start).seconds
            timings.framesGenerated += target

            samples += audio
            sentences.append(SpokenSentence(text: text, frames: target))
            emit(.audio(samples: audio, chunkIndex: index))
        }
        let take = Take(text: plan.turnText, codes: (0..<ModelConstants.numCodebooks).map { c in frames.map { $0[1 + c] } })
        emit(.finished(SpeechResult(take: take, samples: samples, timings: timings, finishedByEOS: true, sentences: sentences)))
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

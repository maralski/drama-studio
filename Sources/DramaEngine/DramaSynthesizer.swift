import Foundation
import Synchronization

/// The real engine behind the app: Metal backbone and tower, Core ML codec,
/// Swift tokenizer. Requests run one at a time on a serial queue.
public actor DramaSynthesizer: SpeechSynthesizer {
    public nonisolated let location: ModelLocation
    public let codecDirectory: URL
    public let mode: ModelPrecisionMode
    /// Where the kernel tuning cache lives; tests point it at a temp directory.
    public let tuningDirectory: URL
    /// Where a quantized mode's weight cache lives.
    public let quantizedDirectory: URL
    private var model: DramaModel?
    private var tokenizer: Tokenizer?
    private var codec: DramaCodec?
    private var loading: Task<Void, Error>?

    public init(location: ModelLocation, codecDirectory: URL, mode: ModelPrecisionMode = .full,
                tuningDirectory: URL = TuningCache.defaultDirectory, quantizedDirectory: URL = QuantizedWeightFile.defaultDirectory) {
        self.location = location
        self.codecDirectory = codecDirectory
        self.mode = mode
        self.tuningDirectory = tuningDirectory
        self.quantizedDirectory = quantizedDirectory
    }

    /// Generation runs synchronously on this actor, so status reads go
    /// through a lock instead of waiting for the run to end. Nil until the
    /// model loads, then whether its schedule came from the tuning cache.
    private nonisolated let tuned = Mutex<Bool?>(nil)

    public nonisolated var isLoaded: Bool { tuned.withLock { $0 != nil } }
    public nonisolated var isTuned: Bool? { tuned.withLock { $0 } }

    /// The first load of a quantized mode builds its weight cache, and the
    /// first load on a Mac also tunes the kernel schedule and caches it, so
    /// later loads find both and skip the steps. A tuning failure leaves the
    /// default schedule in place; it never fails the load.
    public func load(onPhase: @escaping @Sendable (String) -> Void) async throws {
        if isLoaded { return }
        if let loading { return try await loading.value }
        let task = Task { [location, codecDirectory, mode, tuningDirectory, quantizedDirectory] in
            let codec = try await DramaCodec(directory: codecDirectory)
            let tokenizer = try Tokenizer(url: location.directory.appendingPathComponent("tokenizer.json"))
            let cacheURL = QuantizedWeightFile.url(snapshot: location.directory.lastPathComponent, mode: mode, directory: quantizedDirectory)
            if mode != .full, !FileManager.default.fileExists(atPath: cacheURL.path) {
                onPhase("Quantizing weights for the \(mode.rawValue) mode (first load only)")
            }
            let model = try DramaModel(location: location, mode: mode, tuningDirectory: tuningDirectory, quantizedDirectory: quantizedDirectory)
            model.warmUp()
            if model.scheduleSource == .default {
                onPhase("Tuning kernels for this Mac, about a minute")
                do {
                    try Autotune.tune(model: model, cacheDirectory: tuningDirectory)
                } catch {
                    FileHandle.standardError.write("kernel autotune failed, keeping the default schedule: \(error)\n".data(using: .utf8)!)
                }
            }
            self.install(model: model, tokenizer: tokenizer, codec: codec)
        }
        loading = task
        defer { loading = nil }
        try await task.value
    }

    private func install(model: DramaModel, tokenizer: Tokenizer, codec: DramaCodec) {
        self.model = model
        self.tokenizer = tokenizer
        self.codec = codec
        let fromCache = if case .cache = model.scheduleSource { true } else { false }
        tuned.withLock { $0 = fromCache }
    }

    public func encodeReference(audioURL: URL, transcript: String, name: String) async throws -> ReferenceVoice {
        try await load()
        let samples = try DramaCodec.loadAudio(url: audioURL)
        let codes = try await codec!.encode(samples: samples)
        return ReferenceVoice(name: name, transcript: transcript, codes: codes, sourceFileName: audioURL.lastPathComponent)
    }

    public func decode(frames: [Frame]) async throws -> [Float] {
        try await load()
        let codes = (0..<ModelConstants.numCodebooks).map { c in frames.map { $0[1 + c] } }
        return try await codec!.decode(codes: codes)
    }

    /// Seconds of trailing audio whose 2048-sample frames all stay under an RMS of 0.01.
    public nonisolated static func trailingSilenceSeconds(_ samples: [Float]) -> Double {
        let n = ModelConstants.samplesPerFrame
        var silentFrames = 0
        var end = samples.count / n * n
        while end >= n {
            var energy: Float = 0
            for i in (end - n)..<end { energy += samples[i] * samples[i] }
            if (energy / Float(n)).squareRoot() >= 0.01 { break }
            silentFrames += 1
            end -= n
        }
        return Double(silentFrames * n) / Double(ModelConstants.sampleRate)
    }

    /// Runs a codec decode from synchronous generator code.
    public nonisolated static func decodeBlocking(_ codec: DramaCodec, codes: [[Int32]]) throws -> [Float] {
        let group = DispatchGroup()
        group.enter()
        var result: Result<[Float], Error> = .failure(EngineError.badModel("codec did not answer"))
        Task.detached {
            do { result = .success(try await codec.decode(codes: codes)) } catch { result = .failure(error) }
            group.leave()
        }
        group.wait()
        return try result.get()
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

    private func run(_ request: SpeechRequest, continuation: AsyncThrowingStream<SpeechEvent, Error>.Continuation) async throws {
        continuation.yield(.loadingModel(progress: 0, detail: "Loading Drama 3"))
        try await load { detail in continuation.yield(.loadingModel(progress: 0, detail: detail)) }
        continuation.yield(.loadingModel(progress: 1, detail: "Ready"))
        let generator = FrameGenerator(model: model!, tokenizer: tokenizer!)
        let codec = self.codec!
        generator.silenceProbe = { frames in
            // The codec is causal, so decoding the whole chunk so far gives the true tail.
            let codes = (0..<ModelConstants.numCodebooks).map { c in frames.map { $0[1 + c] } }
            return Self.trailingSilenceSeconds(try Self.decodeBlocking(codec, codes: codes))
        }
        let plan = try GenerationPlan(request)
        var allFrames: [Frame] = []
        var allSamples: [Float] = []
        var timings = SpeechTimings()
        var eos = true
        // Generation is synchronous GPU work; decode each finished chunk right after it lands.
        let results = try generator.generate(plan: plan, config: request.config, shouldContinue: { !Task.isCancelled }) { event in
            switch event {
            case .chunkStarted(let i, let n, let text): continuation.yield(.chunkStarted(index: i, count: n, text: text))
            case .prefilled(let rows): continuation.yield(.prefill(rows: rows))
            case .frame(let i, let max): continuation.yield(.frame(index: i, maxFrames: max))
            case .retry(let i, let attempt, _, _): continuation.yield(.retry(chunkIndex: i, attempt: attempt))
            case .chunkFinished: break
            }
        }
        try Task.checkCancellation()
        for r in results {
            timings.prefillSeconds += r.prefillSeconds
            timings.decodeSeconds += r.decodeSeconds
            timings.promptRows += r.promptRows
            timings.framesGenerated += r.frames.count
            eos = eos && r.finishedByEOS
            allFrames += r.frames
            guard !r.frames.isEmpty else { continue }
            let t0 = Date()
            let samples = try await codec.decode(codes: r.codes)
            timings.codecSeconds += Date().timeIntervalSince(t0)
            allSamples += samples
            continuation.yield(.audio(samples: samples, chunkIndex: r.index))
        }
        let take = Take(text: plan.turnText, codes: (0..<ModelConstants.numCodebooks).map { c in allFrames.map { $0[1 + c] } })
        continuation.yield(.finished(SpeechResult(take: take, samples: allSamples, timings: timings, finishedByEOS: eos)))
    }
}

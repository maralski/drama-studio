// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation
import Synchronization

/// The Drama 3 engine: the Metal backbone and tower, the Core ML codec, and
/// the Swift tokenizer.
///
/// Create one per app and share it; it is an actor. Requests run one at a
/// time: a second ``synthesize(_:)`` waits until the first finishes. Loading
/// maps the 39.6 GB of weights without copying them and leaves the process
/// about 40 GB resident. On an M3 Max a warm load takes about 5 s, a decode
/// frame about 24 ms of GPU time, and generation runs at about 1.8 times
/// real time.
public actor DramaSynthesizer: SpeechSynthesizer {
    /// The snapshot the engine loads.
    public nonisolated let location: ModelLocation
    /// Where the Core ML codec models live.
    public let codecDirectory: URL
    /// Where tuned kernel schedules live, one JSON file per Mac, macOS
    /// version, kernel source and snapshot.
    public let tuningDirectory: URL
    /// `~/Library/Application Support/DramaStudio/Tuning`.
    public static var defaultTuningDirectory: URL { TuningCache.defaultDirectory }
    private var model: DramaModel?
    private var tokenizer: Tokenizer?
    private var codec: DramaCodec?
    private var loading: Task<Void, Error>?

    /// Creates an engine. Nothing loads until ``load(onPhase:)`` or the first
    /// request.
    ///
    /// - Parameters:
    ///   - location: The snapshot, usually ``ModelLocation/huggingFaceCache(repo:)``.
    ///   - codecDirectory: Where the Core ML codec models live, usually
    ///     ``DramaCodec/defaultDirectory``.
    ///   - tuningDirectory: Where tuned kernel schedules live.
    public init(location: ModelLocation, codecDirectory: URL, tuningDirectory: URL = DramaSynthesizer.defaultTuningDirectory) {
        self.location = location
        self.codecDirectory = codecDirectory
        self.tuningDirectory = tuningDirectory
    }

    /// Generation runs synchronously on this actor, so status reads go
    /// through a lock instead of waiting for the run to end. Nil until the
    /// model loads, then whether its schedule came from the tuning cache.
    private nonisolated let tuned = Mutex<Bool?>(nil)

    /// Whether the model is loaded. Readable while a request runs.
    public nonisolated var isLoaded: Bool { tuned.withLock { $0 != nil } }
    /// Nil until the model loads; then true when the kernel schedule came
    /// from the tuning cache, including one this load just tuned, and false
    /// when tuning failed and the default schedule runs. A tuned schedule makes a one-row forward about 20
    /// percent faster on an M3 Max.
    public nonisolated var isTuned: Bool? { tuned.withLock { $0 } }

    /// Loads the codec, the tokenizer and the weights, and compiles the
    /// kernels. Requests call it themselves, so calling it first only moves
    /// the cost. Concurrent calls share one load.
    ///
    /// The first load on a Mac also tunes the kernel schedule and caches it
    /// (about 30 s on an M3 Max), so later loads find the cache and skip the
    /// step. A tuning failure leaves the default schedule in place; it never
    /// fails the load.
    ///
    /// - Throws: ``CodecError`` when the codec models are missing,
    ///   ``TokenizerError`` for a bad `tokenizer.json`, ``EngineError`` for a
    ///   bad snapshot or no Metal device.
    public func load(onPhase: @escaping @Sendable (String) -> Void) async throws {
        if isLoaded { return }
        if let loading { return try await loading.value }
        let task = Task { [location, codecDirectory, tuningDirectory] in
            let codec = try await DramaCodec(directory: codecDirectory)
            let tokenizer = try Tokenizer(url: location.directory.appendingPathComponent("tokenizer.json"))
            let model = try DramaModel(location: location, tuningDirectory: tuningDirectory)
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

    /// Clones a voice from a clip: loads the engine if needed, reads the
    /// clip (any format AVFoundation opens), mixes it to mono at 44.1 kHz, and
    /// encodes it. A 10 s clip takes about 2.1 s in the encoder on an M3 Max.
    /// `transcript` must say what the clip says.
    public func encodeReference(audioURL: URL, transcript: String, name: String) async throws -> ReferenceVoice {
        try await load()
        let samples = try DramaCodec.loadAudio(url: audioURL)
        let codes = try await codec!.encode(samples: samples)
        return ReferenceVoice(name: name, transcript: transcript, codes: codes, sourceFileName: audioURL.lastPathComponent)
    }

    /// Decodes frames to 44.1 kHz mono samples, 2048 per frame, loading the
    /// engine if needed. The decoder takes about 1.5 s per 512-frame window
    /// (23.8 s of audio) on an M3 Max.
    public func decode(frames: [Frame]) async throws -> [Float] {
        try await load()
        let codes = (0..<ModelConstants.numCodebooks).map { c in frames.map { $0[1 + c] } }
        return try await codec!.decode(codes: codes)
    }

    /// A 2048-sample frame whose RMS is under this is silent.
    package nonisolated static let silenceThreshold: Float = 0.01

    /// Whether one frame of samples is silent: its RMS is under ``silenceThreshold``.
    package nonisolated static func isSilent(_ frame: ArraySlice<Float>) -> Bool {
        var energy: Float = 0
        for x in frame { energy += x * x }
        return (energy / Float(frame.count)).squareRoot() < silenceThreshold
    }

    /// Seconds of trailing audio whose frames are all silent.
    package nonisolated static func trailingSilenceSeconds(_ samples: [Float]) -> Double {
        let n = ModelConstants.samplesPerFrame
        var silentFrames = 0
        var end = samples.count / n * n
        while end >= n, isSilent(samples[(end - n) ..< end]) {
            silentFrames += 1
            end -= n
        }
        return Double(silentFrames * n) / Double(ModelConstants.sampleRate)
    }

    /// Runs a codec decode from synchronous generator code.
    package nonisolated static func decodeBlocking(_ codec: DramaCodec, codes: [[Int32]]) throws -> [Float] {
        try blocking { try await codec.decode(codes: codes) }
    }

    package nonisolated static func decodeBlocking(_ codec: DramaCodec, codes: [[Int32]], window: DecodeWindow) throws -> [Float] {
        try blocking { try await codec.decode(codes: codes, window: window) }
    }

    private nonisolated static func blocking(_ work: @escaping @Sendable () async throws -> [Float]) throws -> [Float] {
        let group = DispatchGroup()
        group.enter()
        nonisolated(unsafe) var result: Result<[Float], Error> = .failure(EngineError.badModel("codec did not answer"))
        Task.detached {
            do { result = .success(try await work()) } catch { result = .failure(error) }
            group.leave()
        }
        group.wait()
        return try result.get()
    }

    /// Speaks `request`, streaming ``SpeechEvent``s that end with
    /// ``SpeechEvent/finished(_:)``. Loads the engine first if needed, which
    /// shows as ``SpeechEvent/loadingModel(progress:detail:)`` events.
    /// Cancelling the task that consumes the stream stops the generation.
    /// The stream throws ``PromptError`` for a request the prompt builder
    /// rejects.
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
        generator.codec = { codes, window in try Self.decodeBlocking(codec, codes: codes, window: window) }
        let plan = try GenerationPlan(request)
        let results = try generator.generate(plan: plan, config: request.config, shouldContinue: { !Task.isCancelled }) { event in
            switch event {
            case .chunkStarted(let i, let n, let text): continuation.yield(.chunkStarted(index: i, count: n, text: text))
            case .prefilled(let rows): continuation.yield(.prefill(rows: rows))
            case .frame(let i, let max): continuation.yield(.frame(index: i, maxFrames: max))
            case .retry(let i, let attempt, _, _): continuation.yield(.retry(chunkIndex: i, attempt: attempt))
            case .partialAudio(let samples, let i, let frames): continuation.yield(.partialAudio(samples: samples, chunkIndex: i, frameRange: frames))
            case .chunkFinished(let r): if !r.samples.isEmpty { continuation.yield(.audio(samples: r.samples, chunkIndex: r.index)) }
            }
        }
        try Task.checkCancellation()
        var timings = SpeechTimings()
        for r in results {
            timings.prefillSeconds += r.prefillSeconds
            timings.decodeSeconds += r.decodeSeconds
            timings.codecSeconds += r.codecSeconds
            timings.promptRows += r.promptRows
            timings.framesGenerated += r.frames.count
        }
        let allFrames = results.flatMap(\.frames)
        let allSamples = results.flatMap(\.samples)
        let eos = results.allSatisfy(\.finishedByEOS)
        let take = Take(text: plan.turnText, codes: (0..<ModelConstants.numCodebooks).map { c in allFrames.map { $0[1 + c] } })
        let sentences = results.map { SpokenSentence(text: plan.chunks[$0.index].line, frames: $0.frames.count) }
        continuation.yield(.finished(SpeechResult(take: take, samples: allSamples, timings: timings, finishedByEOS: eos, sentences: sentences)))
    }
}

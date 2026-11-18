// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation
import Synchronization
internal import DramaTTS

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
    public static var defaultTuningDirectory: URL { applicationSupport("Tuning") }
    /// `~/Library/Application Support/DramaStudio/Packed`, where the packed
    /// form of the weights is built on a snapshot's first load.
    package static var packedDirectory: URL { applicationSupport("Packed") }
    private static func applicationSupport(_ folder: String) -> URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("DramaStudio/\(folder)", isDirectory: true)
    }
    private var model: DramaModel?
    private var tokenizer: DramaTokenizer?
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
            let (tokenizer, model) = try Drama3Errors.translating {
                (try DramaTokenizer(url: location.directory.appendingPathComponent("tokenizer.json")),
                 try DramaModel(checkpoint: location.checkpoint, tuningDirectory: tuningDirectory, packedDirectory: Self.packedDirectory))
            }
            model.warmUpKernels()
            if model.needsTuning {
                onPhase("Tuning kernels for this Mac, about a minute")
                do {
                    try model.tune(cacheDirectory: tuningDirectory)
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

    private func install(model: DramaModel, tokenizer: DramaTokenizer, codec: DramaCodec) {
        self.model = model
        self.tokenizer = tokenizer
        self.codec = codec
        let fromCache = model.hasTunedSchedule
        tuned.withLock { $0 = fromCache }
    }

    public nonisolated var engineDescription: EngineDescription {
        EngineDescription(engine: .drama3, modelID: SpeechEngineID.drama3.install.repo, snapshot: location.directory.lastPathComponent, isMock: false)
    }

    /// Encodes a clip to Drama 3 codec codes, loading the engine if needed.
    /// A 10 s clip takes about 2.1 s in the encoder on an M3 Max.
    public func encodeVoice(samples: [Float], transcript: String) async throws -> VoiceEncoding {
        try await load()
        return .drama3(codes: try await codec!.encode(samples: samples))
    }

    /// A 2048-sample frame whose RMS is under this is silent.
    package nonisolated static let silenceThreshold = DramaSilence.threshold

    /// Whether one frame of samples is silent: its RMS is under ``silenceThreshold``.
    package nonisolated static func isSilent(_ frame: ArraySlice<Float>) -> Bool { DramaSilence.isSilent(frame) }

    /// Seconds of trailing audio whose frames are all silent.
    package nonisolated static func trailingSilenceSeconds(_ samples: [Float]) -> Double { DramaSilence.trailingSeconds(samples) }

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
        let plan = try GenerationPlan(request)
        let options = request.config.drama
        let outline = try SpeechOutline(request, engine: .drama3)
        let verifier = try LocalWordVerifier.prepare(lines: plan.chunks.map(\.line), language: request.language, config: request.config, shouldContinue: { !Task.isCancelled })
        continuation.yield(.loadingModel(progress: 0, detail: "Mapping weights"))
        try await load { detail in continuation.yield(.loadingModel(progress: 0, detail: detail)) }
        continuation.yield(.loadingModel(progress: 1, detail: "Ready"))
        let generator = FrameGenerator(model: model!, tokenizer: tokenizer!)
        let codec = self.codec!.audioCodec
        generator.codec = { codes, window in try codec.decodeBlocking(codes: codes, window: window) }
        generator.onDiscardedAttempt = { attempt in
            SpeechAttemptAudit.record(samples: attempt.samples, line: attempt.line, reason: attempt.reason, frames: attempt.frames, error: attempt.error)
        }
        var chain = VerifiedChunkChain(verifier)
        let review = verifier.map { _ in
            DramaAttemptReview(accept: { samples, index in try chain.accept(samples, at: index, shouldContinue: { !Task.isCancelled }) },
                               commit: { try chain.commit() })
        }
        let results = try Drama3Errors.translating { try generator.generate(plan: plan, config: options, review: review, shouldContinue: { !Task.isCancelled }) { event in
            switch event {
            case .chunkStarted(let i, let n, let text): continuation.yield(.chunkStarted(index: i, count: n, text: text))
            case .prefilled(let rows): continuation.yield(.prefill(rows: rows))
            case .frame(let i, let max): continuation.yield(.frame(index: i, maxFrames: max))
            case .retry(let i, let attempt, _, _): continuation.yield(.retry(chunkIndex: i, attempt: attempt))
            case .partialAudio(let samples, let i, let frames): continuation.yield(.partialAudio(samples: samples, chunkIndex: i, frameRange: frames))
            case .chunkFinished(let r): if !r.samples.isEmpty { continuation.yield(.audio(samples: r.samples, chunkIndex: r.index)) }
            }
        } }
        try Task.checkCancellation()
        let timings = SpeechTimings(generator.timings)
        let allFrames = results.flatMap(\.frames)
        let allSamples = results.flatMap(\.samples)
        let eos = results.allSatisfy(\.finishedByEOS)
        let take = Take(text: plan.turnText, context: .drama3(codes: (0..<ModelConstants.numCodebooks).map { c in allFrames.map { $0[1 + c] } }))
        let sentences = results.map { SpokenSentence(text: plan.chunks[$0.index].line, samples: $0.frames.count * ModelConstants.samplesPerFrame) }
        guard !results.isEmpty else { throw PromptError.emptyText("spoken words") }
        continuation.yield(.finished(SpeechResult(take: take, samples: allSamples, timings: timings, finishedByEOS: eos, sentences: sentences,
                                                wordCoverage: request.config.wordVerification == .required ? .recognized : .unchecked,
                                                notices: outline.notices + (verifier?.notices ?? []))))
    }
}

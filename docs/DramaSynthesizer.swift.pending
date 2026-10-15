import Foundation

/// The real engine behind the app: Metal backbone and tower, Core ML codec,
/// Swift tokenizer. Requests run one at a time on a serial queue.
public actor DramaSynthesizer: SpeechSynthesizer {
    public let location: ModelLocation
    public let codecDirectory: URL
    private var model: DramaModel?
    private var tokenizer: Tokenizer?
    private var codec: DramaCodec?
    private var loading: Task<Void, Error>?

    public init(location: ModelLocation, codecDirectory: URL) {
        self.location = location
        self.codecDirectory = codecDirectory
    }

    public var isLoaded: Bool { model != nil && codec != nil }

    public func load() async throws {
        if isLoaded { return }
        if let loading { return try await loading.value }
        let task = Task { [location, codecDirectory] in
            let codec = try await DramaCodec(directory: codecDirectory)
            let tokenizer = try Tokenizer(url: location.directory.appendingPathComponent("tokenizer.json"))
            let model = try DramaModel(location: location)
            model.warmUp()
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
        try await load()
        continuation.yield(.loadingModel(progress: 1, detail: "Ready"))
        let generator = FrameGenerator(model: model!, tokenizer: tokenizer!)
        let references = request.references.map { Drama3Prompt.Reference(transcript: $0.transcript, codes: $0.codes) }
        var allFrames: [Frame] = []
        var allSamples: [Float] = []
        var timings = SpeechTimings()
        var eos = true
        var pendingChunks: [FrameGenerator.ChunkResult] = []
        // Generation is synchronous GPU work; decode each finished chunk right after it lands.
        let results = try generator.generate(text: request.text, references: references, config: request.config, shouldContinue: { !Task.isCancelled }) { event in
            switch event {
            case .chunkStarted(let i, let n, let text): continuation.yield(.chunkStarted(index: i, count: n, text: text))
            case .prefilled(let rows): continuation.yield(.prefill(rows: rows))
            case .frame(let i, let max): continuation.yield(.frame(index: i, maxFrames: max))
            case .chunkFinished(let r): pendingChunks.append(r)
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
            let samples = try await codec!.decode(codes: r.codes)
            timings.codecSeconds += Date().timeIntervalSince(t0)
            allSamples += samples
            continuation.yield(.audio(samples: samples, chunkIndex: r.index))
        }
        _ = pendingChunks
        continuation.yield(.finished(SpeechResult(frames: allFrames, samples: allSamples, timings: timings, finishedByEOS: eos)))
    }
}

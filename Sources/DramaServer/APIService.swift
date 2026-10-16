import DramaEngine
import Foundation

public struct SpeechJob: Sendable, Equatable {
    public enum Progress: Sendable, Equatable {
        case starting
        case loadingModel(String)
        case generating(chunk: Int, of: Int)
    }

    public let id: UUID
    public var progress: Progress

    public init(id: UUID, progress: Progress) {
        self.id = id
        self.progress = progress
    }

    public var summary: String {
        switch progress {
        case .starting: "Starting"
        case .loadingModel(let detail): "Loading model: \(detail)"
        case .generating(let chunk, let count): "Generating chunk \(chunk) of \(count)"
        }
    }
}

/// A queue of size one: the engine runs one generation at a time, and a
/// request that finds the slot taken is turned away instead of waiting.
public actor SpeechJobSlot {
    public private(set) var current: SpeechJob?
    private let onChange: @Sendable (SpeechJob?) -> Void
    private var estimate: GenerationEstimate?
    private var estimateStart = ContinuousClock.now

    public init(onChange: @escaping @Sendable (SpeechJob?) -> Void = { _ in }) {
        self.onChange = onChange
    }

    func claim() -> UUID? {
        guard current == nil else { return nil }
        let job = SpeechJob(id: UUID(), progress: .starting)
        current = job
        onChange(job)
        return job.id
    }

    /// Starts timing a generation; a live session begins one per sentence.
    func begin(_ id: UUID, estimate: GenerationEstimate) {
        guard current?.id == id else { return }
        self.estimate = estimate
        estimateStart = .now
    }

    public var progress: EstimateSnapshot? {
        guard current != nil, let estimate else { return nil }
        let elapsed = (ContinuousClock.now - estimateStart).seconds
        return EstimateSnapshot(elapsedSeconds: elapsed, remainingSeconds: estimate.remainingSeconds(at: elapsed),
                                framesDone: estimate.framesDone, framesExpected: estimate.framesExpected)
    }

    func advance(_ id: UUID, with event: SpeechEvent) {
        guard current?.id == id else { return }
        estimate?.apply(event, at: (ContinuousClock.now - estimateStart).seconds)
        let progress: SpeechJob.Progress
        switch event {
        case .loadingModel(_, let detail): progress = .loadingModel(detail)
        case .chunkStarted(let index, let count, _): progress = .generating(chunk: index + 1, of: count)
        default: return
        }
        guard current?.progress != progress else { return }
        current?.progress = progress
        onChange(current)
    }

    func release(_ id: UUID) {
        guard current?.id == id else { return }
        current = nil
        estimate = nil
        onChange(nil)
    }
}

/// `/v1/health`'s view of the running generation; `remainingSeconds` is nil
/// once the last chunk is in its silence tail or a retry runs.
public struct EstimateSnapshot: Sendable, Equatable, Encodable {
    public var elapsedSeconds: Double
    public var remainingSeconds: Double?
    public var framesDone: Int
    public var framesExpected: Int

    enum CodingKeys: String, CodingKey {
        case elapsedSeconds = "elapsed_seconds", remainingSeconds = "remaining_seconds", framesDone = "frames_done", framesExpected = "frames_expected"
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode((elapsedSeconds * 10).rounded() / 10, forKey: .elapsedSeconds)
        try container.encode(remainingSeconds.map { ($0 * 10).rounded() / 10 }, forKey: .remainingSeconds)
        try container.encode(framesDone, forKey: .framesDone)
        try container.encode(framesExpected, forKey: .framesExpected)
    }
}

private extension Duration {
    var seconds: Double { Double(components.seconds) + Double(components.attoseconds) / 1e18 }
}

public struct EngineInfo: Sendable, Equatable {
    public enum Kind: String, Sendable { case metal, mock }

    public var kind: Kind
    public var modelID: String
    public var snapshot: String
    public var version: String

    public init(describing synthesizer: any SpeechSynthesizer, version: String) {
        if let engine = synthesizer as? DramaSynthesizer {
            kind = .metal
            snapshot = engine.location.directory.lastPathComponent
        } else {
            kind = .mock
            snapshot = "mock"
        }
        modelID = "fishaudio/drama-3"
        self.version = version
    }
}

/// Answers HTTP requests against an engine and a voice store. Everything
/// here is transport-free so tests drive it without a socket.
public final class APIService: Sendable {
    public static let defaultPort: UInt16 = 8780
    public static let retryAfterSeconds = 5

    public let synthesizer: any SpeechSynthesizer
    public let voices: any VoiceStore
    public let token: String?
    public let info: EngineInfo
    public let jobs: SpeechJobSlot
    public let takes = TakeCache()
    private let engineBusyElsewhere: @Sendable () async -> Bool

    public init(
        synthesizer: any SpeechSynthesizer,
        voices: any VoiceStore,
        token: String?,
        version: String,
        jobs: SpeechJobSlot = SpeechJobSlot(),
        engineBusyElsewhere: @escaping @Sendable () async -> Bool = { false }
    ) {
        self.synthesizer = synthesizer
        self.voices = voices
        self.token = token
        self.info = EngineInfo(describing: synthesizer, version: version)
        self.jobs = jobs
        self.engineBusyElsewhere = engineBusyElsewhere
    }

    /// `body` carries the request body as it arrives, for routes where
    /// `APIRequest.streamsBody` holds; other routes read `request.body`.
    public func respond(to request: HTTPRequest, body: AsyncThrowingStream<Data, Error>? = nil) async -> APIResponse {
        guard APIToken.authorizes(request, token: token) else {
            return .error(APIError(.unauthorized, "Missing or wrong bearer token. Send Authorization: Bearer <token>, with the token from ~/Library/Application Support/DramaStudio/api-token.", headers: ["WWW-Authenticate": "Bearer"]))
        }
        switch APIRequest.parse(request) {
        case .failure(let error):
            return .error(error)
        case .success(let api):
            do {
                if case .live(let parameters) = api { return try await live(parameters, body: body ?? AsyncThrowingStream { $0.finish() }) }
                return try await handle(api)
            } catch let error as APIError {
                return .error(error)
            } catch {
                return .error(APIError(.internalServerError, Self.describe(error)))
            }
        }
    }

    private func handle(_ request: APIRequest) async throws -> APIResponse {
        switch request {
        case .health:
            let tuned: Bool? = if let engine = synthesizer as? DramaSynthesizer { engine.isTuned } else { false }
            return .json(Health(modelLoaded: await synthesizer.isLoaded, engine: info.kind.rawValue, busy: await isBusy(), tuned: tuned, version: info.version,
                                estimate: await jobs.progress))
        case .models:
            return .json(ModelList(data: [ModelEntry(id: info.modelID, snapshot: info.snapshot)]))
        case .listVoices:
            return .json(VoiceList(voices: await voices.allVoices().map(VoiceSummary.init)))
        case .addVoice(let new):
            var voice = try await withAudioFile(new.audio, field: "the clip") { url in
                try await synthesizer.encodeReference(audioURL: url, transcript: new.transcript, name: new.name)
            }
            voice.sourceFileName = new.fileName
            try await voices.add(voice)
            return .json(VoiceSummary(voice), status: .created)
        case .deleteVoice(let key):
            guard case .success(let voice) = VoiceLookup.find(key, in: await voices.allVoices()) else {
                throw APIError(.notFound, "No voice \"\(key)\" in the library.")
            }
            try await voices.remove(voice.id)
            return .json(["deleted": voice.id.uuidString])
        case .speech(let parameters):
            return try await speak(parameters)
        case .live:
            throw APIError(.internalServerError, "The live route reads its body as it arrives.")
        case .take(let key):
            guard let id = UUID(uuidString: key), let entry = await takes.entry(id) else {
                throw APIError(.notFound, "No take \"\(key)\"; the server keeps its last \(TakeCache.capacity) takes.")
            }
            return .json(TakeSummary(entry))
        }
    }

    /// A take id in `previous_takes` stands for that take and the cached
    /// takes it was made after, so naming a line's latest take carries its
    /// first take (the voice) and its recent history. Without input, the
    /// request retakes the line, voice and character of the last named take.
    private func resolve(_ parameters: SpeechParameters) async throws -> (line: String, casting: VoiceCasting, character: VoiceDesign?, previous: [Take], lineage: [UUID]) {
        var previous: [Take] = []
        var lineage: [UUID] = []
        var named: TakeCache.Entry?
        for source in parameters.previousTakes {
            switch source {
            case .inline(let take):
                previous.append(take)
            case .cached(let id):
                guard let entry = await takes.entry(id) else {
                    throw APIError(.notFound, "No take \(id.uuidString); the server keeps its last \(TakeCache.capacity) takes.")
                }
                for earlier in entry.lineage where !lineage.contains(earlier) {
                    guard let found = await takes.entry(earlier) else { continue }
                    previous.append(found.take)
                    lineage.append(earlier)
                }
                named = entry
            }
        }
        let inherits = parameters.casting == .none && parameters.character == nil
        let casting = inherits ? named?.casting ?? .none : parameters.casting
        let character = inherits ? named?.character : parameters.character
        guard let line = parameters.text ?? named?.line else { throw APIError.bad("input is required: the text to speak.") }
        return (line, casting, character, previous, lineage)
    }

    private func isBusy() async -> Bool {
        let claimed = await jobs.current != nil
        if claimed { return true }
        return await engineBusyElsewhere()
    }

    private static let busy = APIError(.tooManyRequests, "Drama Studio is already generating; one request runs at a time. Retry shortly.", headers: ["Retry-After": String(retryAfterSeconds)])

    private func speak(_ parameters: SpeechParameters) async throws -> APIResponse {
        let busy = Self.busy
        let resolved = try await resolve(parameters)
        let fromLibrary = try await libraryReferences(for: resolved.casting)
        guard await !engineBusyElsewhere(), let job = await jobs.claim() else { throw busy }
        let request: SpeechRequest
        let plan: GenerationPlan
        do {
            var references = fromLibrary ?? []
            if fromLibrary == nil { references = try await encodedClips(for: resolved.casting) }
            request = SpeechRequest(text: resolved.line, references: references, character: resolved.character, note: parameters.note,
                                    previousTakes: resolved.previous, config: parameters.config)
            plan = try GenerationPlan(request)
        } catch {
            await jobs.release(job)
            if let error = error as? PromptError { throw APIError.bad(error.description) }
            throw error
        }
        await jobs.begin(job, estimate: GenerationEstimate(plan: plan))
        let takeID = UUID()
        let remember: @Sendable (Take) async -> Void = { [takes] take in
            await takes.insert(TakeCache.Entry(id: takeID, take: take, line: resolved.line, casting: resolved.casting, character: resolved.character,
                                               lineage: resolved.lineage + [takeID]))
        }
        switch parameters.delivery {
        case .pcmStream, .events:
            let framing = StreamFraming(parameters.delivery)!
            let (body, generation) = stream(request, framing: framing, job: job, takeID: takeID, remember: remember)
            var headers = framing.headers
            headers["X-Drama-Take-Id"] = takeID.uuidString
            headers["X-Drama-Estimated-Seconds"] = String(format: "%.1f", GenerationEstimate(plan: plan).remainingSeconds(at: 0) ?? 0)
            return APIResponse(status: .ok, headers: headers, body: .chunked(body), abandon: { generation.cancel() })
        case .file(let format):
            let start = ContinuousClock.now
            let result: SpeechResult?
            do {
                result = try await run(request, job: job)
            } catch {
                await jobs.release(job)
                throw error
            }
            await jobs.release(job)
            guard let result else { throw APIError(.internalServerError, "The engine stopped without producing audio.") }
            await remember(result.take)
            let elapsed = ContinuousClock.now - start
            var headers = format == .pcm ? Self.pcmHeaders(contentType: format.contentType) : ["Content-Type": format.contentType]
            headers["X-Drama-Take-Id"] = takeID.uuidString
            headers["X-Drama-Frames"] = String(result.take.frameCount)
            headers["X-Drama-Seconds"] = String(format: "%.3f", Double(result.samples.count) / Double(AudioEncoding.sampleRate))
            headers["X-Drama-Generation-Seconds"] = String(format: "%.3f", Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) / 1e18)
            return APIResponse(status: .ok, headers: headers, body: .data(try AudioEncoding.encode(result.samples, as: format)))
        }
    }

    private static func pcmHeaders(contentType: String) -> [String: String] {
        ["Content-Type": contentType, "X-Drama-Sample-Rate": String(AudioEncoding.sampleRate), "X-Drama-Format": "pcm_s16le_mono"]
    }

    private func run(_ request: SpeechRequest, job: UUID) async throws -> SpeechResult? {
        var result: SpeechResult?
        for try await event in synthesizer.synthesize(request) {
            await jobs.advance(job, with: event)
            if case .finished(let finished) = event { result = finished }
        }
        return result
    }

    private func stream(_ request: SpeechRequest, framing: StreamFraming, job: UUID, takeID: UUID,
                        remember: @escaping @Sendable (Take) async -> Void) -> (AsyncThrowingStream<Data, Error>, Task<Void, Never>) {
        let (stream, out) = AsyncThrowingStream<Data, Error>.makeStream()
        let task = Task {
            do {
                var sentences = 0
                if let result = try await forward(request, framing: framing, job: job, into: out, sentences: &sentences) {
                    await remember(result.take)
                    Self.send(framing.done(["take_id": takeID.uuidString, "frames": result.take.frameCount,
                                            "seconds": Self.seconds(result.samples)]), to: out)
                }
                await jobs.release(job)
                out.finish()
            } catch {
                Self.send(framing.error(Self.describe(error)), to: out)
                await jobs.release(job)
                out.finish(throwing: error)
            }
        }
        out.onTermination = { _ in task.cancel() }
        return (stream, task)
    }

    /// Streams one request's audio, and a `sentence` event per chunk, into
    /// `out`; `sentences` numbers the chunks across a live session.
    private func forward(_ request: SpeechRequest, framing: StreamFraming, job: UUID, into out: AsyncThrowingStream<Data, Error>.Continuation,
                         sentences: inout Int) async throws -> SpeechResult? {
        var audio = StreamedAudio()
        var texts: [Int: String] = [:]
        var result: SpeechResult?
        for try await event in synthesizer.synthesize(request) {
            await jobs.advance(job, with: event)
            Self.send(framing.audio(audio.newSamples(from: event)), to: out)
            switch event {
            case .chunkStarted(let index, _, let text):
                texts[index] = text
            case .audio(let samples, let index):
                Self.send(framing.sentence(index: sentences, text: texts[index] ?? "", seconds: Self.seconds(samples)), to: out)
                sentences += 1
            case .finished(let finished):
                result = finished
            default:
                break
            }
        }
        return result
    }

    /// Speaks text as it arrives in `body`, one sentence at a time, on one
    /// connection. Each sentence's take is the voice and history of the next.
    private func live(_ parameters: SpeechParameters, body: AsyncThrowingStream<Data, Error>) async throws -> APIResponse {
        let references = try await libraryReferences(for: parameters.casting) ?? []
        guard await !engineBusyElsewhere(), let job = await jobs.claim() else { throw Self.busy }
        let framing = StreamFraming(parameters.delivery)!
        let (stream, out) = AsyncThrowingStream<Data, Error>.makeStream()
        let task = Task {
            var lines = LineSplitter()
            var text = LiveText()
            var spoken: [Take] = []
            var takeIDs: [String] = []
            var sentences = 0
            var seconds = 0.0
            func say(_ sentence: String) async throws {
                let context = spoken.count <= 3 ? spoken : [spoken[0]] + spoken.suffix(2)
                let request = SpeechRequest(text: sentence, references: references, character: parameters.character,
                                            previousTakes: context, config: parameters.config)
                await jobs.begin(job, estimate: GenerationEstimate(plan: try GenerationPlan(request)))
                guard let result = try await forward(request, framing: framing, job: job, into: out, sentences: &sentences) else { return }
                spoken.append(result.take)
                seconds += Self.seconds(result.samples)
                let id = UUID()
                await takes.insert(TakeCache.Entry(id: id, take: result.take, line: sentence, casting: parameters.casting,
                                                   character: parameters.character, lineage: [id]))
                takeIDs.append(id.uuidString)
            }
            do {
                for try await data in body {
                    for line in lines.feed(data) {
                        for sentence in text.append(line: line) { try await say(sentence) }
                    }
                }
                for sentence in lines.finish().flatMap({ text.append(line: $0) }) + text.flush() { try await say(sentence) }
                Self.send(framing.done(["sentences": sentences, "seconds": (seconds * 1000).rounded() / 1000, "take_ids": takeIDs]), to: out)
                await jobs.release(job)
                out.finish()
            } catch {
                let message = (error as? PromptError)?.description ?? (error as? APIError)?.message ?? Self.describe(error)
                Self.send(framing.error(message), to: out)
                await jobs.release(job)
                out.finish(throwing: error)
            }
        }
        out.onTermination = { _ in task.cancel() }
        return APIResponse(status: .ok, headers: framing.headers, body: .chunked(stream), abandon: { task.cancel() })
    }

    private static func send(_ data: Data, to out: AsyncThrowingStream<Data, Error>.Continuation) {
        if !data.isEmpty { out.yield(data) }
    }

    private static func seconds(_ samples: [Float]) -> Double {
        (Double(samples.count) / Double(AudioEncoding.sampleRate) * 1000).rounded() / 1000
    }

    /// Library voices resolve before the job slot is claimed, so a typo gets
    /// a 400 without turning another request away. Nil for clips.
    private func libraryReferences(for casting: VoiceCasting) async throws -> [ReferenceVoice]? {
        switch casting {
        case .none:
            return []
        case .voice(let key):
            return [try VoiceLookup.find(key, in: await voices.allVoices()).get()]
        case .speakers(let keys):
            let library = await voices.allVoices()
            return try keys.map { try VoiceLookup.find($0, in: library).get() }
        case .clips:
            return nil
        }
    }

    /// Encoding runs on the engine, so it happens only once the job slot is held.
    private func encodedClips(for casting: VoiceCasting) async throws -> [ReferenceVoice] {
        guard case .clips(let clips) = casting else { return [] }
        var encoded: [ReferenceVoice] = []
        for (index, clip) in clips.enumerated() {
            encoded.append(try await withAudioFile(clip.audio, field: "references[\(index)]") { url in
                try await synthesizer.encodeReference(audioURL: url, transcript: clip.transcript, name: "Reference \(index + 1)")
            })
        }
        return encoded
    }

    private func withAudioFile<T>(_ source: AudioSource, field: String, _ body: (URL) async throws -> T) async throws -> T {
        let url: URL
        var temporary: URL?
        switch source {
        case .path(let path):
            url = URL(filePath: (path as NSString).expandingTildeInPath)
            guard FileManager.default.fileExists(atPath: url.path) else { throw APIError.bad("\(field): no file at \(path).") }
        case .bytes(let data):
            url = FileManager.default.temporaryDirectory.appending(path: "drama-clip-\(UUID().uuidString).\(AudioEncoding.fileExtension(sniffing: data))")
            try data.write(to: url)
            temporary = url
        }
        defer { if let temporary { try? FileManager.default.removeItem(at: temporary) } }
        do {
            return try await body(url)
        } catch {
            throw APIError.bad("\(field): couldn't read the audio (\(Self.describe(error))).")
        }
    }

    static func describe(_ error: Error) -> String {
        (error as? LocalizedError)?.errorDescription ?? String(describing: error)
    }
}

struct Health: Encodable {
    var modelLoaded: Bool
    var engine: String
    var busy: Bool
    var tuned: Bool?
    var version: String
    var estimate: EstimateSnapshot?

    enum CodingKeys: String, CodingKey { case status, model_loaded, engine, busy, tuned, version, estimate }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode("ok", forKey: .status)
        try container.encode(modelLoaded, forKey: .model_loaded)
        try container.encode(engine, forKey: .engine)
        try container.encode(busy, forKey: .busy)
        try container.encode(tuned, forKey: .tuned)
        try container.encode(version, forKey: .version)
        try container.encodeIfPresent(estimate, forKey: .estimate)
    }
}

struct ModelEntry: Encodable {
    var id: String
    var object = "model"
    var owned_by = "fishaudio"
    var snapshot: String

    init(id: String, snapshot: String) {
        self.id = id
        self.snapshot = snapshot
    }
}

struct ModelList: Encodable {
    var object = "list"
    var data: [ModelEntry]
}

struct VoiceSummary: Encodable {
    var id: String
    var name: String
    var transcript: String
    var seconds: Double
    var source_file: String?

    init(_ voice: ReferenceVoice) {
        id = voice.id.uuidString
        name = voice.name
        transcript = voice.transcript
        seconds = (voice.durationSeconds * 1000).rounded() / 1000
        source_file = voice.sourceFileName
    }
}

struct VoiceList: Encodable {
    var voices: [VoiceSummary]
}

struct TakeSummary: Encodable {
    var id: String
    var input: String
    var text: String
    var seconds: Double
    var frames: Int
    var codes_base64: String
    var previous_take_ids: [String]

    init(_ entry: TakeCache.Entry) {
        id = entry.id.uuidString
        input = entry.line
        text = entry.take.text
        seconds = (entry.take.durationSeconds * 1000).rounded() / 1000
        frames = entry.take.frameCount
        codes_base64 = TakeCodes.encode(entry.take.codes)
        previous_take_ids = entry.lineage.dropLast().map(\.uuidString)
    }
}

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

    func advance(_ id: UUID, with event: SpeechEvent) {
        guard current?.id == id else { return }
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
        onChange(nil)
    }
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
    public static let defaultPort: UInt16 = 8765
    public static let retryAfterSeconds = 5

    public let synthesizer: any SpeechSynthesizer
    public let voices: any VoiceStore
    public let token: String?
    public let info: EngineInfo
    public let jobs: SpeechJobSlot
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

    public func respond(to request: HTTPRequest) async -> APIResponse {
        guard APIToken.authorizes(request, token: token) else {
            return .error(APIError(.unauthorized, "Missing or wrong bearer token. Send Authorization: Bearer <token>, with the token from ~/Library/Application Support/DramaStudio/api-token.", headers: ["WWW-Authenticate": "Bearer"]))
        }
        switch APIRequest.parse(request) {
        case .failure(let error):
            return .error(error)
        case .success(let api):
            do {
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
            let tuned: Bool? = if let engine = synthesizer as? DramaSynthesizer { await engine.isTuned } else { false }
            return .json(Health(modelLoaded: await synthesizer.isLoaded, engine: info.kind.rawValue, busy: await isBusy(), tuned: tuned, version: info.version))
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
        }
    }

    private func isBusy() async -> Bool {
        let claimed = await jobs.current != nil
        if claimed { return true }
        return await engineBusyElsewhere()
    }

    private func speak(_ parameters: SpeechParameters) async throws -> APIResponse {
        let busy = APIError(.tooManyRequests, "Drama Studio is already generating; one request runs at a time. Retry shortly.", headers: ["Retry-After": String(Self.retryAfterSeconds)])
        let fromLibrary = try await libraryReferences(for: parameters.casting)
        guard await !engineBusyElsewhere(), let job = await jobs.claim() else { throw busy }
        let request: SpeechRequest
        do {
            var references = fromLibrary ?? []
            if fromLibrary == nil { references = try await encodedClips(for: parameters.casting) }
            request = SpeechRequest(text: parameters.text, references: references, config: parameters.config)
        } catch {
            await jobs.release(job)
            throw error
        }
        switch parameters.delivery {
        case .pcmStream:
            return APIResponse(status: .ok, headers: Self.pcmHeaders(contentType: AudioFormat.pcm.contentType), body: .chunked(stream(request, job: job)))
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
            let elapsed = ContinuousClock.now - start
            var headers = format == .pcm ? Self.pcmHeaders(contentType: format.contentType) : ["Content-Type": format.contentType]
            headers["X-Drama-Frames"] = String(result.frames.count)
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

    private func stream(_ request: SpeechRequest, job: UUID) -> AsyncThrowingStream<Data, Error> {
        let (stream, continuation) = AsyncThrowingStream<Data, Error>.makeStream()
        let task = Task { [synthesizer, jobs] in
            do {
                for try await event in synthesizer.synthesize(request) {
                    await jobs.advance(job, with: event)
                    if case .audio(let samples, _) = event { continuation.yield(AudioEncoding.pcm16(samples)) }
                }
                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
            await jobs.release(job)
        }
        continuation.onTermination = { _ in task.cancel() }
        return stream
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

    enum CodingKeys: String, CodingKey { case status, model_loaded, engine, busy, tuned, version }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode("ok", forKey: .status)
        try container.encode(modelLoaded, forKey: .model_loaded)
        try container.encode(engine, forKey: .engine)
        try container.encode(busy, forKey: .busy)
        try container.encode(tuned, forKey: .tuned)
        try container.encode(version, forKey: .version)
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

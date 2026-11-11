// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
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

    public var engine: SpeechEngineID

    public init(describing synthesizer: any SpeechSynthesizer, version: String) {
        let description = synthesizer.engineDescription
        kind = description.isMock ? .mock : .metal
        snapshot = description.snapshot
        modelID = description.modelID
        engine = description.engine
        self.version = version
    }
}

/// Answers HTTP requests against an engine and a voice store. Everything
/// here is transport-free so tests drive it without a socket.
public final class APIService: Sendable {
    public static let defaultPort: UInt16 = 8780
    public static let retryAfterSeconds = 5

    /// Every engine the server can run; requests use the selected one unless they name another.
    public let engines: SpeechEngines
    public let voices: any VoiceStore
    public let token: String?
    public let version: String
    public let jobs: SpeechJobSlot
    public let takes = TakeCache()
    private let engineBusyElsewhere: @Sendable () async -> Bool
    /// Why generation is refused right now, such as license terms not yet
    /// agreed to; nil allows it. Asked before every request that speaks or
    /// clones a voice.
    private let refusal: @Sendable () async -> String?
    /// Read per request: the app edits the file while the server runs.
    private let pronunciation: @Sendable () -> PronunciationDictionary

    public init(
        synthesizer: any SpeechSynthesizer,
        voices: any VoiceStore,
        token: String?,
        version: String,
        jobs: SpeechJobSlot = SpeechJobSlot(),
        engineBusyElsewhere: @escaping @Sendable () async -> Bool = { false },
        refusal: @escaping @Sendable () async -> String? = { nil },
        pronunciation: @escaping @Sendable () -> PronunciationDictionary = { (try? PronunciationDictionary.load()) ?? PronunciationDictionary() }
    ) {
        engines = synthesizer as? SpeechEngines ?? SpeechEngines(only: synthesizer)
        self.voices = voices
        self.token = token
        self.version = version
        self.jobs = jobs
        self.engineBusyElsewhere = engineBusyElsewhere
        self.refusal = refusal
        self.pronunciation = pronunciation
    }

    /// The selected engine, read per call: the app switches engines while the server runs.
    public var info: EngineInfo { EngineInfo(describing: engines, version: version) }

    /// `body` carries the request body as it arrives, for routes where
    /// `APIRequest.streamsBody` holds; other routes read `request.body`.
    public func respond(to request: HTTPRequest, body: AsyncThrowingStream<Data, Error>? = nil) async -> APIResponse {
        guard APIToken.authorizes(request, token: token) else {
            return .error(APIError(.unauthorized, "Missing or wrong bearer token. Send Authorization: Bearer <token>, with the token from \((APIToken.defaultURL.path as NSString).abbreviatingWithTildeInPath).", headers: ["WWW-Authenticate": "Bearer"]))
        }
        switch APIRequest.parse(request) {
        case .failure(let error):
            return .error(error)
        case .success(let api):
            if api.generates, let message = await refusal() { return .error(APIError(.forbidden, message)) }
            do {
                if case .live(let parameters) = api { return try await live(parameters, body: body ?? AsyncThrowingStream { $0.finish() }) }
                return try await handle(api)
            } catch let error as APIError {
                return .error(error)
            } catch let error as PromptError {
                return .error(.bad(error.description))
            } catch SpeechVerificationError.language(let reason) {
                return .error(.bad(Self.describe(SpeechVerificationError.language(reason))))
            } catch let error as EngineError {
                guard case .noClip = error else { return .error(APIError(.internalServerError, error.description)) }
                return .error(.bad(error.description))
            } catch {
                return .error(APIError(.internalServerError, Self.describe(error)))
            }
        }
    }

    private func handle(_ request: APIRequest) async throws -> APIResponse {
        switch request {
        case .health:
            let info = info
            return .json(Health(modelLoaded: await engines.isLoaded, engine: info.kind.rawValue, speechEngine: info.engine.rawValue, busy: await isBusy(),
                                tuned: engines.isTuned, version: info.version, estimate: await jobs.progress))
        case .models:
            let selected = engines.selected
            return .json(ModelList(data: SpeechEngineID.allCases.compactMap { engine in
                engines.description(of: engine).map { ModelEntry($0, selected: engine == selected) }
            }))
        case .listVoices:
            let builtIn = SpeechEngineID.allCases.filter(engines.available.contains).flatMap(\.builtInVoices)
            return .json(VoiceList(voices: await voices.allVoices().map(VoiceSummary.init), built_in: builtIn.map(BuiltInVoiceSummary.init)))
        case .addVoice(let new):
            var (voice, clip) = try await withAudioFile(new.audio, field: "the clip") { url in
                try await engines.cloneVoice(audioURL: url, transcript: new.transcript, name: new.name)
            }
            voice.sourceFileName = new.fileName
            try await voices.add(voice, clip: clip)
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
    /// `engine` is the engine that made the last named take.
    /// `continued` holds every cached take the request builds on.
    private func resolve(_ parameters: SpeechParameters) async throws -> (line: String, casting: VoiceCasting, character: VoiceDesign?, previous: [Take],
                                                                           lineage: [UUID], engine: SpeechEngineID?, continued: [TakeCache.Entry]) {
        var continued: [TakeCache.Entry] = []
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
                    continued.append(found)
                }
                named = entry
            }
        }
        let inherits = parameters.casting == .none && parameters.character == nil
        let casting = inherits ? named?.casting ?? .none : parameters.casting
        let character = inherits ? named?.character : parameters.character
        guard let line = parameters.text ?? named?.line else { throw APIError.bad("input is required: the text to speak.") }
        return (line, casting, character, previous, lineage, named?.take.context.engine, continued)
    }

    private func isBusy() async -> Bool {
        let claimed = await jobs.current != nil
        if claimed { return true }
        return await engineBusyElsewhere()
    }

    private static let busy = APIError(.tooManyRequests, "Drama Studio is already generating; one request runs at a time. Retry shortly.", headers: ["Retry-After": String(retryAfterSeconds)])

    private func rules(for parameters: SpeechParameters) -> PronunciationDictionary {
        parameters.applyPronunciation ? pronunciation() : PronunciationDictionary()
    }

    /// `line` is the client's text, which the take cache keeps; the engine
    /// speaks it rewritten by `rules`.
    private static func speechRequest(speaking line: String, rules: PronunciationDictionary, parameters: SpeechParameters,
                                      references: [ReferenceVoice], builtInVoice: BuiltInVoice?, character: VoiceDesign?,
                                      previousTakes: [Take]) -> SpeechRequest {
        var request = SpeechRequest(text: rules.apply(to: line), references: references, character: character, note: parameters.note,
                                    previousTakes: previousTakes, config: parameters.config)
        request.builtInVoice = builtInVoice
        request.language = parameters.language
        return request
    }

    /// The engine the request names; else, for a request that continues a
    /// cached take, the engine that made it; else the selected one. It stays
    /// fixed for the whole request.
    private func synthesizer(for parameters: SpeechParameters, continuing take: SpeechEngineID? = nil) throws -> any SpeechSynthesizer {
        let installed = SpeechEngineID.allCases.filter(engines.available.contains).map(\.rawValue).joined(separator: ", ")
        let engine: SpeechEngineID
        switch parameters.model {
        case .unspecified: engine = take ?? engines.selected
        case .engine(let named): engine = named
        case .unknown(let name):
            throw APIError.bad("No engine is called \(name). Installed: \(installed); GET /v1/models lists them. "
                + "OpenAI's \(ModelChoice.openAIModels.joined(separator: ", ")) run on the selected engine.")
        }
        do {
            return try engines.pinned(to: engine)
        } catch {
            throw APIError.bad("\(engine.capabilities.name) isn't installed on this server. Installed: \(installed); GET /v1/models lists them.")
        }
    }

    private func speak(_ parameters: SpeechParameters) async throws -> APIResponse {
        let busy = Self.busy
        let resolved = try await resolve(parameters)
        let synthesizer = try synthesizer(for: parameters, continuing: resolved.engine)
        let rules = rules(for: parameters)
        let cast = try await cast(resolved.casting)
        if let voice = cast.builtIn {
            try voice.check(on: synthesizer.engine)
            // On its own engine an earlier take is the voice a built-in voice continues from, so one spoken in another voice would replace it.
            if let other = resolved.continued.first(where: { $0.take.context.engine == voice.engine && $0.casting != cast.casting }) {
                throw APIError.bad("The take \(other.id.uuidString) was not spoken in the built-in voice \(voice.name), so it can't continue in it. "
                    + "Leave out previous_takes to start in \(voice.name), or leave out voice to keep the take's voice.")
            }
        }
        // Decided before the job slot is claimed, so a language the word check
        // can't hear is a 400 on a stream as on a file, and turns no one away.
        let checkNotices = try await LocalWordVerifier.notices(text: rules.apply(to: resolved.line), language: parameters.language,
                                                               config: parameters.config)
        guard await !engineBusyElsewhere(), let job = await jobs.claim() else { throw busy }
        let request: SpeechRequest
        let outline: SpeechOutline
        do {
            var references = cast.references
            if case .clips = cast.casting { references = try await encodedClips(for: cast.casting, on: synthesizer) }
            request = Self.speechRequest(speaking: resolved.line, rules: rules, parameters: parameters, references: references,
                                         builtInVoice: cast.builtIn, character: resolved.character, previousTakes: resolved.previous)
            outline = try SpeechOutline(request, engine: synthesizer.engine)
        } catch {
            await jobs.release(job)
            throw error
        }
        let estimate = GenerationEstimate(outline: outline)
        await jobs.begin(job, estimate: estimate)
        let takeID = UUID()
        let remember: @Sendable (Take) async -> Void = { [takes] take in
            await takes.insert(TakeCache.Entry(id: takeID, take: take, line: resolved.line, casting: cast.casting, character: resolved.character,
                                               lineage: resolved.lineage + [takeID]))
        }
        switch parameters.delivery {
        case .pcmStream, .events:
            let framing = StreamFraming(parameters.delivery)!
            let (body, generation) = stream(request, on: synthesizer, framing: framing, job: job, takeID: takeID, remember: remember)
            var headers = framing.headers
            headers["X-Drama-Take-Id"] = takeID.uuidString
            headers.merge(Self.engineHeaders(synthesizer.engineDescription)) { $1 }
            headers["X-Drama-Estimated-Seconds"] = String(format: "%.1f", estimate.remainingSeconds(at: 0) ?? 0)
            headers["X-Drama-Notices"] = Self.header(outline.notices + checkNotices, on: outline.engine)
            return APIResponse(status: .ok, headers: headers, body: .chunked(body), abandon: { generation.cancel() })
        case .file(let format):
            let start = ContinuousClock.now
            let result: SpeechResult?
            do {
                result = try await run(request, on: synthesizer, job: job)
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
            headers.merge(Self.engineHeaders(synthesizer.engineDescription)) { $1 }
            if let frames = result.take.apiFrames {
                headers["X-Drama-Frames"] = String(frames)
                headers["X-Drama-Generated-Frames"] = String(result.timings.framesGenerated)
            }
            headers["X-Drama-Samples"] = String(result.take.sampleCount)
            headers["X-Drama-Word-Coverage"] = result.wordCoverage.rawValue
            headers["X-Drama-Notices"] = Self.header(result.notices, on: synthesizer.engine)
            headers["X-Drama-Verification-Seconds"] = String(format: "%.3f", result.timings.verificationSeconds)
            headers["X-Drama-Seconds"] = String(format: "%.3f", Double(result.samples.count) / Double(AudioEncoding.sampleRate))
            headers["X-Drama-Generation-Seconds"] = String(format: "%.3f", Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) / 1e18)
            return APIResponse(status: .ok, headers: headers, body: .data(try AudioEncoding.encode(result.samples, as: format)))
        }
    }

    /// Which engine and model made the take.
    private static func engineHeaders(_ description: EngineDescription) -> [String: String] {
        ["X-Drama-Engine": description.engine.rawValue, "X-Drama-Model": description.modelID]
    }

    private static func pcmHeaders(contentType: String) -> [String: String] {
        ["Content-Type": contentType, "X-Drama-Sample-Rate": String(AudioEncoding.sampleRate), "X-Drama-Format": "pcm_s16le_mono"]
    }

    private func run(_ request: SpeechRequest, on synthesizer: any SpeechSynthesizer, job: UUID) async throws -> SpeechResult? {
        var result: SpeechResult?
        for try await event in synthesizer.synthesize(request) {
            await jobs.advance(job, with: event)
            if case .finished(let finished) = event { result = finished }
        }
        return result
    }

    private func stream(_ request: SpeechRequest, on synthesizer: any SpeechSynthesizer, framing: StreamFraming, job: UUID, takeID: UUID,
                        remember: @escaping @Sendable (Take) async -> Void) -> (AsyncThrowingStream<Data, Error>, Task<Void, Never>) {
        let (stream, out) = AsyncThrowingStream<Data, Error>.makeStream()
        let task = Task {
            do {
                var sentences = 0
                if let result = try await forward(request, on: synthesizer, framing: framing, job: job, into: out, sentences: &sentences) {
                    await remember(result.take)
                    var done: [String: Any] = ["take_id": takeID.uuidString, "engine": result.take.context.engine.rawValue, "samples": result.take.sampleCount,
                                               "word_coverage": result.wordCoverage.rawValue, "seconds": Self.seconds(result.samples)]
                    if let frames = result.take.apiFrames { done["frames"] = frames }
                    if !result.notices.isEmpty { done["notices"] = result.notices.map { $0.message(on: result.take.context.engine) } }
                    Self.send(framing.done(done), to: out)
                }
                await jobs.release(job)
                out.finish()
            } catch {
                Self.send(framing.error(Self.describe(error)), to: out)
                await jobs.release(job)
                framing.finish(out, after: error)
            }
        }
        out.onTermination = { _ in task.cancel() }
        return (stream, task)
    }

    /// Streams one request's audio, and a `sentence` event per chunk, into
    /// `out`; `sentences` numbers the chunks across a live session.
    private func forward(_ request: SpeechRequest, on synthesizer: any SpeechSynthesizer, framing: StreamFraming, job: UUID,
                         into out: AsyncThrowingStream<Data, Error>.Continuation,
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
        let synthesizer = try synthesizer(for: parameters)
        let cast = try await cast(parameters.casting)
        try cast.builtIn?.check(on: synthesizer.engine)
        let rules = rules(for: parameters)
        // Text arrives after the headers, so only a named language can be refused up front.
        if parameters.config.wordVerification == .required, let language = parameters.language {
            _ = try LocalWordVerifier.checkedLanguage(of: "", named: language)
        }
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
            var notices: [SpeechNotice] = []
            // The language of the sentence before, which a sentence too short to tell is in.
            var language = parameters.language
            func say(_ sentence: String) async throws {
                let context = spoken.count <= 3 ? spoken : [spoken[0]] + spoken.suffix(2)
                var request = Self.speechRequest(speaking: sentence, rules: rules, parameters: parameters,
                                                 references: cast.references, builtInVoice: cast.builtIn, character: parameters.character,
                                                 previousTakes: context)
                if parameters.language == nil, let told = try? ScriptLanguage.identify(request.text, fallback: language ?? ScriptLanguage.userLanguage) {
                    language = told
                    request.language = told
                }
                await jobs.begin(job, estimate: GenerationEstimate(outline: try SpeechOutline(request, engine: synthesizer.engine)))
                guard let result = try await forward(request, on: synthesizer, framing: framing, job: job, into: out, sentences: &sentences) else { return }
                spoken.append(result.take)
                notices += result.notices.filter { !notices.contains($0) }
                seconds += Self.seconds(result.samples)
                let id = UUID()
                await takes.insert(TakeCache.Entry(id: id, take: result.take, line: sentence, casting: cast.casting,
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
                var done: [String: Any] = ["sentences": sentences, "seconds": (seconds * 1000).rounded() / 1000, "take_ids": takeIDs,
                                           "engine": synthesizer.engine.rawValue]
                if !notices.isEmpty { done["notices"] = notices.map { $0.message(on: synthesizer.engine) } }
                Self.send(framing.done(done), to: out)
                await jobs.release(job)
                out.finish()
            } catch {
                let message = (error as? PromptError)?.description ?? (error as? APIError)?.message ?? Self.describe(error)
                Self.send(framing.error(message), to: out)
                await jobs.release(job)
                framing.finish(out, after: error)
            }
        }
        out.onTermination = { _ in task.cancel() }
        return APIResponse(status: .ok, headers: framing.headers.merging(Self.engineHeaders(synthesizer.engineDescription)) { $1 },
                           body: .chunked(stream), abandon: { task.cancel() })
    }

    private static func send(_ data: Data, to out: AsyncThrowingStream<Data, Error>.Continuation) {
        if !data.isEmpty { out.yield(data) }
    }

    /// What the engine could not perform, as a JSON array of messages in
    /// ASCII so it fits a header; nil when there is nothing to say.
    private static func header(_ notices: [SpeechNotice], on engine: SpeechEngineID) -> String? {
        guard !notices.isEmpty,
              let json = try? JSONSerialization.data(withJSONObject: notices.map { $0.message(on: engine) }, options: .withoutEscapingSlashes) else { return nil }
        return String(decoding: json, as: UTF8.self).unicodeScalars.map { scalar in
            scalar.isASCII ? String(scalar) : String(scalar).utf16.map { String(format: "\\u%04x", $0) }.joined()
        }.joined()
    }

    private static func seconds(_ samples: [Float]) -> Double {
        (Double(samples.count) / Double(AudioEncoding.sampleRate) * 1000).rounded() / 1000
    }

    /// A casting with its keys looked up. `casting` is what the take cache
    /// keeps: a built-in voice under its qualified name, so a retake finds
    /// the same voice whatever the library holds by then.
    private struct Cast {
        var casting: VoiceCasting
        /// Library voices in speaker order; empty for clips, which are encoded once the job slot is held.
        var references: [ReferenceVoice] = []
        var builtIn: BuiltInVoice?
    }

    /// Voice keys resolve before the job slot is claimed, so a typo gets a
    /// 400 without turning another request away.
    private func cast(_ casting: VoiceCasting) async throws -> Cast {
        switch casting {
        case .none, .clips:
            return Cast(casting: casting)
        case .voice(let key):
            switch try VoiceLookup.resolve(key, in: await voices.allVoices()).get() {
            case .library(let voice): return Cast(casting: casting, references: [voice])
            case .builtIn(let voice): return Cast(casting: .voice(voice.qualifiedName), builtIn: voice)
            }
        case .speakers(let keys):
            let library = await voices.allVoices()
            return Cast(casting: casting, references: try keys.map { key in
                switch try VoiceLookup.resolve(key, in: library).get() {
                case .library(let voice): return voice
                case .builtIn(let voice):
                    throw APIError.bad("speaker_voices takes library voices, and \"\(key)\" is a built-in \(voice.engine.capabilities.name) voice. "
                        + "A request speaks in one built-in voice; pass it as voice.")
                }
            })
        }
    }

    /// Encoding runs on the engine, so it happens only once the job slot is held.
    private func encodedClips(for casting: VoiceCasting, on synthesizer: any SpeechSynthesizer) async throws -> [ReferenceVoice] {
        guard case .clips(let clips) = casting else { return [] }
        var encoded: [ReferenceVoice] = []
        for (index, clip) in clips.enumerated() {
            encoded.append(try await withAudioFile(clip.audio, field: "references[\(index)]") { url in
                try await synthesizer.cloneVoice(audioURL: url, transcript: clip.transcript, name: "Reference \(index + 1)").voice
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
    /// `metal`, or `mock` for the preview engine.
    var engine: String
    /// The engine new requests use: `drama3` or `ming`.
    var speechEngine: String
    var busy: Bool
    var tuned: Bool?
    var version: String
    var estimate: EstimateSnapshot?

    enum CodingKeys: String, CodingKey { case status, model_loaded, engine, speech_engine, busy, tuned, version, estimate }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode("ok", forKey: .status)
        try container.encode(modelLoaded, forKey: .model_loaded)
        try container.encode(engine, forKey: .engine)
        try container.encode(speechEngine, forKey: .speech_engine)
        try container.encode(busy, forKey: .busy)
        try container.encode(tuned, forKey: .tuned)
        try container.encode(version, forKey: .version)
        try container.encodeIfPresent(estimate, forKey: .estimate)
    }
}

/// One installed engine. `id` is its Hugging Face model id; a speech
/// request's `model` takes that, the `engine` id or the `name`.
struct ModelEntry: Encodable {
    var id: String
    var object = "model"
    var owned_by: String
    var engine: String
    var name: String
    var snapshot: String
    var mock: Bool
    /// The engine requests use when they name none.
    var selected: Bool

    init(_ description: EngineDescription, selected: Bool) {
        id = description.modelID
        owned_by = String(description.modelID.split(separator: "/").first ?? "")
        engine = description.engine.rawValue
        name = description.engine.capabilities.name
        snapshot = description.snapshot
        mock = description.isMock
        self.selected = selected
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

/// A built-in voice of an installed engine. A speech request's `voice`
/// takes `name`, or `voice` when a library voice has the same name.
struct BuiltInVoiceSummary: Encodable {
    var name: String
    var engine: String
    var voice: String
    var group: String
    var note: String?

    init(_ voice: BuiltInVoice) {
        name = voice.name
        engine = voice.engine.rawValue
        self.voice = voice.qualifiedName
        group = voice.group
        note = voice.note
    }
}

struct VoiceList: Encodable {
    var voices: [VoiceSummary]
    var built_in: [BuiltInVoiceSummary]
}

struct TakeSummary: Encodable {
    var id: String
    var engine: String
    var input: String
    var text: String
    var seconds: Double
    var samples: Int
    /// Drama 3 takes only; absent for another engine's take.
    var frames: Int?
    var codes_base64: String?
    var previous_take_ids: [String]

    init(_ entry: TakeCache.Entry) {
        id = entry.id.uuidString
        engine = entry.take.context.engine.rawValue
        input = entry.line
        text = entry.take.text
        seconds = (entry.take.durationSeconds * 1000).rounded() / 1000
        samples = entry.take.sampleCount
        frames = entry.take.apiFrames
        codes_base64 = entry.take.apiCodesBase64
        previous_take_ids = entry.lineage.dropLast().map(\.uuidString)
    }
}

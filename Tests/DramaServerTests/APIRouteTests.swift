// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import DramaEngine
import DramaServer
import XCTest

actor RecordingSynthesizer: SpeechSynthesizer {
    let mock: MockSynthesizer
    private(set) var requests: [SpeechRequest] = []

    init(pace: MockSynthesizer.Pace = .instant) { mock = MockSynthesizer(pace: pace) }

    var isLoaded: Bool { get async { await mock.isLoaded } }
    func load(onPhase: @escaping @Sendable (String) -> Void) async throws { try await mock.load(onPhase: onPhase) }
    func encodeReference(audioURL: URL, transcript: String, name: String) async throws -> ReferenceVoice {
        try await mock.encodeReference(audioURL: audioURL, transcript: transcript, name: name)
    }
    func decode(frames: [Frame]) async throws -> [Float] { try await mock.decode(frames: frames) }
    private func record(_ request: SpeechRequest) { requests.append(request) }
    nonisolated func synthesize(_ request: SpeechRequest) -> AsyncThrowingStream<SpeechEvent, Error> {
        let inner = mock.synthesize(request)
        return AsyncThrowingStream { continuation in
            let task = Task {
                await self.record(request)
                do {
                    for try await event in inner { continuation.yield(event) }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }
}

final class APIRouteTests: XCTestCase {
    private var directory: URL!
    private var engine: RecordingSynthesizer!
    private var service: APIService!

    override func setUp() async throws {
        directory = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        engine = RecordingSynthesizer()
        service = APIService(synthesizer: engine, voices: VoiceDirectory(url: directory), token: nil, version: "test", pronunciation: { PronunciationDictionary() })
    }

    override func tearDown() async throws {
        try? FileManager.default.removeItem(at: directory)
    }

    private static let clip = AudioEncoding.wav((0..<44_100).map { Float(sin(Double($0) * 0.05) * 0.3) })
    private static let twoChunkText = String(repeating: "This sentence is about forty bytes long. ", count: 2)

    private func send(_ method: String, _ path: String, json: Any? = nil, body: Data? = nil, headers: [String: String] = [:]) async -> APIResponse {
        let data = body ?? json.map { try! JSONSerialization.data(withJSONObject: $0) } ?? Data()
        return await service.respond(to: HTTPRequest(method: method, path: path, headers: headers, body: data))
    }

    private func object(_ response: APIResponse) -> [String: Any] {
        (try? JSONSerialization.jsonObject(with: response.bodyData ?? Data())) as? [String: Any] ?? [:]
    }

    private func text(_ response: APIResponse) -> String { String(decoding: response.bodyData ?? Data(), as: UTF8.self) }

    private func addVoice(_ name: String) async -> String {
        let response = await send("POST", "/v1/voices", json: ["name": name, "transcript": "Hello there.", "audio_base64": Self.clip.base64EncodedString()])
        XCTAssertEqual(response.status, .created, text(response))
        return object(response)["id"] as? String ?? ""
    }

    func testHealthAndModels() async {
        var health = object(await send("GET", "/v1/health"))
        XCTAssertEqual(health["status"] as? String, "ok")
        XCTAssertEqual(health["engine"] as? String, "mock")
        XCTAssertEqual(health["model_loaded"] as? Bool, false)
        XCTAssertEqual(health["busy"] as? Bool, false)
        XCTAssertEqual(health["tuned"] as? Bool, false)
        XCTAssertEqual(health["version"] as? String, "test")
        _ = await send("POST", "/v1/audio/speech", json: ["input": "Load the model."])
        health = object(await send("GET", "/v1/health"))
        XCTAssertEqual(health["model_loaded"] as? Bool, true)

        let models = object(await send("GET", "/v1/models"))
        let entry = (models["data"] as? [[String: Any]])?.first
        XCTAssertEqual(entry?["id"] as? String, "fishaudio/drama-3")
        XCTAssertEqual(entry?["snapshot"] as? String, "mock")
    }

    func testVoicesCreateListDelete() async throws {
        let id = await addVoice("Narrator")
        var boundary = Data()
        boundary += Data("--b0\r\nContent-Disposition: form-data; name=\"transcript\"\r\n\r\nSecond clip.\r\n".utf8)
        boundary += Data("--b0\r\nContent-Disposition: form-data; name=\"file\"; filename=\"Villain.wav\"\r\n\r\n".utf8) + Self.clip + Data("\r\n--b0--\r\n".utf8)
        let multipart = await send("POST", "/v1/voices", body: boundary, headers: ["Content-Type": "multipart/form-data; boundary=b0"])
        XCTAssertEqual(multipart.status, .created, text(multipart))
        XCTAssertEqual(object(multipart)["name"] as? String, "Villain", "the file name stands in for a missing name field")
        XCTAssertEqual(object(multipart)["source_file"] as? String, "Villain.wav", "the upload's name, not the temporary file's")

        let listed = (object(await send("GET", "/v1/voices"))["voices"] as? [[String: Any]]) ?? []
        XCTAssertEqual(listed.map { $0["name"] as? String }, ["Narrator", "Villain"])
        XCTAssertEqual(listed.first?["seconds"] as? Double ?? 0, 1.0, accuracy: 0.05)
        XCTAssertEqual(VoiceDirectory(url: directory).load().voices.count, 2, "voices persist as files")

        let deleted = await send("DELETE", "/v1/voices/\(id)")
        XCTAssertEqual(deleted.status, .ok)
        XCTAssertEqual(object(deleted)["deleted"] as? String, id)
        let again = await send("DELETE", "/v1/voices/\(id)")
        XCTAssertEqual(again.status, .notFound)
        XCTAssertEqual(VoiceDirectory(url: directory).load().voices.map(\.name), ["Villain"])
    }

    func testVoiceUploadErrors() async {
        let missingTranscript = await send("POST", "/v1/voices", json: ["name": "A", "audio_base64": Self.clip.base64EncodedString()])
        XCTAssertEqual(missingTranscript.status, .badRequest)
        XCTAssertTrue(text(missingTranscript).contains("transcript"), text(missingTranscript))
        let noAudio = await send("POST", "/v1/voices", json: ["name": "A", "transcript": "Hi."])
        XCTAssertEqual(noAudio.status, .badRequest)
        let missingFile = await send("POST", "/v1/voices", json: ["name": "A", "transcript": "Hi.", "audio_path": "/nonexistent/clip.wav"])
        XCTAssertEqual(missingFile.status, .badRequest)
        let notAudio = await send("POST", "/v1/voices", json: ["name": "A", "transcript": "Hi.", "audio_base64": Data("not audio at all".utf8).base64EncodedString()])
        XCTAssertEqual(notAudio.status, .badRequest, text(notAudio))
    }

    func testSpeechReturnsAWavWithTimingHeaders() async {
        let response = await send("POST", "/v1/audio/speech", json: ["input": "Hello from the API.", "seed": 7])
        XCTAssertEqual(response.status, .ok, text(response))
        XCTAssertEqual(response.headers["Content-Type"], "audio/wav")
        let wav = response.bodyData ?? Data()
        XCTAssertEqual(String(decoding: wav.prefix(4), as: UTF8.self), "RIFF")
        let frames = Int(response.headers["X-Drama-Frames"] ?? "") ?? 0
        XCTAssertGreaterThan(frames, 0)
        XCTAssertEqual(wav.count - 44, frames * ModelConstants.samplesPerFrame * 2)
        XCTAssertEqual(Double(response.headers["X-Drama-Seconds"] ?? "") ?? 0, Double(frames * 2048) / 44_100, accuracy: 0.001)
        XCTAssertNotNil(Double(response.headers["X-Drama-Generation-Seconds"] ?? ""))
        let request = await engine.requests.last
        XCTAssertEqual(request?.config.seed, 7)
        XCTAssertEqual(request?.config.temperature, GenerationConfig().temperature, "expressiveness defaults to the vendor sampling")
    }

    func testSpeechFormatsAndControls() async {
        let pcm = await send("POST", "/v1/audio/speech", json: ["input": "Raw samples.", "response_format": "pcm"])
        XCTAssertEqual(pcm.headers["Content-Type"], "audio/pcm")
        XCTAssertEqual(pcm.headers["X-Drama-Format"], "pcm_s16le_mono")
        XCTAssertEqual(pcm.bodyData?.count, (Int(pcm.headers["X-Drama-Frames"] ?? "") ?? -1) * 2048 * 2)

        let m4a = await send("POST", "/v1/audio/speech", json: ["input": "Compressed.", "response_format": "m4a"])
        XCTAssertEqual(m4a.status, .ok, text(m4a))
        XCTAssertEqual(m4a.headers["Content-Type"], "audio/mp4")
        XCTAssertEqual(String(decoding: (m4a.bodyData ?? Data()).dropFirst(4).prefix(4), as: UTF8.self), "ftyp")

        _ = await send("POST", "/v1/audio/speech", json: ["input": "Wild.", "expressiveness": 1, "quick_preview": true])
        let request = await engine.requests.last
        XCTAssertEqual(request?.config, GenerationConfig.make(expressiveness: 1, quickPreview: true))
    }

    func testTTSAliasTakesText() async {
        let response = await send("POST", "/v1/tts", json: ["text": "Alias route."])
        XCTAssertEqual(response.status, .ok, text(response))
        XCTAssertEqual(response.headers["Content-Type"], "audio/wav")
        let missing = await send("POST", "/v1/tts", json: ["voice": "x"])
        XCTAssertEqual(missing.status, .badRequest)
        XCTAssertTrue(text(missing).hasPrefix("text is required"), text(missing))
        let request = await engine.requests.last
        XCTAssertEqual(request?.text, "Alias route.")
    }

    func testSpeechCastsLibraryVoicesAndClips() async {
        let narrator = await addVoice("Narrator")
        let villain = await addVoice("Villain")
        _ = await send("POST", "/v1/audio/speech", json: ["input": "By name.", "voice": "narrator"])
        var request = await engine.requests.last
        XCTAssertEqual(request?.references.map(\.id.uuidString), [narrator])

        _ = await send("POST", "/v1/audio/speech", json: ["input": "<|speaker:0|>A <|speaker:1|>B", "speaker_voices": ["1": narrator, "0": "Villain"]])
        request = await engine.requests.last
        XCTAssertEqual(request?.references.map(\.id.uuidString), [villain, narrator], "reference i speaks for speaker i")

        _ = await send("POST", "/v1/audio/speech", json: ["input": "Cookbook shape.", "references": [["audio_base64": Self.clip.base64EncodedString(), "text": "Hello there."]]])
        request = await engine.requests.last
        XCTAssertEqual(request?.references.map(\.transcript), ["Hello there."])
        XCTAssertEqual(request?.references.first?.codes.count, ModelConstants.numCodebooks)
    }

    func testSpeechBadRequests() async {
        let cases: [(Any, String)] = [
            (["voice": "x"], "input is required"),
            (["input": "  "], "input is required"),
            (["input": "Hi", "response_format": "flac"], "response_format"),
            (["input": "Hi", "stream": true, "response_format": "wav"], "stream needs"),
            (["input": "Hi", "expressiveness": 2], "expressiveness"),
            (["input": "Hi", "seed": -1], "seed"),
            (["input": "Hi", "voice": "Nobody"], "No voice"),
            (["input": "Hi", "voice": "a", "references": [["audio_path": "/x.wav", "text": "x"]]], "only one"),
            (["input": "Hi", "references": [["audio_path": "/x.wav"]]], "references[0].text"),
            (["input": "Hi", "references": [["text": "x"]]], "exactly one"),
            (["input": "Hi", "speaker_voices": ["0": "a", "2": "b"]], "without gaps"),
            (["input": "Hi", "speaker_voices": ["one": "a"]], "speaker numbers"),
            (["input": 5], "input has the wrong type"),
        ]
        for (body, expected) in cases {
            let response = await send("POST", "/v1/audio/speech", json: body)
            XCTAssertEqual(response.status, .badRequest, "\(body)")
            XCTAssertTrue(text(response).contains(expected), "\(body): \(text(response))")
            XCTAssertEqual(response.headers["Content-Type"], "text/plain; charset=utf-8")
        }
        let malformed = await send("POST", "/v1/audio/speech", body: Data("{not json".utf8))
        XCTAssertEqual(malformed.status, .badRequest)
        XCTAssertEqual(text(malformed), "The body is not valid JSON.\n")
        let empty = await engine.requests.count
        XCTAssertEqual(empty, 0, "no bad request reaches the engine")
    }

    func testUnknownRoutesAndMethods() async {
        let missing = await send("GET", "/v1/nothing")
        XCTAssertEqual(missing.status, .notFound)
        let wrongMethod = await send("GET", "/v1/audio/speech")
        XCTAssertEqual(wrongMethod.status, .methodNotAllowed)
        XCTAssertEqual(wrongMethod.headers["Allow"], "POST")
    }

    func testStreamingWritesPCMAsItIsDecoded() async throws {
        let response = await send("POST", "/v1/audio/speech", json: ["input": Self.twoChunkText, "stream": true, "response_format": "pcm"])
        XCTAssertEqual(response.status, .ok)
        XCTAssertEqual(response.headers["X-Drama-Sample-Rate"], "44100")
        XCTAssertEqual(response.headers["X-Drama-Format"], "pcm_s16le_mono")
        guard case .chunked(let stream) = response.body else { return XCTFail("not chunked") }
        var chunks: [Data] = []
        for try await chunk in stream { chunks.append(chunk) }
        XCTAssertGreaterThanOrEqual(chunks.count, 2, "at least one HTTP chunk per sentence")
        XCTAssertTrue(chunks.allSatisfy { $0.count % (ModelConstants.samplesPerFrame * 2) == 0 })
        let next = await send("POST", "/v1/tts", json: ["text": "Right after."])
        XCTAssertEqual(next.status, .ok, "the slot is free by the time the client sees the stream end")
    }

    func testSecondRequestWhileBusyGets429() async throws {
        let slow = APIService(synthesizer: MockSynthesizer(pace: .init(loadSeconds: 0, framesPerSecond: 20)), voices: VoiceDirectory(url: directory), token: nil, version: "test")
        let first = await slow.respond(to: HTTPRequest(method: "POST", path: "/v1/audio/speech", body: Data("{\"input\":\"Long enough to hold the slot.\",\"stream\":true}".utf8)))
        XCTAssertEqual(first.status, .ok)
        let second = await slow.respond(to: HTTPRequest(method: "POST", path: "/v1/tts", body: Data("{\"text\":\"Me too.\"}".utf8)))
        XCTAssertEqual(second.status, .tooManyRequests)
        XCTAssertEqual(second.headers["Retry-After"], String(APIService.retryAfterSeconds))
        let typo = await slow.respond(to: HTTPRequest(method: "POST", path: "/v1/tts", body: Data("{\"text\":\"Hi.\",\"voice\":\"Nobody\"}".utf8)))
        XCTAssertEqual(typo.status, .badRequest, "a bad voice is rejected before the slot is consulted")
        let health = await slow.respond(to: HTTPRequest(method: "GET", path: "/v1/health"))
        XCTAssertTrue(String(decoding: health.bodyData ?? Data(), as: UTF8.self).contains("\"busy\":true"))

        let elsewhere = APIService(synthesizer: MockSynthesizer(pace: .instant), voices: VoiceDirectory(url: directory), token: nil, version: "test", engineBusyElsewhere: { true })
        let blocked = await elsewhere.respond(to: HTTPRequest(method: "POST", path: "/v1/tts", body: Data("{\"text\":\"Hi.\"}".utf8)))
        XCTAssertEqual(blocked.status, .tooManyRequests, "a generation running in the window also holds the engine")
        _ = first
    }

    func testAbandonedStreamFreesTheEngine() async throws {
        let slow = APIService(synthesizer: MockSynthesizer(pace: .init(loadSeconds: 0, framesPerSecond: 200)), voices: VoiceDirectory(url: directory), token: nil, version: "test")
        let response = await slow.respond(to: HTTPRequest(method: "POST", path: "/v1/audio/speech", body: try JSONSerialization.data(withJSONObject: ["input": Self.twoChunkText, "stream": true])))
        guard case .chunked(let stream) = response.body else { return XCTFail("not chunked") }
        var iterator = stream.makeAsyncIterator()
        _ = try await iterator.next()
        response.abandon()
        let deadline = Date().addingTimeInterval(0.5)
        var busy = true
        while busy, Date() < deadline {
            busy = String(decoding: await slow.respond(to: HTTPRequest(method: "GET", path: "/v1/health")).bodyData ?? Data(), as: UTF8.self).contains("\"busy\":true")
            try await Task.sleep(for: .milliseconds(20))
        }
        XCTAssertFalse(busy, "a client that hangs up mid-stream releases the job slot")
    }

    func testARefusalAnswers403BeforeTheEngineRunsAndLeavesReadsAlone() async throws {
        let refused = APIService(synthesizer: engine, voices: VoiceDirectory(url: directory), token: nil, version: "test",
                                 refusal: { "Agree to the license terms in Drama Studio first." })
        let generating = [
            HTTPRequest(method: "POST", path: "/v1/audio/speech", body: Data("{\"input\":\"Hello.\"}".utf8)),
            HTTPRequest(method: "POST", path: "/v1/tts", body: Data("{\"text\":\"Hello.\"}".utf8)),
            HTTPRequest(method: "GET", path: "/v1/audio/speech/stream", query: ["input": "Hello."]),
            HTTPRequest(method: "POST", path: "/v1/audio/speech/live"),
            HTTPRequest(method: "POST", path: "/v1/voices",
                        body: try JSONSerialization.data(withJSONObject: ["name": "Ann", "transcript": "Hello there.", "audio_base64": Self.clip.base64EncodedString()])),
        ]
        for request in generating {
            let response = await refused.respond(to: request)
            XCTAssertEqual(response.status, .forbidden, request.path)
            XCTAssertEqual(text(response), "Agree to the license terms in Drama Studio first.\n")
        }
        let spoken = await engine.requests
        XCTAssertTrue(spoken.isEmpty, "a refused request never reaches the engine")
        let voices = await refused.respond(to: HTTPRequest(method: "GET", path: "/v1/voices"))
        XCTAssertEqual(String(decoding: voices.bodyData ?? Data(), as: UTF8.self), "{\"voices\":[]}", "no voice was cloned")
        for path in ["/v1/health", "/v1/models", "/v1/voices"] {
            let response = await refused.respond(to: HTTPRequest(method: "GET", path: path))
            XCTAssertEqual(response.status, .ok, path)
        }
    }

    func testTokenIsEnforced() async {
        let guarded = APIService(synthesizer: engine, voices: VoiceDirectory(url: directory), token: "t0ken", version: "test")
        let open = await guarded.respond(to: HTTPRequest(method: "GET", path: "/v1/health"))
        XCTAssertEqual(open.status, .ok)
        let denied = await guarded.respond(to: HTTPRequest(method: "GET", path: "/v1/voices"))
        XCTAssertEqual(denied.status, .unauthorized)
        XCTAssertEqual(denied.headers["WWW-Authenticate"], "Bearer")
        let allowed = await guarded.respond(to: HTTPRequest(method: "GET", path: "/v1/voices", headers: ["Authorization": "Bearer t0ken"]))
        XCTAssertEqual(allowed.status, .ok)
    }

    func testJobSlotReportsProgress() async throws {
        let seen = Seen()
        let slot = SpeechJobSlot { job in seen.append(job?.summary) }
        let tracked = APIService(synthesizer: engine, voices: VoiceDirectory(url: directory), token: nil, version: "test", jobs: slot)
        _ = await tracked.respond(to: HTTPRequest(method: "POST", path: "/v1/audio/speech", body: try JSONSerialization.data(withJSONObject: ["input": Self.twoChunkText])))
        let summaries = seen.values
        XCTAssertEqual(summaries.first, "Starting")
        XCTAssertTrue(summaries.contains("Generating chunk 1 of 2"), "\(summaries)")
        XCTAssertTrue(summaries.contains("Generating chunk 2 of 2"), "\(summaries)")
        XCTAssertEqual(summaries.last, .some(nil))
    }
}

final class Seen: @unchecked Sendable {
    private let lock = NSLock()
    private var stored: [String?] = []
    func append(_ value: String?) { lock.withLock { stored.append(value) } }
    var values: [String?] { lock.withLock { stored } }
}

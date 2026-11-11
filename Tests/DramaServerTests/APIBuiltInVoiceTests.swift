// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import DramaEngine
import DramaServer
import XCTest

/// A mock engine that keeps the requests it ran.
private final class RequestLog: @unchecked Sendable {
    private let lock = NSLock()
    private var kept: [SpeechRequest] = []
    var requests: [SpeechRequest] { lock.withLock { kept } }
    func add(_ request: SpeechRequest) { lock.withLock { kept.append(request) } }

    func factory(_ engine: SpeechEngineID) -> SpeechEngines.Factory {
        let mock = MockSynthesizer(engine: engine, pace: .instant)
        return SpeechEngines.Factory(description: mock.engineDescription) { LoggingSynthesizer(mock: mock, log: self) }
    }
}

private struct LoggingSynthesizer: SpeechSynthesizer {
    let mock: MockSynthesizer
    let log: RequestLog
    var isLoaded: Bool { get async { await mock.isLoaded } }
    func load(onPhase: @escaping @Sendable (String) -> Void) async throws { try await mock.load(onPhase: onPhase) }
    var engineDescription: EngineDescription { mock.engineDescription }
    var isTuned: Bool? { mock.isTuned }
    func encodeVoice(samples: [Float], transcript: String) async throws -> VoiceEncoding { try await mock.encodeVoice(samples: samples, transcript: transcript) }
    func synthesize(_ request: SpeechRequest) -> AsyncThrowingStream<SpeechEvent, Error> {
        log.add(request)
        return mock.synthesize(request)
    }
}

/// A request's `voice` reaches the engine as the built-in voice it names.
final class APIBuiltInVoiceTests: XCTestCase {
    private var directory: URL!
    private var log: RequestLog!
    private var service: APIService!
    private static let clip = AudioEncoding.wav((0..<44_100).map { Float(sin(Double($0) * 0.05) * 0.3) })
    private let emma = BuiltInVoice(engine: .ming, name: "US_Emma")
    private let george = BuiltInVoice(engine: .ming, name: "UK_George")

    override func setUp() async throws {
        directory = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        log = RequestLog()
        service = try makeService(selected: .ming, installed: [.drama3, .ming])
    }

    override func tearDown() async throws {
        try? FileManager.default.removeItem(at: directory)
    }

    private func makeService(selected: SpeechEngineID, installed: [SpeechEngineID]) throws -> APIService {
        let voices = VoiceDirectory(url: directory)
        let engines = try SpeechEngines(selected: selected, factories: Dictionary(uniqueKeysWithValues: installed.map { ($0, log.factory($0)) }), voices: voices)
        return APIService(synthesizer: engines, voices: voices, token: nil, version: "test", pronunciation: { PronunciationDictionary() })
    }

    private func send(_ method: String, _ path: String, json: Any? = nil, to service: APIService? = nil) async -> APIResponse {
        let body = json.map { try! JSONSerialization.data(withJSONObject: $0) } ?? Data()
        return await (service ?? self.service).respond(to: HTTPRequest(method: method, path: path, headers: [:], body: body))
    }

    private func text(_ response: APIResponse) -> String { String(decoding: response.bodyData ?? Data(), as: UTF8.self) }

    private func speak(_ json: [String: Any]) async -> APIResponse {
        await send("POST", "/v1/audio/speech", json: json.merging(["input": "The light is lit."]) { given, _ in given })
    }

    /// Catches the voice dropped on the way to the engine, which then speaks as US_Emma.
    func testAVoiceNameReachesTheEngineAsThatBuiltInVoice() async throws {
        for (key, voice) in [("UK_George", george), ("uk_george", george), ("ming:US_Emma", emma), ("灵晓声_高兴", BuiltInVoice(engine: .ming, name: "灵晓声_高兴"))] {
            let response = await speak(["voice": key])
            XCTAssertEqual(response.status, .ok, text(response))
            XCTAssertEqual(log.requests.last?.builtInVoice, voice, key)
            XCTAssertEqual(log.requests.last?.references, [], key)
        }
        _ = await speak([:])
        XCTAssertNil(log.requests.last?.builtInVoice, "a request with no voice names none")
    }

    func testTheVoiceListAddsTheInstalledEnginesBuiltInVoices() async throws {
        let added = await send("POST", "/v1/voices", json: ["name": "Keeper", "transcript": "Hello there.", "audio_base64": Self.clip.base64EncodedString()])
        XCTAssertEqual(added.status, .created, text(added))
        let listed = await send("GET", "/v1/voices")
        let list = try XCTUnwrap(JSONSerialization.jsonObject(with: listed.bodyData ?? Data()) as? [String: Any])
        XCTAssertEqual((list["voices"] as? [[String: Any]])?.map { $0["name"] as? String }, ["Keeper"])
        let builtIn = try XCTUnwrap(list["built_in"] as? [[String: Any]])
        XCTAssertEqual(builtIn.count, 241)
        let emma = try XCTUnwrap(builtIn.first { $0["name"] as? String == "US_Emma" })
        XCTAssertEqual(emma as? [String: String], ["name": "US_Emma", "engine": "ming", "voice": "ming:US_Emma", "group": "English"])
        XCTAssertEqual(builtIn.first { $0["name"] as? String == "灵晓声_高兴" }?["note"] as? String, "happy")

        let dramaOnly = try makeService(selected: .drama3, installed: [.drama3])
        let listedWithout = await send("GET", "/v1/voices", to: dramaOnly)
        let without = try XCTUnwrap(JSONSerialization.jsonObject(with: listedWithout.bodyData ?? Data()) as? [String: Any])
        XCTAssertEqual((without["built_in"] as? [Any])?.count, 0)
    }

    /// Catches a client's own voice replaced by a built-in voice of the same name.
    func testALibraryVoiceWinsItsNameAndTheQualifiedNameMeansTheBuiltInOne() async throws {
        let added = await send("POST", "/v1/voices", json: ["name": "US_Emma", "transcript": "Hello there.", "audio_base64": Self.clip.base64EncodedString()])
        XCTAssertEqual(added.status, .created, text(added))
        let library = await speak(["voice": "US_Emma"])
        XCTAssertEqual(library.status, .ok, text(library))
        XCTAssertEqual(log.requests.last?.references.map(\.name), ["US_Emma"])
        XCTAssertNil(log.requests.last?.builtInVoice)

        let builtIn = await speak(["voice": "ming:US_Emma"])
        XCTAssertEqual(builtIn.status, .ok, text(builtIn))
        XCTAssertEqual(log.requests.last?.references, [])
        XCTAssertEqual(log.requests.last?.builtInVoice, emma)

        let prefixed = await send("POST", "/v1/voices", json: ["name": "Ming: Narrator", "transcript": "Hello there.", "audio_base64": Self.clip.base64EncodedString()])
        XCTAssertEqual(prefixed.status, .created, text(prefixed))
        let old = await speak(["voice": "Ming: Narrator"])
        XCTAssertEqual(old.status, .ok, text(old))
        XCTAssertEqual(log.requests.last?.references.map(\.name), ["Ming: Narrator"], "a library name that starts like an engine id still names its voice")
    }

    /// Catches Drama 3 speaking a Ming voice's line in a voice of its own.
    func testAnotherEngineAnswers400OnEveryRouteAndNeverRuns() async throws {
        let message = "Drama 3 can't speak in UK_George, a built-in Ming voice; switch to Ming or choose another voice"
        let file = await speak(["voice": "UK_George", "model": "drama3"])
        let stream = await speak(["voice": "UK_George", "model": "drama3", "stream": true, "response_format": "pcm"])
        let events = await service.respond(to: HTTPRequest(method: "GET", path: "/v1/audio/speech/stream", query: ["input": "Hi there.", "voice": "UK_George", "model": "drama3"],
                                                           headers: ["Accept": "text/event-stream"]))
        for response in [file, stream, events] {
            XCTAssertEqual(response.status, .badRequest)
            XCTAssertTrue(text(response).contains(message), text(response))
        }
        let dramaSelected = try makeService(selected: .drama3, installed: [.drama3, .ming])
        let selected = await send("POST", "/v1/audio/speech", json: ["input": "Hi there.", "voice": "UK_George"], to: dramaSelected)
        XCTAssertEqual(selected.status, .badRequest)
        XCTAssertTrue(text(selected).contains(message), text(selected))
        XCTAssertEqual(log.requests.count, 0)
        let busy = await send("GET", "/v1/health")
        XCTAssertTrue(text(busy).contains("\"busy\":false"), "the refused request holds no job slot")
    }

    func testNamesNoEngineHasAndSpeakerVoicesAreRefused() async throws {
        let cases: [([String: Any], String)] = [
            (["voice": "ming:Nobody"], "Ming has no built-in voice \"Nobody\""),
            (["voice": "Nobody"], "No voice \"Nobody\" in the library or built into an engine"),
            (["speaker_voices": ["0": "US_Emma"]], "speaker_voices takes library voices, and \"US_Emma\" is a built-in Ming voice"),
            (["voice": "US_Emma", "character": ["name": "Captain", "description": "gravelly"]], "can't be combined"),
        ]
        for (json, message) in cases {
            let response = await speak(json)
            XCTAssertEqual(response.status, .badRequest, "\(json)")
            XCTAssertTrue(text(response).contains(message), text(response))
        }
        XCTAssertEqual(log.requests.count, 0)
    }

    /// Catches a retake that falls back to the default voice, and a take in
    /// one voice silently standing in for another.
    func testARetakeKeepsTheBuiltInVoiceAndAnotherVoiceCannotContinueIt() async throws {
        let first = await speak(["voice": "UK_George"])
        let id = try XCTUnwrap(first.headers["X-Drama-Take-Id"])
        let added = await send("POST", "/v1/voices", json: ["name": "UK_George", "transcript": "Hello there.", "audio_base64": Self.clip.base64EncodedString()])
        XCTAssertEqual(added.status, .created, text(added))

        let retake = await send("POST", "/v1/audio/speech", json: ["previous_takes": [["take_id": id]], "note": "happier"])
        XCTAssertEqual(retake.status, .ok, text(retake))
        XCTAssertEqual(log.requests.last?.builtInVoice, george, "the take's built-in voice, not the library voice added since")
        XCTAssertEqual(log.requests.last?.previousTakes.count, 1)

        let same = await speak(["voice": "ming:UK_George", "previous_takes": [["take_id": id]]])
        XCTAssertEqual(same.status, .ok, text(same))
        let other = await speak(["voice": "US_Emma", "previous_takes": [["take_id": id]]])
        XCTAssertEqual(other.status, .badRequest)
        XCTAssertTrue(text(other).contains("was not spoken in the built-in voice US_Emma"), text(other))
        XCTAssertEqual(log.requests.count, 3)
    }

    func testTheLiveRouteSpeaksEverySentenceInTheBuiltInVoice() async throws {
        let request = HTTPRequest(method: "POST", path: "/v1/audio/speech/live", query: ["voice": "UK_George"], headers: [:])
        let body = AsyncThrowingStream<Data, Error> { continuation in
            continuation.yield(Data("The light is lit. The fog rolled in.\n".utf8))
            continuation.finish()
        }
        let response = await service.respond(to: request, body: body)
        XCTAssertEqual(response.status, .ok, text(response))
        guard case .chunked(let stream) = response.body else { return XCTFail("a stream") }
        for try await _ in stream {}
        XCTAssertEqual(log.requests.map(\.builtInVoice), [george, george])
        XCTAssertEqual(log.requests.map(\.previousTakes.count), [0, 1])

        let refused = await service.respond(to: HTTPRequest(method: "POST", path: "/v1/audio/speech/live", query: ["voice": "UK_George", "model": "drama3"], headers: [:]), body: body)
        XCTAssertEqual(refused.status, .badRequest)
        XCTAssertTrue(text(refused).contains("a built-in Ming voice"), text(refused))
    }
}

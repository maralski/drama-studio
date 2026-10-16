import DramaEngine
import DramaServer
import XCTest

final class DictionaryFile: @unchecked Sendable {
    private let lock = NSLock()
    private var stored: PronunciationDictionary

    init(_ rules: [PronunciationRule]) { stored = PronunciationDictionary(rules: rules) }

    var dictionary: PronunciationDictionary {
        get { lock.withLock { stored } }
        set { lock.withLock { stored = newValue } }
    }
}

final class APIPronunciationTests: XCTestCase {
    private var directory: URL!
    private var engine: RecordingSynthesizer!
    private var file: DictionaryFile!
    private var service: APIService!

    override func setUp() async throws {
        directory = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        engine = RecordingSynthesizer()
        file = DictionaryFile([PronunciationRule(match: "Nguyen", sayAs: "Nguyen Win")])
        service = APIService(synthesizer: engine, voices: VoiceDirectory(url: directory), token: nil, version: "test",
                             pronunciation: { [file] in file!.dictionary })
    }

    override func tearDown() async throws {
        try? FileManager.default.removeItem(at: directory)
    }

    private func post(_ path: String, _ json: [String: Any]) async throws -> APIResponse {
        await service.respond(to: HTTPRequest(method: "POST", path: path, body: try JSONSerialization.data(withJSONObject: json)))
    }

    private func spoken() async -> [String] { await engine.requests.map(\.text) }

    private func text(_ response: APIResponse) -> String { String(decoding: response.bodyData ?? Data(), as: UTF8.self) }

    private func drain(_ response: APIResponse) async throws {
        guard case .chunked(let stream) = response.body else { return XCTFail("not chunked: \(text(response))") }
        for try await _ in stream {}
    }

    func testRulesRewriteTheInputButNotTheNoteOrEarlierTakes() async throws {
        let codes: [[Int32]] = (0..<10).map { book in (0..<3).map { Int32(book * 100 + $0) } }
        let response = try await post("/v1/audio/speech", ["input": "Ask Nguyen.", "note": "say Nguyen softly",
                                                           "previous_takes": [["text": "Nguyen spoke.", "codes_base64": TakeCodes.encode(codes)]]])
        XCTAssertEqual(response.status, .ok, text(response))
        let last = await engine.requests.last
        let request = try XCTUnwrap(last)
        XCTAssertEqual(request.text, "Ask Nguyen Win.")
        XCTAssertEqual(request.note, "say Nguyen softly")
        XCTAssertEqual(request.previousTakes.map(\.text), ["Nguyen spoke."])

        let tts = try await post("/v1/tts", ["text": "Nguyen again."])
        XCTAssertEqual(tts.status, .ok, text(tts))
        let ttsText = await spoken().last
        XCTAssertEqual(ttsText, "Nguyen Win again.")
    }

    func testTheTakeKeepsTheClientTextAndARetakeRewritesItOnce() async throws {
        let first = try await post("/v1/audio/speech", ["input": "Ask Nguyen."])
        let id = try XCTUnwrap(first.headers["X-Drama-Take-Id"])
        let take = await service.respond(to: HTTPRequest(method: "GET", path: "/v1/takes/\(id)"))
        let body = try XCTUnwrap(try JSONSerialization.jsonObject(with: take.bodyData ?? Data()) as? [String: Any])
        XCTAssertEqual(body["input"] as? String, "Ask Nguyen.", "input is what the client sent")
        XCTAssertEqual(body["text"] as? String, "Ask Nguyen Win.", "text is what the engine spoke")

        let retake = try await post("/v1/audio/speech", ["note": "slower", "previous_takes": [["take_id": id]]])
        XCTAssertEqual(retake.status, .ok, text(retake))
        let retakeText = await spoken().last
        XCTAssertEqual(retakeText, "Ask Nguyen Win.")
    }

    func testFalseLeavesTheTextAlone() async throws {
        _ = try await post("/v1/audio/speech", ["input": "Ask Nguyen.", "apply_pronunciation": false])
        _ = try await post("/v1/tts", ["text": "Ask Nguyen.", "apply_pronunciation": false])
        let texts = await spoken()
        XCTAssertEqual(texts, ["Ask Nguyen.", "Ask Nguyen."])
    }

    func testEachRequestReadsTheDictionaryAsItStands() async throws {
        _ = try await post("/v1/audio/speech", ["input": "Ask Nguyen."])
        file.dictionary = PronunciationDictionary(rules: [PronunciationRule(match: "Nguyen", sayAs: "Win")])
        _ = try await post("/v1/audio/speech", ["input": "Ask Nguyen."])
        let texts = await spoken()
        XCTAssertEqual(texts, ["Ask Nguyen Win.", "Ask Win."])
    }

    func testTheEventStreamQueryHonorsTheFlag() async throws {
        for (value, expected) in [(nil, "Ask Nguyen Win."), ("true", "Ask Nguyen Win."), ("0", "Ask Nguyen."), ("false", "Ask Nguyen.")] {
            var query = ["input": "Ask Nguyen."]
            query["apply_pronunciation"] = value
            let response = await service.respond(to: HTTPRequest(method: "GET", path: "/v1/audio/speech/stream", query: query))
            try await drain(response)
            let last = await spoken().last
            XCTAssertEqual(last, expected, "apply_pronunciation=\(value ?? "absent")")
        }
    }

    func testLiveRewritesEachSentenceUnlessTurnedOff() async throws {
        for (query, expected) in [([:], ["Ask Nguyen Win.", "Then Nguyen Win left."]),
                                  (["apply_pronunciation": "no"], ["Ask Nguyen.", "Then Nguyen left."])] {
            let before = await engine.requests.count
            let (input, feed) = AsyncThrowingStream<Data, Error>.makeStream()
            let response = await service.respond(to: HTTPRequest(method: "POST", path: "/v1/audio/speech/live", query: query), body: input)
            feed.yield(Data("Ask Nguyen.\nThen Nguyen left.\n".utf8))
            feed.finish()
            try await drain(response)
            let texts = await spoken().dropFirst(before)
            XCTAssertEqual(Array(texts), expected, "query \(query)")
        }
    }

    func testAMalformedValueIsABadRequestNamingTheField() async throws {
        let body = try await post("/v1/audio/speech", ["input": "Hi.", "apply_pronunciation": "maybe"])
        XCTAssertEqual(body.status, .badRequest)
        XCTAssertEqual(text(body), "apply_pronunciation has the wrong type.\n")

        let stream = await service.respond(to: HTTPRequest(method: "GET", path: "/v1/audio/speech/stream", query: ["input": "Hi.", "apply_pronunciation": "maybe"]))
        XCTAssertEqual(stream.status, .badRequest)
        XCTAssertEqual(text(stream), "apply_pronunciation must be true or false.\n")

        let live = await service.respond(to: HTTPRequest(method: "POST", path: "/v1/audio/speech/live", query: ["apply_pronunciation": "2"]),
                                         body: AsyncThrowingStream { $0.finish() })
        XCTAssertEqual(live.status, .badRequest)
        XCTAssertEqual(text(live), "apply_pronunciation must be true or false.\n")
        let reached = await engine.requests.count
        XCTAssertEqual(reached, 0)
    }
}

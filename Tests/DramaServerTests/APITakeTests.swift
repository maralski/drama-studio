import DramaEngine
import DramaServer
import XCTest

final class APITakeTests: XCTestCase {
    private var directory: URL!
    private var engine: RecordingSynthesizer!
    private var service: APIService!
    private let captain: [String: Any] = ["name": "Captain", "description": "gravelly old male voice", "scene": "a storm-battered tavern"]

    override func setUp() async throws {
        directory = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        engine = RecordingSynthesizer()
        service = APIService(synthesizer: engine, voices: VoiceDirectory(url: directory), token: nil, version: "test")
    }

    override func tearDown() async throws {
        try? FileManager.default.removeItem(at: directory)
    }

    private func send(_ method: String, _ path: String, json: Any? = nil) async -> APIResponse {
        let data = json.map { try! JSONSerialization.data(withJSONObject: $0) } ?? Data()
        return await service.respond(to: HTTPRequest(method: method, path: path, headers: [:], body: data))
    }

    private func lastRequest() async throws -> SpeechRequest {
        let last = await engine.requests.last
        return try XCTUnwrap(last)
    }

    private func text(_ response: APIResponse) -> String { String(decoding: response.bodyData ?? Data(), as: UTF8.self) }
    private func object(_ response: APIResponse) -> [String: Any] {
        (try? JSONSerialization.jsonObject(with: response.bodyData ?? Data())) as? [String: Any] ?? [:]
    }

    func testEveryGenerationIsATakeTheClientCanFetch() async throws {
        let response = await send("POST", "/v1/audio/speech", json: ["input": "Ye want to know?", "character": captain])
        XCTAssertEqual(response.status, .ok, text(response))
        let id = try XCTUnwrap(response.headers["X-Drama-Take-Id"])
        let request = await engine.requests.last
        XCTAssertEqual(request?.character, VoiceDesign(name: "Captain", description: "gravelly old male voice", scene: "a storm-battered tavern"))
        XCTAssertTrue(request?.references.isEmpty ?? false)

        let fetched = await send("GET", "/v1/takes/\(id)")
        XCTAssertEqual(fetched.status, .ok, text(fetched))
        let body = object(fetched)
        XCTAssertEqual(body["text"] as? String, "Captain: Ye want to know?")
        XCTAssertEqual(body["input"] as? String, "Ye want to know?")
        let codes = try XCTUnwrap(TakeCodes.decode(body["codes_base64"] as? String ?? ""))
        XCTAssertEqual(codes.count, 10)
        XCTAssertEqual(codes[0].count, Int(response.headers["X-Drama-Frames"] ?? ""))
        XCTAssertEqual(body["previous_take_ids"] as? [String], [])
    }

    func testRetakeByIdInheritsTheLineAndVoiceAndCarriesTheLineage() async throws {
        let first = await send("POST", "/v1/audio/speech", json: ["input": "[low] Ye want to know? Pray.", "character": captain])
        let firstID = try XCTUnwrap(first.headers["X-Drama-Take-Id"])
        let second = await send("POST", "/v1/audio/speech", json: ["note": "angrier", "previous_takes": [["take_id": firstID]]])
        XCTAssertEqual(second.status, .ok, text(second))
        var request = try await lastRequest()
        XCTAssertEqual(request.text, "[low] Ye want to know? Pray.")
        XCTAssertEqual(request.character?.name, "Captain")
        XCTAssertEqual(request.note, "angrier")
        XCTAssertEqual(request.previousTakes.map(\.text), ["Captain: [low] Ye want to know? Pray."])

        let secondID = try XCTUnwrap(second.headers["X-Drama-Take-Id"])
        let third = await send("POST", "/v1/audio/speech", json: ["note": "whispered", "previous_takes": [["take_id": secondID]]])
        XCTAssertEqual(third.status, .ok, text(third))
        request = try await lastRequest()
        XCTAssertEqual(request.previousTakes.map(\.text), ["Captain: [low] Ye want to know? Pray.",
                                                           "Captain: Direction: angrier\nLine: [angrier] Ye want to know? [angrier] Pray."],
                       "naming the latest take brings the line's first take, the voice, along")
        let thirdID = try XCTUnwrap(third.headers["X-Drama-Take-Id"])
        let thirdTake = await send("GET", "/v1/takes/\(thirdID)")
        XCTAssertEqual(object(thirdTake)["previous_take_ids"] as? [String], [firstID, secondID])

        let explicit = await send("POST", "/v1/audio/speech", json: ["input": "A new line.", "previous_takes": [["take_id": firstID], ["take_id": secondID]]])
        XCTAssertEqual(explicit.status, .ok, text(explicit))
        request = try await lastRequest()
        XCTAssertEqual(request.previousTakes.count, 2, "listing a take and its lineage does not repeat it")
        XCTAssertEqual(request.text, "A new line.")
    }

    func testInlinePreviousTakesDecodeTheirCodes() async throws {
        let codes: [[Int32]] = (0..<10).map { book in (0..<3).map { Int32(book * 100 + $0) } }
        let response = await send("POST", "/v1/audio/speech", json: ["input": "Say it again.", "note": "faster",
                                                                     "previous_takes": [["text": "Say it again.", "codes_base64": TakeCodes.encode(codes)]]])
        XCTAssertEqual(response.status, .ok, text(response))
        let request = try await lastRequest()
        XCTAssertEqual(request.previousTakes, [Take(text: "Say it again.", codes: codes)])
        XCTAssertEqual(request.note, "faster")
    }

    func testStreamedTakesAreCachedOnceFinished() async throws {
        let response = await send("POST", "/v1/audio/speech", json: ["input": "Streamed line.", "stream": true, "response_format": "pcm"])
        let id = try XCTUnwrap(response.headers["X-Drama-Take-Id"])
        guard case .chunked(let stream) = response.body else { return XCTFail("not chunked") }
        for try await _ in stream {}
        let fetched = await send("GET", "/v1/takes/\(id)")
        XCTAssertEqual(fetched.status, .ok, text(fetched))
    }

    func testTheCacheKeepsTheLastTwentyTakes() async throws {
        var ids: [String] = []
        for i in 0...TakeCache.capacity {
            let response = await send("POST", "/v1/audio/speech", json: ["input": "Line number \(i)."])
            ids.append(try XCTUnwrap(response.headers["X-Drama-Take-Id"]))
        }
        let evicted = await send("GET", "/v1/takes/\(ids[0])")
        XCTAssertEqual(evicted.status, .notFound)
        let kept = await send("GET", "/v1/takes/\(ids[1])")
        XCTAssertEqual(kept.status, .ok)
        let retake = await send("POST", "/v1/audio/speech", json: ["note": "softer", "previous_takes": [["take_id": ids[0]]]])
        XCTAssertEqual(retake.status, .notFound, text(retake))
    }

    func testBadDirectingRequests() async {
        for (body, fragment) in [
            (["input": "Hi.", "character": captain, "voice": "Narrator"] as [String: Any], "can't be combined"),
            (["input": "Hi.", "character": ["name": "Captain"]], "name and a description"),
            (["note": "angrier"], "input is required"),
            (["input": "Hi.", "previous_takes": [["take_id": "nope"]]], "not a take id"),
            (["input": "Hi.", "previous_takes": [["text": "Hi.", "codes_base64": "AAEC"]]], "codes_base64"),
            (["input": "Hi.", "previous_takes": [["text": "Hi."]]], "either take_id"),
            (["input": "[laughs]", "note": "angrier"], "line"),
        ] {
            let response = await send("POST", "/v1/audio/speech", json: body)
            XCTAssertEqual(response.status, .badRequest, "\(body): \(text(response))")
            XCTAssertTrue(text(response).contains(fragment), "\(body): \(text(response))")
        }
        let response = await send("GET", "/v1/takes/\(UUID().uuidString)")
        XCTAssertEqual(response.status, .notFound)
    }
}

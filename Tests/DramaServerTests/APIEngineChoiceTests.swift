// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import DramaEngine
import DramaServer
import XCTest

/// The API lists every installed engine, runs a request on the engine its
/// `model` names, and says which engine made each take.
final class APIEngineChoiceTests: XCTestCase {
    private var directory: URL!
    private var engines: SpeechEngines!
    private var service: APIService!

    override func setUp() async throws {
        directory = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        engines = try SpeechEngines(selected: .ming, factories: [.drama3: .mock(.drama3, pace: .instant), .ming: .mock(.ming, pace: .instant)],
                                    voices: VoiceDirectory(url: directory))
        service = APIService(synthesizer: engines, voices: VoiceDirectory(url: directory), token: nil, version: "test", pronunciation: { PronunciationDictionary() })
    }

    override func tearDown() async throws {
        try? FileManager.default.removeItem(at: directory)
    }

    private func send(_ method: String, _ path: String, json: Any? = nil) async -> APIResponse {
        let body = json.map { try! JSONSerialization.data(withJSONObject: $0) } ?? Data()
        return await service.respond(to: HTTPRequest(method: method, path: path, headers: [:], body: body))
    }

    private func object(_ response: APIResponse) -> [String: Any] {
        (try? JSONSerialization.jsonObject(with: response.bodyData ?? Data())) as? [String: Any] ?? [:]
    }

    private func text(_ response: APIResponse) -> String { String(decoding: response.bodyData ?? Data(), as: UTF8.self) }

    func testModelsListsEachInstalledEngineAndMarksTheSelectedOne() async throws {
        let models = await send("GET", "/v1/models")
        let data = try XCTUnwrap(object(models)["data"] as? [[String: Any]])
        XCTAssertEqual(data.map { $0["id"] as? String }, ["fishaudio/drama-3", "inclusionAI/Ming-3.0-tts-0.5B"])
        XCTAssertEqual(data.map { $0["engine"] as? String }, ["drama3", "ming"])
        XCTAssertEqual(data.map { $0["selected"] as? Bool }, [false, true])
        XCTAssertEqual(data.map { $0["owned_by"] as? String }, ["fishaudio", "inclusionAI"])
        let health = await send("GET", "/v1/health")
        XCTAssertEqual(object(health)["speech_engine"] as? String, "ming")
    }

    func testARequestRunsOnTheEngineItsModelNamesAndLeavesTheSelectionAlone() async throws {
        for (model, engine) in [("drama3", "drama3"), ("fishaudio/drama-3", "drama3"), ("Ming", "ming"), ("tts-1", "ming"), ("TTS-1-HD", "ming"), ("gpt-4o-mini-tts", "ming")] {
            let response = await send("POST", "/v1/audio/speech", json: ["input": "The light is lit.", "model": model])
            XCTAssertEqual(response.status, .ok, text(response))
            XCTAssertEqual(response.headers["X-Drama-Engine"], engine, model)
            let take = await send("GET", "/v1/takes/\(response.headers["X-Drama-Take-Id"] ?? "")")
            XCTAssertEqual(object(take)["engine"] as? String, engine, model)
        }
        XCTAssertEqual(engines.selected, .ming)
        let health = await send("GET", "/v1/health")
        XCTAssertEqual(object(health)["speech_engine"] as? String, "ming")
    }

    func testTheStreamNamesItsEngineInTheHeadersAndTheDoneEvent() async throws {
        let response = await service.respond(to: HTTPRequest(method: "GET", path: "/v1/audio/speech/stream", query: ["input": "Hi there.", "model": "drama3"]))
        XCTAssertEqual(response.headers["X-Drama-Engine"], "drama3")
        XCTAssertEqual(response.headers["X-Drama-Model"], "fishaudio/drama-3")
        guard case .chunked(let body) = response.body else { return XCTFail("not streamed") }
        var data = Data()
        for try await chunk in body { data += chunk }
        let done = try XCTUnwrap(String(decoding: data, as: UTF8.self).components(separatedBy: "event: done\ndata: ").last?.components(separatedBy: "\n").first)
        let event = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(done.utf8)) as? [String: Any])
        XCTAssertEqual(event["engine"] as? String, "drama3")
    }

    func testAModelThatIsNeitherAnEngineNorAnOpenAIModelIsRefusedWithTheInstalledEngines() async throws {
        for model in ["ming-3.0", "drama-3", ""] {
            let response = await send("POST", "/v1/audio/speech", json: ["input": "Hello.", "model": model])
            XCTAssertEqual(response.status, .badRequest, model)
            XCTAssertTrue(text(response).contains("Installed: drama3, ming"), text(response))
            let stream = await service.respond(to: HTTPRequest(method: "GET", path: "/v1/audio/speech/stream", query: ["input": "Hi.", "model": model]))
            XCTAssertEqual(stream.status, .badRequest, model)
        }
    }

    func testNamingAnEngineThatIsNotInstalledIsRefusedWithTheInstalledOnes() async throws {
        service = APIService(synthesizer: try SpeechEngines(selected: .ming, factories: [.ming: .mock(.ming, pace: .instant)]),
                             voices: VoiceDirectory(url: directory), token: nil, version: "test")
        let response = await send("POST", "/v1/audio/speech", json: ["input": "Hello.", "model": "drama3"])
        XCTAssertEqual(response.status, .badRequest)
        XCTAssertTrue(text(response).contains("Drama 3 isn't installed"), text(response))
        XCTAssertTrue(text(response).contains("Installed: ming"), text(response))
    }

    func testSwitchingTheSelectedEngineChangesWhatHealthAndRequestsReport() async throws {
        try engines.select(.drama3)
        let health = await send("GET", "/v1/health")
        XCTAssertEqual(object(health)["speech_engine"] as? String, "drama3")
        let response = await send("POST", "/v1/audio/speech", json: ["input": "Hello."])
        XCTAssertEqual(response.headers["X-Drama-Engine"], "drama3")
    }

    func testARetakeStaysOnTheEngineThatMadeTheTakeAfterTheSelectionChanges() async throws {
        let first = await send("POST", "/v1/audio/speech", json: ["input": "The light is lit."])
        XCTAssertEqual(first.headers["X-Drama-Engine"], "ming")
        try engines.select(.drama3)
        let retake = await send("POST", "/v1/audio/speech", json: ["note": "softer", "previous_takes": [["take_id": first.headers["X-Drama-Take-Id"] ?? ""]]])
        XCTAssertEqual(retake.status, .ok, text(retake))
        XCTAssertEqual(retake.headers["X-Drama-Engine"], "ming")
        let fresh = await send("POST", "/v1/audio/speech", json: ["input": "A new line."])
        XCTAssertEqual(fresh.headers["X-Drama-Engine"], "drama3")
    }
}

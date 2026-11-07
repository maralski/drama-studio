// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import DramaEngine
import DramaServer
import XCTest

/// The API on a Ming engine: no Drama frames or codes for a Ming take, and sample counts that match the audio.
final class APIMingTakeTests: XCTestCase {
    private var directory: URL!
    private var service: APIService!

    override func setUp() async throws {
        directory = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        service = APIService(synthesizer: MockSynthesizer(engine: .ming, pace: .instant), voices: VoiceDirectory(url: directory), token: nil, version: "test",
                             pronunciation: { PronunciationDictionary() })
    }

    override func tearDown() async throws {
        try? FileManager.default.removeItem(at: directory)
    }

    private func send(_ method: String, _ path: String, json: Any? = nil) async -> APIResponse {
        let data = json.map { try! JSONSerialization.data(withJSONObject: $0) } ?? Data()
        return await service.respond(to: HTTPRequest(method: method, path: path, headers: [:], body: data))
    }

    private func object(_ response: APIResponse) -> [String: Any] {
        (try? JSONSerialization.jsonObject(with: response.bodyData ?? Data())) as? [String: Any] ?? [:]
    }

    func testAMingTakeHasSamplesButNoDramaFramesOrCodes() async throws {
        let response = await send("POST", "/v1/audio/speech", json: ["input": "One sentence to say. Then another one.", "response_format": "pcm"])
        XCTAssertEqual(response.status, .ok)
        XCTAssertNil(response.headers["X-Drama-Frames"])
        XCTAssertNil(response.headers["X-Drama-Generated-Frames"])
        let samples = try XCTUnwrap(Int(response.headers["X-Drama-Samples"] ?? ""))
        XCTAssertEqual(response.bodyData?.count, samples * 2, "the PCM body holds exactly the take's samples")
        XCTAssertEqual(Double(response.headers["X-Drama-Seconds"] ?? "") ?? 0, Double(samples) / 44_100, accuracy: 0.001)

        let id = try XCTUnwrap(response.headers["X-Drama-Take-Id"])
        let fetched = object(await send("GET", "/v1/takes/\(id)"))
        XCTAssertEqual(fetched["samples"] as? Int, samples)
        XCTAssertNil(fetched["frames"])
        XCTAssertNil(fetched["codes_base64"])
        XCTAssertEqual(fetched["seconds"] as? Double, (Double(samples) / 44_100 * 1000).rounded() / 1000)
    }

    func testADramaTakeStillCarriesFramesAndCodes() async throws {
        let drama = APIService(synthesizer: MockSynthesizer(pace: .instant), voices: VoiceDirectory(url: directory), token: nil, version: "test",
                               pronunciation: { PronunciationDictionary() })
        let response = await drama.respond(to: HTTPRequest(method: "POST", path: "/v1/audio/speech", headers: [:],
                                                            body: try JSONSerialization.data(withJSONObject: ["input": "One sentence."])))
        let frames = try XCTUnwrap(Int(response.headers["X-Drama-Frames"] ?? ""))
        XCTAssertEqual(Int(response.headers["X-Drama-Samples"] ?? ""), frames * 2048)
        XCTAssertNotNil(response.headers["X-Drama-Generated-Frames"])
    }

    func testARetakeOfAMingTakeByIdRuns() async throws {
        let first = await send("POST", "/v1/audio/speech", json: ["input": "Ye want to know?"])
        let id = try XCTUnwrap(first.headers["X-Drama-Take-Id"])
        let second = await send("POST", "/v1/audio/speech", json: ["note": "angrier", "previous_takes": [["take_id": id]]])
        XCTAssertEqual(second.status, .ok, String(decoding: second.bodyData ?? Data(), as: UTF8.self))
        XCTAssertNil(second.headers["X-Drama-Frames"])
    }
}

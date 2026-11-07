// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import DramaEngine
import DramaServer
import XCTest

/// Refuses every request the way an engine does once it runs.
private struct RefusingSynthesizer: SpeechSynthesizer {
    var error: any Error
    var isLoaded: Bool { true }
    func load(onPhase: @escaping @Sendable (String) -> Void) async throws {}
    var engineDescription: EngineDescription { EngineDescription(engine: .drama3, modelID: "fishaudio/drama-3", snapshot: "test", isMock: true) }
    var isTuned: Bool? { false }
    func encodeVoice(samples: [Float], transcript: String) async throws -> VoiceEncoding { throw error }
    func synthesize(_ request: SpeechRequest) -> AsyncThrowingStream<SpeechEvent, Error> {
        AsyncThrowingStream { $0.finish(throwing: error) }
    }
}

/// The API behind the engine registry: a voice encoded for one engine speaks
/// on the other, and what an engine cannot perform reaches the client.
final class APIEngineNeutralTests: XCTestCase {
    private var directory: URL!
    private var engines: SpeechEngines!
    private var service: APIService!
    private static let clip = AudioEncoding.wav((0..<44_100).map { Float(sin(Double($0) * 0.05) * 0.3) })

    override func setUp() async throws {
        directory = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        let voices = VoiceDirectory(url: directory)
        engines = try SpeechEngines(selected: .ming, factories: [.drama3: .mock(.drama3, pace: .instant), .ming: .mock(.ming, pace: .instant)],
                                    voices: voices)
        service = APIService(synthesizer: engines, voices: voices, token: nil, version: "test", pronunciation: { PronunciationDictionary() })
    }

    override func tearDown() async throws {
        try? FileManager.default.removeItem(at: directory)
    }

    private func send(_ method: String, _ path: String, json: Any) async -> APIResponse {
        await service.respond(to: HTTPRequest(method: method, path: path, headers: [:], body: try! JSONSerialization.data(withJSONObject: json)))
    }

    private func text(_ response: APIResponse) -> String { String(decoding: response.bodyData ?? Data(), as: UTF8.self) }

    private func streamed(_ response: APIResponse) async throws -> String {
        guard case .chunked(let body) = response.body else { return text(response) }
        var data = Data()
        for try await chunk in body { data += chunk }
        return String(decoding: data, as: UTF8.self)
    }

    private func cloneKeeper() async throws -> Set<SpeechEngineID> {
        let added = await send("POST", "/v1/voices", json: ["name": "Keeper", "transcript": "Hello there.", "audio_base64": Self.clip.base64EncodedString()])
        XCTAssertEqual(added.status, .created, text(added))
        return try await keeperEncodings()
    }

    private func keeperEncodings() async throws -> Set<SpeechEngineID> {
        let voices = await VoiceDirectory(url: directory).allVoices()
        let keeper = try XCTUnwrap(voices.first { $0.name == "Keeper" })
        return Set(keeper.encodings.keys)
    }

    func testAVoiceClonedOnMingSpeaksOnMing() async throws {
        let encodings = try await cloneKeeper()
        XCTAssertEqual(encodings, [.ming], "cloned on Ming, with no Drama 3 codes")
        let file = await send("POST", "/v1/audio/speech", json: ["input": "The light is lit.", "voice": "Keeper"])
        XCTAssertEqual(file.status, .ok, text(file))
        let events = await service.respond(to: HTTPRequest(method: "GET", path: "/v1/audio/speech/stream", query: ["input": "The light is lit.", "voice": "Keeper"]))
        XCTAssertEqual(events.status, .ok, text(events))
        let body = try await streamed(events)
        XCTAssertTrue(body.contains("event: done"), body)
    }

    func testInlineReferencesSpeakOnMing() async throws {
        let response = await send("POST", "/v1/audio/speech", json: ["input": "The light is lit.",
                                                                     "references": [["audio_base64": Self.clip.base64EncodedString(), "text": "Hello there."]]])
        XCTAssertEqual(response.status, .ok, text(response))
    }

    func testAVoiceClonedOnMingSpeaksOnDrama3AndKeepsItsCodes() async throws {
        _ = try await cloneKeeper()
        try engines.select(.drama3)
        let response = await send("POST", "/v1/audio/speech", json: ["input": "The light is lit.", "voice": "Keeper"])
        XCTAssertEqual(response.status, .ok, text(response))
        XCTAssertNotNil(response.headers["X-Drama-Frames"], "a Drama 3 take")
        let encodings = try await keeperEncodings()
        XCTAssertEqual(encodings, [.drama3, .ming], "the registry encoded the clip for Drama 3 and kept it")
    }

    func testTheLiveRouteSpeaksAVoiceClonedOnMing() async throws {
        _ = try await cloneKeeper()
        let body = AsyncThrowingStream<Data, Error> { $0.yield(Data("The light is lit.\n".utf8)); $0.finish() }
        let response = await service.respond(to: HTTPRequest(method: "POST", path: "/v1/audio/speech/live", query: ["voice": "Keeper"], headers: ["Accept": "text/event-stream"]),
                                             body: body)
        XCTAssertEqual(response.status, .ok, text(response))
        let events = try await streamed(response)
        XCTAssertTrue(events.contains("event: done"), events)
        XCTAssertFalse(events.contains("event: error"), events)
    }

    func testARequestTheEngineRefusesIsTheClientsMistake() async throws {
        for error: any Error in [PromptError.structuralToken("text", "<|im_start|>"), EngineError.noClip(voiceName: "Keeper")] {
            let refusing = APIService(synthesizer: RefusingSynthesizer(error: error), voices: VoiceDirectory(url: directory), token: nil, version: "test",
                                      pronunciation: { PronunciationDictionary() })
            let response = await refusing.respond(to: HTTPRequest(method: "POST", path: "/v1/audio/speech",
                                                                  body: try JSONSerialization.data(withJSONObject: ["input": "Hello."])))
            XCTAssertEqual(response.status, .badRequest, text(response))
            XCTAssertEqual(text(response), "\(error)\n")
        }
    }
}

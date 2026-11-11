// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import DramaEngine
import DramaServer
import Synchronization
import XCTest
@testable import DramaStudio

@MainActor
final class AppAPIServerTests: XCTestCase {
    private var directory: URL!

    override func setUp() async throws {
        directory = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
    }

    override func tearDown() async throws {
        try? FileManager.default.removeItem(at: directory)
    }

    func testVoicesAddedOverHTTPShowInTheWindowsLibrary() async throws {
        let session = SessionModel(synthesizer: MockSynthesizer(pace: .instant), library: VoiceLibrary(directory: directory))
        let service = APIService(synthesizer: session.synthesizer, voices: session.library, token: nil, version: "test")
        let clip = AudioEncoding.wav(Array(repeating: 0.1, count: 22_050))
        let body = try JSONSerialization.data(withJSONObject: ["name": "Ann", "transcript": "Hi there.", "audio_base64": clip.base64EncodedString()])
        let added = await service.respond(to: HTTPRequest(method: "POST", path: "/v1/voices", body: body))
        XCTAssertEqual(added.status, .created)
        XCTAssertEqual(session.library.voices.map(\.name), ["Ann"])

        let id = try XCTUnwrap(session.library.voices.first?.id.uuidString)
        let deleted = await service.respond(to: HTTPRequest(method: "DELETE", path: "/v1/voices/\(id)"))
        XCTAssertEqual(deleted.status, .ok)
        XCTAssertTrue(session.library.voices.isEmpty)
    }

    func testAGenerationInTheWindowTurnsAPIRequestsAway() async throws {
        let session = SessionModel(synthesizer: MockSynthesizer(pace: .init(loadSeconds: 0, framesPerSecond: 10)), library: VoiceLibrary(directory: directory))
        session.playsWhileGenerating = false
        let service = APIServerController(session: session).makeService()
        session.script = "A line long enough to keep the engine busy for a moment."
        session.generate()
        XCTAssertTrue(session.state.isGenerating)
        let response = await service.respond(to: HTTPRequest(method: "POST", path: "/v1/tts", body: Data(#"{"text":"Me too."}"#.utf8)))
        XCTAssertEqual(response.status, .tooManyRequests)
        session.stopGeneration()
    }

    /// The test holds the main thread the way a modal alert shown from a task does.
    func testHealthAnswersTruthfullyWhileTheMainThreadIsHeld() throws {
        let session = SessionModel(synthesizer: MockSynthesizer(pace: .init(loadSeconds: 0, framesPerSecond: 10)), library: VoiceLibrary(directory: directory))
        session.playsWhileGenerating = false
        let service = APIServerController(session: session).makeService()
        XCTAssertEqual(try busy(accordingTo: service), false)
        session.script = "A line long enough to keep the engine busy for a moment."
        session.generate()
        XCTAssertEqual(try busy(accordingTo: service), true, "health reports the window's generation, not an idle guess")
        session.stopGeneration()
        XCTAssertEqual(try busy(accordingTo: service), false)
    }

    /// `/v1/health`'s `busy`, asked off the main thread while this thread waits for the answer.
    private func busy(accordingTo service: APIService) throws -> Bool? {
        let answer = Mutex<Data?>(nil)
        let answered = DispatchSemaphore(value: 0)
        Task.detached {
            let response = await service.respond(to: HTTPRequest(method: "GET", path: "/v1/health"))
            if case .data(let body) = response.body { answer.withLock { $0 = body } }
            answered.signal()
        }
        guard answered.wait(timeout: .now() + 5) == .success else {
            XCTFail("GET /v1/health did not answer while the main thread was held")
            return nil
        }
        let body = try XCTUnwrap(answer.withLock { $0 })
        return try XCTUnwrap(JSONSerialization.jsonObject(with: body) as? [String: Any])["busy"] as? Bool
    }

    func testTheAPIChipTooltipShowsThePortUngrouped() {
        XCTAssertEqual(APIChip.tooltip(port: 8841, job: nil), "API server listening on http://127.0.0.1:8841")
    }

    func testMenuBarStatusLines() {
        let job = SpeechJob(id: UUID(), progress: .generating(chunk: 2, of: 5))
        XCTAssertEqual(MenuBarStatus.job(session: .idle, job: job), "API request: Generating chunk 2 of 5")
        XCTAssertEqual(MenuBarStatus.job(session: .generating(chunk: 1, of: 3, playing: true), job: nil), "Generating chunk 1 of 3")
        XCTAssertEqual(MenuBarStatus.job(session: .ready, job: nil), "No generation running")
        XCTAssertEqual(MenuBarStatus.job(session: .loading(progress: 0, detail: "Tuning kernels for this Mac, about a minute"), job: nil),
                       "Loading model: Tuning kernels for this Mac, about a minute", "the load phase shows in the menu bar")
        let drama = EngineDescription(engine: .drama3, modelID: "fishaudio/drama-3", snapshot: "e0a9218", isMock: false)
        let ming = EngineDescription(engine: .ming, modelID: "inclusionAI/Ming-3.0-tts-0.5B", snapshot: "7247f4c", isMock: false)
        XCTAssertEqual(MenuBarStatus.model(drama, loaded: false, session: .idle, job: nil), "Drama 3 on Metal: not loaded yet")
        XCTAssertEqual(MenuBarStatus.model(ming, loaded: true, session: .idle, job: nil), "Ming on Metal: loaded", "the status names the selected engine")
        XCTAssertEqual(MenuBarStatus.model(MockSynthesizer().engineDescription, loaded: true, session: .ready, job: nil), "Preview engine: loaded")
        XCTAssertEqual(MenuBarStatus.model(drama, loaded: false, session: .idle, job: SpeechJob(id: UUID(), progress: .loadingModel("Mapping weights"))), "Drama 3 on Metal: loading, Mapping weights")
    }

    func testSessionLearnsTheModelLoadedFromTheEngine() async {
        let session = SessionModel(synthesizer: MockSynthesizer(pace: .instant), library: VoiceLibrary(directory: directory))
        session.playsWhileGenerating = false
        XCTAssertFalse(session.modelLoaded)
        session.script = "Load it."
        session.generate()
        let deadline = Date().addingTimeInterval(5)
        while !session.modelLoaded, Date() < deadline { try? await Task.sleep(for: .milliseconds(10)) }
        XCTAssertTrue(session.modelLoaded)
    }
}

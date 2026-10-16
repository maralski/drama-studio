import DramaEngine
import DramaServer
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
        let service = APIService(synthesizer: session.synthesizer, voices: session.library, token: nil, version: "test", engineBusyElsewhere: { await session.state.isGenerating })
        session.script = "A line long enough to keep the engine busy for a moment."
        session.generate()
        XCTAssertTrue(session.state.isGenerating)
        let response = await service.respond(to: HTTPRequest(method: "POST", path: "/v1/tts", body: Data(#"{"text":"Me too."}"#.utf8)))
        XCTAssertEqual(response.status, .tooManyRequests)
        session.stopGeneration()
    }

    func testMenuBarStatusLines() {
        let job = SpeechJob(id: UUID(), progress: .generating(chunk: 2, of: 5))
        XCTAssertEqual(MenuBarStatus.job(session: .idle, job: job), "API request: Generating chunk 2 of 5")
        XCTAssertEqual(MenuBarStatus.job(session: .generating(chunk: 1, of: 3, playing: true), job: nil), "Generating chunk 1 of 3")
        XCTAssertEqual(MenuBarStatus.job(session: .ready, job: nil), "No generation running")
        XCTAssertEqual(MenuBarStatus.job(session: .loading(progress: 0, detail: "Quantizing weights for the balanced mode (first load only)"), job: nil),
                       "Loading model: Quantizing weights for the balanced mode (first load only)", "the load phase shows in the menu bar")
        XCTAssertEqual(MenuBarStatus.model(preview: false, loaded: false, session: .idle, job: nil), "Drama 3 on Metal: not loaded yet")
        XCTAssertEqual(MenuBarStatus.model(preview: true, loaded: true, session: .ready, job: nil), "Preview engine: loaded")
        XCTAssertEqual(MenuBarStatus.model(preview: false, loaded: false, session: .idle, job: SpeechJob(id: UUID(), progress: .loadingModel("Mapping weights"))), "Drama 3 on Metal: loading, Mapping weights")
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

import DramaEngine
import XCTest
@testable import DramaStudio

/// The mock engine, keeping every request it was sent.
final class RecordingSynthesizer: SpeechSynthesizer, @unchecked Sendable {
    private let mock = MockSynthesizer(pace: .instant)
    private let lock = NSLock()
    private var sent: [SpeechRequest] = []

    var requests: [SpeechRequest] { lock.withLock { sent } }

    func load(onPhase: @escaping @Sendable (String) -> Void) async throws { try await mock.load(onPhase: onPhase) }
    var isLoaded: Bool { get async { await mock.isLoaded } }
    func encodeReference(audioURL: URL, transcript: String, name: String) async throws -> ReferenceVoice {
        try await mock.encodeReference(audioURL: audioURL, transcript: transcript, name: name)
    }
    func decode(frames: [Frame]) async throws -> [Float] { try await mock.decode(frames: frames) }
    func synthesize(_ request: SpeechRequest) -> AsyncThrowingStream<SpeechEvent, Error> {
        lock.withLock { sent.append(request) }
        return mock.synthesize(request)
    }
}

@MainActor
final class AppDirectingTests: XCTestCase {
    private var directory: URL!
    private var synthesizer: RecordingSynthesizer!
    private let design = VoiceDesign(name: "Nana", description: "warm elderly woman", scene: "a quiet bedroom")

    override func setUp() async throws {
        directory = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        synthesizer = RecordingSynthesizer()
    }

    override func tearDown() async throws {
        try? FileManager.default.removeItem(at: directory)
    }

    private func makeSession(script: String = "[sighs] Once there was a lighthouse.") -> SessionModel {
        let session = SessionModel(synthesizer: synthesizer, library: VoiceLibrary(directory: directory))
        session.playsWhileGenerating = false
        session.script = script
        return session
    }

    private func finish(_ action: () -> Void, in session: SessionModel) async {
        let before = synthesizer.requests.count
        action()
        XCTAssertEqual(synthesizer.requests.count, before + 1, "the action sent a request")
        let deadline = Date().addingTimeInterval(10)
        while session.state.isGenerating, Date() < deadline { try? await Task.sleep(for: .milliseconds(10)) }
    }

    func testSurpriseVoiceIsReusedUntilItIsPickedAgain() async {
        let session = makeSession()
        session.setVoice(.surprise, forSpeaker: 0)
        await finish(session.generate, in: session)
        let first = synthesizer.requests[0]
        XCTAssertEqual(first.previousTakes, [])
        XCTAssertNil(first.character)
        let surprise = session.result?.take
        XCTAssertNotNil(surprise)
        XCTAssertEqual(session.sessionVoices[.surprise], surprise)

        await finish(session.generate, in: session)
        XCTAssertEqual(synthesizer.requests[1].previousTakes, [surprise!], "the second take reuses the first take's voice")
        XCTAssertEqual(session.sessionVoices[.surprise], surprise, "a later take does not replace the session voice")

        session.setVoice(.modelDefault, forSpeaker: 0)
        session.setVoice(.surprise, forSpeaker: 0)
        XCTAssertNil(session.sessionVoices[.surprise])
        await finish(session.generate, in: session)
        XCTAssertEqual(synthesizer.requests[2].previousTakes, [], "picking Surprise me again gives a fresh surprise")
    }

    func testDesignedVoiceSendsTheCharacterAndThenItsFirstTake() async {
        let session = makeSession()
        session.setVoice(.designed(design), forSpeaker: 0)
        await finish(session.generate, in: session)
        XCTAssertEqual(synthesizer.requests[0].character, design)
        XCTAssertEqual(synthesizer.requests[0].references.count, 0)
        let voice = session.result!.take
        XCTAssertTrue(voice.text.hasPrefix("Nana: "), voice.text)

        await finish(session.generate, in: session)
        XCTAssertEqual(synthesizer.requests[1].character, design)
        XCTAssertEqual(synthesizer.requests[1].previousTakes, [voice])
    }

    func testKeepThisVoiceAddsTheSpokenTranscriptToTheLibrary() async throws {
        let session = makeSession()
        let speaker = try session.prepared.get().speakers[0]
        session.setVoice(.designed(design), for: speaker)
        XCTAssertFalse(session.canKeepVoice(for: speaker), "nothing to keep before the first take")
        await finish(session.generate, in: session)
        let take = try XCTUnwrap(session.result?.take)
        XCTAssertTrue(session.canKeepVoice(for: speaker))

        session.keepVoice(for: speaker)
        let kept = try XCTUnwrap(session.library.voices.first)
        XCTAssertEqual(kept.name, "Nana")
        XCTAssertEqual(kept.transcript, "Once there was a lighthouse.")
        XCTAssertEqual(kept.codes, take.codes)
        XCTAssertEqual(session.voice(for: speaker), .library(kept.id))
        XCTAssertEqual(session.casting.references, [kept])
        XCTAssertNil(session.casting.character)
        XCTAssertTrue(session.sessionVoices.isEmpty)
    }

    func testKeptSurpriseIsNamedSurpriseVoice() async throws {
        let session = makeSession(script: "Hello there.")
        let speaker = try session.prepared.get().speakers[0]
        session.setVoice(.surprise, for: speaker)
        await finish(session.generate, in: session)
        session.keepVoice(for: speaker)
        XCTAssertEqual(session.library.voices.map(\.name), ["Surprise voice"])
        XCTAssertEqual(session.library.voices.first?.transcript, "Hello there.")
    }
}

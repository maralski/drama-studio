import AVFoundation
import DramaEngine
import XCTest
@testable import DramaStudio

@MainActor
final class AppSessionTests: XCTestCase {
    private var directory: URL!

    override func setUp() async throws {
        directory = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
    }

    override func tearDown() async throws {
        try? FileManager.default.removeItem(at: directory)
    }

    private func makeSession(pace: MockSynthesizer.Pace = .instant) -> SessionModel {
        let session = SessionModel(synthesizer: MockSynthesizer(pace: pace), library: VoiceLibrary(directory: directory))
        session.playsWhileGenerating = false
        return session
    }

    private func waitUntil(_ condition: @escaping () -> Bool, timeout: Double = 10) async {
        let deadline = Date().addingTimeInterval(timeout)
        while !condition(), Date() < deadline { try? await Task.sleep(for: .milliseconds(10)) }
    }

    func testGenerationStreamsEveryChunkThenSettlesReady() async {
        let session = makeSession()
        session.script = String(repeating: "This sentence is about forty bytes long. ", count: 3)
        var states: [PlaybackState] = []
        session.generate()
        await waitUntil {
            if states.last != session.state { states.append(session.state) }
            return !session.state.isGenerating
        }
        XCTAssertEqual(session.state, .ready)
        XCTAssertTrue(states.contains { if case .generating(_, let of, _) = $0 { return of == 3 } else { return false } }, "\(states)")
        let result = try? XCTUnwrap(session.result)
        XCTAssertEqual(session.audio.count, result?.samples.count)
        XCTAssertEqual(session.audio.count, (result?.frames.count ?? 0) * ModelConstants.samplesPerFrame)
        XCTAssertTrue(session.canExport)
    }

    func testRequestFollowsTheControlsAndNewTakeNeedsATake() async {
        let session = makeSession()
        session.script = "One line to say."
        XCTAssertEqual(session.config, GenerationConfig())
        session.quickPreview = true
        session.expressiveness = 1
        XCTAssertEqual(session.config, .make(expressiveness: 1, quickPreview: true))

        XCTAssertFalse(session.canTakeAgain, "nothing to take again yet")
        session.generate()
        await waitUntil { !session.state.isGenerating }
        XCTAssertTrue(session.canTakeAgain)
    }

    func testStopKeepsPartialAudioAndDropsLateEvents() async {
        let session = makeSession(pace: MockSynthesizer.Pace(loadSeconds: 0, framesPerSecond: 2000))
        session.script = String(repeating: "A sentence that will be spoken by the mock. ", count: 30)
        session.generate()
        await waitUntil { !session.audio.isEmpty }
        session.stopGeneration()
        let partial = session.audio.count
        XCTAssertEqual(session.state, .ready)
        XCTAssertNil(session.result)
        try? await Task.sleep(for: .milliseconds(300))
        XCTAssertEqual(session.audio.count, partial, "no audio after stop")
        XCTAssertEqual(session.state, .ready)
    }

    func testRegeneratingRightAfterStopIgnoresTheCancelledRun() async {
        let session = makeSession(pace: MockSynthesizer.Pace(loadSeconds: 0, framesPerSecond: 4000))
        session.script = String(repeating: "Another sentence for the mock to speak. ", count: 12)
        session.generate()
        await waitUntil { !session.audio.isEmpty }
        session.stopGeneration()
        session.generate()
        var states: [PlaybackState] = []
        await waitUntil {
            if states.last != session.state { states.append(session.state) }
            return session.result != nil
        }
        XCTAssertFalse(states.contains { if case .failed = $0 { true } else { false } }, "\(states)")
        XCTAssertEqual(states.first { !$0.isGenerating }, .ready, "\(states)")
        XCTAssertEqual(session.audio.count, session.result?.samples.count)
    }

    func testEmptyScriptCannotGenerate() {
        let session = makeSession()
        session.script = "   "
        XCTAssertFalse(session.canGenerate)
        session.generate()
        XCTAssertEqual(session.state, .idle)
    }

    func testScreenplayRequestCarriesAssignedVoicesInSpeakerOrder() async throws {
        let session = makeSession()
        let clip = directory.appending(path: "clip.wav")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try AudioExporter.write([Float](repeating: 0.1, count: 44_100), to: clip, as: .wav16)
        try await session.encodeVoice(clip: clip, transcript: "Hello.", name: "Ann")
        let ann = try XCTUnwrap(session.library.voices.first)
        XCTAssertEqual(ann.frameCount, 22)
        XCTAssertEqual(ann.durationSeconds, 22 * 2048 / 44_100, accuracy: 1e-9)

        session.screenplayMode = true
        session.script = "ANN: Hi.\nBEN: Hello."
        let speakers = try session.prepared.get().speakers
        session.setVoice(.library(ann.id), for: speakers[0])
        XCTAssertEqual(session.casting.references.map(\.name), ["Ann"])

        session.removeVoice(ann.id)
        XCTAssertTrue(session.library.voices.isEmpty)
        XCTAssertTrue(session.speakerVoices.isEmpty)
        XCTAssertEqual(session.casting.references, [])
    }

    func testVoiceLibraryPersistsAsJSONAndSurvivesReload() throws {
        let library = VoiceLibrary(directory: directory)
        let voice = ReferenceVoice(name: "Narrator", transcript: "Once upon a time.", codes: Array(repeating: [1, 2, 3], count: 10), sourceFileName: "n.wav")
        try library.add(voice)
        let files = try FileManager.default.contentsOfDirectory(atPath: directory.path)
        XCTAssertEqual(files, ["\(voice.id.uuidString).json"])
        try Data("not json".utf8).write(to: directory.appending(path: "broken.json"))

        let reloaded = VoiceLibrary(directory: directory)
        XCTAssertEqual(reloaded.voices, [voice])
        XCTAssertEqual(reloaded.unreadableFiles, ["broken.json"])
        XCTAssertEqual(voice.frames[2], [3 + ModelConstants.semanticStart] + Array(repeating: 3, count: 10))
    }
}

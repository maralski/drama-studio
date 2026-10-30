// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
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

    func testGenerationStreamsEveryChunkThenSettlesReady() async throws {
        let session = makeSession()
        session.script = String(repeating: "This sentence is about forty bytes long. ", count: 3)
        session.generate()
        await waitUntil { !session.state.isGenerating }
        XCTAssertEqual(session.state, .ready)
        let result = try XCTUnwrap(session.result)
        XCTAssertEqual(result.sentences.map(\.text), Array(repeating: "This sentence is about forty bytes long.", count: 3))
        XCTAssertEqual(session.audio.count, result.samples.count)
        XCTAssertEqual(session.audio.count, result.frames.count * ModelConstants.samplesPerFrame)
        XCTAssertTrue(session.canExport)
    }

    func testChunkEventsUpdateGenerationProgress() {
        let session = makeSession()
        session.apply(.chunkStarted(index: 0, count: 3, text: "First sentence."))
        XCTAssertEqual(session.state, .generating(chunk: 1, of: 3, playing: false))
        session.apply(.chunkStarted(index: 1, count: 3, text: "Second sentence."))
        XCTAssertEqual(session.state, .generating(chunk: 2, of: 3, playing: false))
        session.apply(.chunkStarted(index: 2, count: 3, text: "Third sentence."))
        XCTAssertEqual(session.state, .generating(chunk: 3, of: 3, playing: false))
    }

    func testAudioArrivesBeforeTheFirstSentenceIsDoneAndMatchesTheTake() async {
        let session = makeSession(pace: .init(loadSeconds: 0, framesPerSecond: 200))
        session.quickPreview = true
        session.script = "The first sentence is long enough to take a while to say. The second sentence ends the script."
        session.generate()
        await waitUntil { !session.audio.isEmpty || !session.state.isGenerating }
        XCTAssertEqual(session.state, .generating(chunk: 1, of: 2, playing: false), "the first audio lands while sentence one is generating")
        XCTAssertEqual(session.audio.count, 64 * ModelConstants.samplesPerFrame, "one 64-frame partial")
        XCTAssertNotNil(session.estimateLabel(at: .now))
        await waitUntil { !session.state.isGenerating }
        XCTAssertEqual(session.audio, session.result?.samples, "partials, then each chunk's remainder, add up to the take")
        XCTAssertEqual(session.takes.last?.samples, session.audio)
    }

    func testNormalGenerationPublishesCompletedChunksWithoutDuplicatingAudio() async throws {
        let session = makeSession(pace: .init(loadSeconds: 0, framesPerSecond: 200))
        session.script = "The first sentence is long enough to take a while to say. The second sentence ends the script."
        session.generate()
        await waitUntil { !session.audio.isEmpty || !session.state.isGenerating }
        XCTAssertEqual(session.state, .generating(chunk: 2, of: 2, playing: false))
        let firstChunk = session.audio
        await waitUntil { !session.state.isGenerating }
        let result = try XCTUnwrap(session.result)
        XCTAssertEqual(firstChunk.count, result.sentences[0].frames * ModelConstants.samplesPerFrame)
        XCTAssertEqual(firstChunk, Array(result.samples.prefix(firstChunk.count)))
        XCTAssertEqual(session.audio, result.samples)
        XCTAssertEqual(session.audio.count, result.frames.count * ModelConstants.samplesPerFrame)
        XCTAssertEqual(session.takes.last?.samples, result.samples)
    }

    func testStreamingSettingTurnsLowLatencyOff() {
        let session = makeSession()
        UserDefaults.standard.set(false, forKey: SessionModel.streamsAudioKey)
        defer { UserDefaults.standard.removeObject(forKey: SessionModel.streamsAudioKey) }
        XCTAssertEqual(session.config.streaming, .sentence)
    }

    func testRequestFollowsTheControlsAndNewTakeNeedsATake() async {
        let session = makeSession()
        session.script = "One line to say."
        var expected = GenerationConfig()
        expected.streaming = .defaultLowLatency
        XCTAssertEqual(session.config, expected, "playback streams by default")
        session.quickPreview = true
        session.expressiveness = 1
        expected = .make(expressiveness: 1, quickPreview: true)
        expected.streaming = .defaultLowLatency
        XCTAssertEqual(session.config, expected)

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

    func testStoppedPartialAudioSurvivesProjectSaveAndReopen() async throws {
        let session = makeSession(pace: .init(loadSeconds: 0, framesPerSecond: 500))
        session.script = String(repeating: "A sentence that takes time to generate. ", count: 10)
        session.generate()
        await waitUntil { !session.audio.isEmpty }
        session.stopGeneration()
        XCTAssertTrue(session.takes.last?.isIncomplete == true)
        XCTAssertNotNil(session.exportWarning)
        XCTAssertTrue(session.hasUnsavedChanges)
        let bundle = directory.appending(path: "Partial.dramaproj")
        XCTAssertTrue(session.saveProject(to: bundle))
        let restored = makeSession()
        restored.openProject(from: bundle)
        XCTAssertTrue(restored.audio == session.audio)
        XCTAssertTrue(restored.takes.last?.isIncomplete == true)
        XCTAssertNotNil(restored.exportWarning)
        XCTAssertFalse(restored.canRetake)
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

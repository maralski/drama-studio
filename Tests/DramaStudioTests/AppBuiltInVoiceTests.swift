// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import DramaEngine
import XCTest
@testable import DramaStudio

/// A mock engine that keeps the requests it ran and reports every take as word-checked.
private final class RequestLog: @unchecked Sendable {
    private let lock = NSLock()
    private var kept: [SpeechRequest] = []
    var requests: [SpeechRequest] { lock.withLock { kept } }
    func add(_ request: SpeechRequest) { lock.withLock { kept.append(request) } }

    func factory(_ engine: SpeechEngineID) -> SpeechEngines.Factory {
        let mock = MockSynthesizer(engine: engine, pace: .instant)
        return SpeechEngines.Factory(description: mock.engineDescription) { LoggingSynthesizer(mock: mock, log: self) }
    }
}

private struct LoggingSynthesizer: SpeechSynthesizer {
    let mock: MockSynthesizer
    let log: RequestLog
    var isLoaded: Bool { get async { await mock.isLoaded } }
    func load(onPhase: @escaping @Sendable (String) -> Void) async throws { try await mock.load(onPhase: onPhase) }
    var engineDescription: EngineDescription { mock.engineDescription }
    var isTuned: Bool? { mock.isTuned }
    func encodeVoice(samples: [Float], transcript: String) async throws -> VoiceEncoding { try await mock.encodeVoice(samples: samples, transcript: transcript) }
    func synthesize(_ request: SpeechRequest) -> AsyncThrowingStream<SpeechEvent, Error> {
        log.add(request)
        let events = mock.synthesize(request)
        return AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    for try await event in events {
                        guard case .finished(var result) = event else { continuation.yield(event); continue }
                        result.wordCoverage = .recognized
                        continuation.yield(.finished(result))
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }
}

@MainActor
final class AppBuiltInVoiceTests: XCTestCase {
    private var directory: URL!
    private var log: RequestLog!
    private let george = BuiltInVoice(engine: .ming, name: "UK_George")
    private let narrator = Speaker(index: 0, name: "Narrator")

    override func setUp() async throws {
        directory = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        log = RequestLog()
    }

    override func tearDown() async throws {
        AppHome.defaults.removeObject(forKey: ModelSetup.engineKey)
        try? FileManager.default.removeItem(at: directory)
    }

    private func makeSession(selected: SpeechEngineID, installed: [SpeechEngineID] = [.drama3, .ming]) throws -> SessionModel {
        let library = VoiceLibrary(directory: directory.appending(path: "Voices"))
        let engines = try SpeechEngines(selected: selected, factories: Dictionary(uniqueKeysWithValues: installed.map { ($0, log.factory($0)) }), voices: library)
        return SessionModel(synthesizer: engines, library: library, pronunciationURL: directory.appending(path: "pronunciation.json"))
    }

    private func generate(_ session: SessionModel) async {
        session.generate()
        let deadline = Date().addingTimeInterval(10)
        while session.state.isGenerating, Date() < deadline { try? await Task.sleep(for: .milliseconds(10)) }
    }

    func testASoloBuiltInVoiceCastsTheRequestAndSeveralSpeakersIgnoreIt() {
        let take = Take(text: "Hello.", context: .drama3(codes: [[7]]))
        let fresh = Casting(speakers: [narrator], assignments: [0: .builtIn(george)], library: [])
        XCTAssertEqual(fresh.builtInVoice, george)
        XCTAssertEqual(fresh.recordsVoice, .builtIn(george))
        XCTAssertEqual(fresh.request(text: "Hi.", previousTakes: [], config: GenerationConfig()).builtInVoice, george)

        let recorded = Casting(speakers: [narrator], assignments: [0: .builtIn(george)], library: [], sessionVoices: [.builtIn(george): take])
        XCTAssertEqual(recorded.builtInVoice, george)
        XCTAssertEqual(recorded.voiceTake, take)
        XCTAssertNil(recorded.recordsVoice)

        let duo = Casting(speakers: [narrator, Speaker(index: 1, name: "B")], assignments: [0: .builtIn(george)], library: [])
        XCTAssertNil(duo.builtInVoice)
        XCTAssertEqual(duo.ignored.map(\.reason), [.needsLibrary])
    }

    /// Catches the app speaking as US_Emma whatever the picker says, and the
    /// second take losing the voice.
    func testGenerateSpeaksInTheChosenVoiceAndLaterTakesContinueIt() async throws {
        let session = try makeSession(selected: .ming)
        session.script = "The lamp was lit before the fog rolled in."
        session.setVoice(.builtIn(george), for: narrator)
        XCTAssertNil(session.voiceProblem(for: narrator))
        await generate(session)
        XCTAssertEqual(session.takes.count, 1)
        XCTAssertEqual(log.requests.map(\.builtInVoice), [george])
        XCTAssertEqual(log.requests[0].previousTakes, [])

        await generate(session)
        XCTAssertEqual(log.requests.map(\.builtInVoice), [george, george])
        XCTAssertEqual(log.requests[1].previousTakes, [try XCTUnwrap(session.takes.first?.take)])

        session.setVoice(.builtIn(BuiltInVoice(engine: .ming, name: "US_Emma")), for: narrator)
        await generate(session)
        XCTAssertEqual(log.requests[2].builtInVoice?.name, "US_Emma")
        XCTAssertEqual(log.requests[2].previousTakes, [], "another voice does not continue from George's take")
    }

    /// Catches Drama 3 reading the line in a voice of its own with nothing said.
    func testOnAnotherEngineGenerateFailsWithTheReasonAndRunsNothing() async throws {
        let session = try makeSession(selected: .drama3)
        session.script = "The lamp was lit before the fog rolled in."
        session.setVoice(.builtIn(george), for: narrator)
        XCTAssertEqual(session.voiceProblem(for: narrator),
                       "UK_George is a built-in Ming voice, and the engine is Drama 3. Switch the engine to Ming, or choose another voice.")
        await generate(session)
        XCTAssertEqual(session.state, .failed("Drama 3 can't speak in UK_George, a built-in Ming voice; switch to Ming or choose another voice"))
        XCTAssertEqual(session.takes.count, 0)
        XCTAssertEqual(log.requests.count, 0)

        session.engine = .ming
        XCTAssertNil(session.voiceProblem(for: narrator))
        await generate(session)
        XCTAssertEqual(session.takes.count, 1)
        XCTAssertEqual(log.requests.map(\.builtInVoice), [george])
    }

    func testAProjectKeepsItsBuiltInVoice() async throws {
        let session = try makeSession(selected: .ming)
        session.script = "The lamp was lit before the fog rolled in."
        session.setVoice(.builtIn(george), for: narrator)
        await generate(session)
        let bundle = directory.appending(path: "george.dramaproj")
        XCTAssertTrue(session.saveProject(to: bundle), session.notice ?? "save")
        let reopened = try makeSession(selected: .ming)
        XCTAssertTrue(reopened.openProject(from: bundle), reopened.notice ?? "open")
        XCTAssertEqual(reopened.voice(for: narrator), .builtIn(george))
        XCTAssertEqual(reopened.castBuiltIns, [george])
        XCTAssertFalse(reopened.isOutdated(try XCTUnwrap(reopened.takes.first)))
    }

    func testThePickerListsInstalledEnginesVoicesUnderHeadingsAndSearchNarrowsThem() throws {
        XCTAssertEqual(try makeSession(selected: .drama3, installed: [.drama3]).builtInVoices, [])
        let voices = try makeSession(selected: .drama3).builtInVoices
        let groups = BuiltInVoiceGroup.groups(of: voices)
        XCTAssertEqual(groups.map(\.name), ["Chinese", "Chinese characters", "Chinese, in one emotion", "English"])
        XCTAssertEqual(groups.map(\.voices.count).reduce(0, +), 241)
        XCTAssertEqual(BuiltInVoiceGroup.groups(of: voices, matching: " emma ").flatMap(\.voices).map(\.name), ["US_Emma"])
        XCTAssertEqual(BuiltInVoiceGroup.groups(of: voices, matching: "english").flatMap(\.voices).count, 3)
        let happy = BuiltInVoiceGroup.groups(of: voices, matching: "happy").flatMap(\.voices)
        XCTAssertFalse(happy.isEmpty)
        XCTAssertTrue(happy.allSatisfy { $0.note == "happy" })
        XCTAssertEqual(BuiltInVoiceGroup.groups(of: voices, matching: "灵晓声").flatMap(\.voices).count, 5)
        XCTAssertEqual(BuiltInVoiceGroup.groups(of: voices, matching: "zzz"), [])
    }
}

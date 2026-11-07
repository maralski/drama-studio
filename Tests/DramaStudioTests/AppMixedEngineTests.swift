// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import DramaEngine
import XCTest
@testable import DramaStudio

/// A session on the engine registry, with mocks standing in for both engines.
@MainActor
final class AppMixedEngineTests: XCTestCase {
    private var directory: URL!

    override func setUp() async throws {
        directory = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    override func tearDown() async throws {
        try? FileManager.default.removeItem(at: directory)
    }

    private func makeSession(library: VoiceLibrary? = nil) throws -> SessionModel {
        let library = library ?? VoiceLibrary(directory: directory.appending(path: "Voices"))
        let engines = try SpeechEngines(selected: .drama3, factories: [.drama3: .mock(.drama3, pace: .instant), .ming: .mock(.ming, pace: .instant)], voices: library)
        return SessionModel(synthesizer: engines, library: library, pronunciationURL: directory.appending(path: "pronunciation.json"))
    }

    private func waitUntil(_ condition: @escaping () -> Bool, timeout: Double = 10) async {
        let deadline = Date().addingTimeInterval(timeout)
        while !condition(), Date() < deadline { try? await Task.sleep(for: .milliseconds(10)) }
    }

    private func generate(_ session: SessionModel, _ script: String) async {
        session.script = script
        session.generate()
        await waitUntil { !session.state.isGenerating }
        XCTAssertNil(session.notice)
    }

    func testAProjectWithTakesFromBothEnginesSavesReopensAndTimesItsSentences() async throws {
        let session = try makeSession()
        await generate(session, "The first take speaks on Drama. It has two sentences.")
        XCTAssertEqual(session.takes.last?.take?.context.engine, .drama3)

        session.engine = .ming
        XCTAssertEqual(session.engine, .ming)
        XCTAssertEqual(AppHome.defaults.string(forKey: ModelSetup.engineKey), "ming")
        await generate(session, "The second take speaks on Ming. [laughs] It also has two sentences.")
        XCTAssertEqual(session.takes.map { $0.take?.context.engine }, [.drama3, .ming])
        XCTAssertEqual(session.notices, [], "every cue in the line maps to a Ming tag")

        let bundle = directory.appending(path: "mixed.dramaproj")
        XCTAssertTrue(session.saveProject(to: bundle), session.notice ?? "save")
        let project = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: bundle.appending(path: "project.json"))) as? [String: Any])
        XCTAssertEqual(project["version"] as? Int, 5)

        let reopened = try makeSession()
        XCTAssertTrue(reopened.openProject(from: bundle), reopened.notice ?? "open")
        XCTAssertEqual(reopened.takes.map { $0.take?.context.engine }, [.drama3, .ming])
        XCTAssertEqual(reopened.takes.map(\.samples), session.takes.map(\.samples))
        for take in reopened.takes {
            XCTAssertEqual(take.take?.sampleCount, take.samples.count, "the context's length matches the audio")
            XCTAssertEqual(take.sentences.map(\.samples).reduce(0, +), take.samples.count, "sentence samples add up to the take")
            XCTAssertEqual(take.sentences.count, 2)
        }
        XCTAssertEqual(reopened.sentenceSpans.map { $0.range.upperBound - $0.range.lowerBound }, session.sentenceSpans.map { $0.range.upperBound - $0.range.lowerBound })
        XCTAssertEqual(reopened.audio.count, session.audio.count)
        AppHome.defaults.removeObject(forKey: ModelSetup.engineKey)
    }

    func testAnOldVoiceGainsAMingEncodingInTheLibraryWhenMingSpeaksIt() async throws {
        let library = VoiceLibrary(directory: directory.appending(path: "Voices"))
        let clip = (0..<22_050).map { Float(sin(Double($0) * 0.02)) }
        let old = ReferenceVoice(name: "Old", transcript: "A clip the library keeps.", encodings: [.drama3(codes: MockSynthesizer.codes(frameCount: 11))])
        try library.add(old, clip: clip)
        let session = try makeSession(library: library)
        session.engine = .ming
        session.setVoice(.library(old.id), forSpeaker: 0)
        await generate(session, "Speak with the old voice.")
        let stored = try XCTUnwrap(library.voice(old.id))
        XCTAssertEqual(Set(stored.encodings.keys), [.drama3, .ming])
        XCTAssertEqual(stored.encodings[.ming]?.sampleCount, clip.count)
        XCTAssertEqual(try library.clip(for: old.id)?.count, clip.count, "the clip is untouched")
        XCTAssertEqual(session.takes.last?.take?.context.engine, .ming)
        AppHome.defaults.removeObject(forKey: ModelSetup.engineKey)
    }

    func testUnsupportedCuesReachTheSessionBeforeAndAfterTheRun() async throws {
        let session = try makeSession()
        session.engine = .ming
        await generate(session, "<whisper>Quietly now.</whisper> [icy calm] Then louder.")
        XCTAssertEqual(session.notices, [.unsupportedCue("<whisper>"), .unsupportedCue("[icy calm]")])
        XCTAssertEqual(session.result?.notices, session.notices)
        AppHome.defaults.removeObject(forKey: ModelSetup.engineKey)
    }

    func testEveryPaletteCueHasADecidedMingMeaning() {
        let expected: [String: CueSupport] = [
            "[laughs]": .tag("[Laughter]"), "[chuckles]": .tag("[Laughter]"), "[sighs]": .tag("[Sigh]"), "[gasps]": .tag("[Surprise-ah]"),
            "[pause]": .unsupported, "[clears throat]": .tag("[Cough]"), "[hesitates]": .tag("[Uhm]"), "[coughs]": .tag("[Cough]"), "[whispers]": .unsupported,
            "[with bright eager excitement]": .style("happy"), "[icy calm]": .unsupported, "[heavy, deflated sadness]": .style("sad"),
            "[sarcastic and deadpan]": .unsupported, "[breathless and terrified]": .style("fearful"), "[furious, barely holding it together]": .style("angry"),
            "[warm, gentle bedtime voice]": .unsupported, "[low and menacing]": .unsupported,
            "<whisper>": .unsupported, "<slow>": .unsupported, "<emphasis>": .unsupported,
        ]
        let actual = Dictionary(uniqueKeysWithValues: PerformanceCue.all.map { ($0.opening, SpeechEngineID.ming.cueSupport($0.opening)) })
        XCTAssertEqual(actual, expected, "a palette cue gained or lost its Ming meaning; decide it here and in MingCues")
        XCTAssertTrue(PerformanceCue.all.allSatisfy { SpeechEngineID.drama3.cueSupport($0.opening) == .native })
    }
}

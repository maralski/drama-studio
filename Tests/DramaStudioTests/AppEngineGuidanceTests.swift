// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import DramaEngine
import Observation
import XCTest
@testable import DramaStudio

/// What the app tells people about the engine they are on, read from the engine table.
@MainActor
final class AppEngineGuidanceTests: XCTestCase {
    private var directory: URL!

    override func setUp() async throws {
        directory = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    override func tearDown() async throws {
        AppHome.defaults.removeObject(forKey: ModelSetup.engineKey)
        try? FileManager.default.removeItem(at: directory)
    }

    private func session(_ factories: [SpeechEngineID: SpeechEngines.Factory], selected: SpeechEngineID) throws -> SessionModel {
        let library = VoiceLibrary(directory: directory.appending(path: "Voices"))
        return SessionModel(synthesizer: try SpeechEngines(selected: selected, factories: factories, voices: library), library: library,
                            pronunciationURL: directory.appending(path: "pronunciation.json"))
    }

    private func generate(_ session: SessionModel, _ script: String) async {
        session.script = script
        session.generate()
        let deadline = Date().addingTimeInterval(10)
        while session.state.isGenerating, Date() < deadline { try? await Task.sleep(for: .milliseconds(10)) }
    }

    /// The Speech menu and the Inspector both read `engine` and
    /// `availableEngines` from the one session, and every change setup makes
    /// to the registry must reach whoever read them.
    func testInstallingAnEngineNotifiesWhatTheEngineMenuRead() async throws {
        let session = try session([.drama3: .placeholder(.drama3)], selected: .drama3)
        XCTAssertEqual(session.availableEngines, [])
        let changed = expectation(description: "the menu's reads are invalidated")
        withObservationTracking {
            _ = session.engine
            _ = session.availableEngines
        } onChange: { changed.fulfill() }
        await session.install(.ming, .mock(.ming, pace: .instant))
        await fulfillment(of: [changed], timeout: 1)
        XCTAssertEqual(session.availableEngines, [.ming])
        XCTAssertEqual(session.engine, .ming, "the first engine installed takes the selection from the placeholder")
        XCTAssertEqual(session.engineDescription.engine, .ming, "the menu bar status follows the selection")
    }

    func testTheSelectionReachesWhatTheMenuBarStatusRead() throws {
        let session = try session([.drama3: .mock(.drama3, pace: .instant), .ming: .mock(.ming, pace: .instant)], selected: .drama3)
        let changed = expectation(description: "the status line is invalidated")
        withObservationTracking { _ = session.engineDescription } onChange: { changed.fulfill() }
        session.engine = .ming
        wait(for: [changed], timeout: 1)
        XCTAssertEqual(session.engineDescription.engine, .ming)
    }

    func testATakeNamesItsEngineOnlyWhereThatTellsTakesApart() async throws {
        let single = try session([.drama3: .mock(.drama3, pace: .instant)], selected: .drama3)
        await generate(single, "One engine.")
        XCTAssertNil(single.engineLabel(for: try XCTUnwrap(single.takes.last)), "nothing to tell apart with one engine")

        let both = try session([.drama3: .mock(.drama3, pace: .instant), .ming: .mock(.ming, pace: .instant)], selected: .drama3)
        await generate(both, "Two engines.")
        XCTAssertEqual(both.engineLabel(for: try XCTUnwrap(both.takes.last)), "Drama 3")
    }

    func testEachTakeKeepsItsOwnGenerationTime() async throws {
        let session = try session([.drama3: .mock(.drama3, pace: .instant)], selected: .drama3)
        await generate(session, "First.")
        await generate(session, "Second, a little longer.")
        XCTAssertEqual(session.takes.count, 2)
        XCTAssertTrue(session.takes.allSatisfy { $0.generationSeconds != nil }, "the status line shows the selected take's own numbers")
    }

    func testMingsLimitsReadInPeoplesTerms() {
        XCTAssertNil(SpeechEngineID.drama3.limits)
        XCTAssertEqual(SpeechEngineID.ming.limits,
                       "Ming performs fewer cues and leaves out the rest with a notice. It has no passage cues such as <whisper>, "
                       + "clones best from clips of up to 15 seconds, speaks 7 languages, doesn't build on earlier takes, and ignores Expressiveness.")
    }

    func testAnIncompleteMingFolderListsEveryMissingFile() throws {
        try Data().write(to: directory.appending(path: "config.json"))
        XCTAssertEqual(MingInstall.missingFiles(in: directory), ["model.safetensors", "tokenizer.json", "campplus.onnx"])
    }
}

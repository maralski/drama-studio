// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import DramaEngine
import Observation
import XCTest
@testable import DramaStudio

/// The engine picker reads the session's engine and installed engines; both
/// live in the registry, so the session must tell observers when they change.
@MainActor
final class AppEngineChoiceTests: XCTestCase {
    private var directory: URL!

    override func setUp() async throws {
        directory = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
    }

    override func tearDown() async throws {
        try? FileManager.default.removeItem(at: directory)
    }

    private func changes(_ read: @escaping () -> Void) -> () -> Bool {
        final class Flag: @unchecked Sendable { var raised = false }
        let flag = Flag()
        withObservationTracking(read) { flag.raised = true }
        return { flag.raised }
    }

    func testInstallingMingMakesItSelectableAndTellsThePicker() async throws {
        let library = VoiceLibrary(directory: directory.appending(path: "Voices"))
        let session = SessionModel(synthesizer: try SpeechEngines(selected: .drama3, factories: [.drama3: .mock(.drama3, pace: .instant)], voices: library),
                                   library: library, pronunciationURL: directory.appending(path: "pronunciation.json"))
        XCTAssertEqual(session.availableEngines, [.drama3])
        session.engine = .ming
        XCTAssertEqual(session.engine, .drama3, "an engine without its files can't be chosen")
        XCTAssertNotNil(session.notice)

        let installedChanged = changes { _ = session.availableEngines }
        await session.install(.ming, .mock(.ming, pace: .instant))
        XCTAssertTrue(installedChanged())
        XCTAssertEqual(session.availableEngines, [.drama3, .ming])

        let engineChanged = changes { _ = session.engine }
        session.engine = .ming
        XCTAssertTrue(engineChanged())
        XCTAssertEqual(session.engine, .ming)
    }
}

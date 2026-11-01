// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import DramaEngine
import XCTest
@testable import DramaStudio

@MainActor
final class AppTakeDeletionTests: XCTestCase {
    private var directory: URL!

    override func setUp() async throws {
        directory = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    override func tearDown() async throws {
        try? FileManager.default.removeItem(at: directory)
    }

    private func session() -> SessionModel {
        let model = SessionModel(synthesizer: RecordingSynthesizer(), library: VoiceLibrary(directory: directory.appending(path: "voices")),
                                 pronunciationURL: directory.appending(path: "pronunciation.json"))
        model.playsWhileGenerating = false
        model.script = "A line to compare."
        return model
    }

    private func generate(_ model: SessionModel) async {
        model.generate()
        for _ in 0..<500 where model.state.isGenerating { try? await Task.sleep(for: .milliseconds(10)) }
        XCTAssertFalse(model.state.isGenerating)
    }

    func testDeletingUnusedAndExportedScriptTakesPreservesRemainingAudio() async throws {
        let model = session()
        for _ in 0..<3 { await generate(model) }
        let takes = model.visibleTakes
        let saved = directory.appending(path: "script.dramaproj")
        XCTAssertTrue(model.saveProject(to: saved))
        model.deleteTake(takes[1].id)
        XCTAssertEqual(model.visibleTakes.map(\.number), [1, 3])
        XCTAssertEqual(model.selectedTakeID, takes[2].id)
        XCTAssertEqual(model.visibleExportTakeID, takes[0].id)
        XCTAssertEqual(model.audio, takes[2].samples)
        XCTAssertTrue(model.hasUnsavedChanges)
        model.deleteTake(takes[0].id)
        XCTAssertEqual(model.visibleExportTakeID, takes[2].id)
        XCTAssertEqual(model.exportSamples, takes[2].samples)
        XCTAssertTrue(model.saveProject(to: saved))
        let reopened = session()
        reopened.openProject(from: saved)
        XCTAssertEqual(reopened.visibleTakes.map(\.number), [3])
        XCTAssertEqual(reopened.exportSamples, takes[2].samples)
    }

    func testDeletingSelectedAndLastScriptTakeClearsPlaybackAndSurvivesReopen() async throws {
        let model = session()
        await generate(model)
        let first = try XCTUnwrap(model.visibleTakes.first)
        await generate(model)
        model.deleteTake(try XCTUnwrap(model.visibleSelectedTakeID))
        XCTAssertEqual(model.selectedTakeID, first.id)
        XCTAssertEqual(model.audio, first.samples)
        model.deleteTake(first.id)
        XCTAssertTrue(model.visibleTakes.isEmpty)
        XCTAssertTrue(model.audio.isEmpty)
        XCTAssertTrue(model.exportSamples.isEmpty)
        XCTAssertNil(model.selectedTakeID)
        XCTAssertNil(model.visibleExportTakeID)
        XCTAssertNil(model.result)
        XCTAssertEqual(model.state, .idle)
        XCTAssertFalse(model.canPlay)
        XCTAssertFalse(model.canExport)
        let saved = directory.appending(path: "empty.dramaproj")
        XCTAssertTrue(model.saveProject(to: saved))
        let reopened = session()
        reopened.openProject(from: saved)
        XCTAssertTrue(reopened.visibleTakes.isEmpty)
        XCTAssertTrue(reopened.audio.isEmpty)
        XCTAssertFalse(reopened.canExport)
        await generate(reopened)
        XCTAssertEqual(reopened.visibleTakes.count, 1)
    }

    func testDeletingSectionTakesReplacesExportAndRemovesOnlyEmptySectionAudio() async throws {
        let model = session()
        let first = NarrationDocument.Segment(index: 1, label: "One", text: "First section.")
        let second = NarrationDocument.Segment(index: 2, label: "Two", text: "Second section.")
        model.open(NarrationDocument(title: "Sections", source: .pptx, segments: [first, second]))
        await generate(model)
        let older = try XCTUnwrap(model.visibleTakes.first)
        await generate(model)
        let newest = try XCTUnwrap(model.visibleTakes.last)
        let other = try XCTUnwrap(model.chosenTake(for: second.id))
        model.selectTake(older.id)
        model.deleteTake(older.id)
        XCTAssertNil(model.previewingTakeID)
        XCTAssertEqual(model.visibleSelectedTakeID, newest.id)
        XCTAssertEqual(model.visibleExportTakeID, newest.id)
        XCTAssertEqual(model.exportSamples, newest.samples + other.samples)
        await generate(model)
        let latest = try XCTUnwrap(model.visibleTakes.last)
        model.deleteTake(latest.id)
        XCTAssertEqual(model.visibleExportTakeID, newest.id)
        let remainingOther = try XCTUnwrap(model.chosenTake(for: second.id))
        XCTAssertEqual(model.exportSamples, newest.samples + remainingOther.samples)
        model.deleteTake(newest.id)
        XCTAssertTrue(model.visibleTakes.isEmpty)
        XCTAssertNil(model.chosenTake(for: first.id))
        XCTAssertNil(model.visibleSelectedTakeID)
        XCTAssertEqual(model.exportSamples, remainingOther.samples)
        XCTAssertEqual(model.narrationTakes.map(\.segmentID), [second.id])
        let saved = directory.appending(path: "sections.dramaproj")
        XCTAssertTrue(model.saveProject(to: saved))
        let reopened = session()
        reopened.openProject(from: saved)
        XCTAssertNil(reopened.chosenTake(for: first.id))
        XCTAssertEqual(reopened.exportSamples, remainingOther.samples)
        let remainingIDs = reopened.sectionTakes.values.flatMap { $0.takes.map(\.id) }
        for id in remainingIDs { reopened.deleteTake(id) }
        XCTAssertTrue(reopened.sectionTakes.isEmpty)
        XCTAssertTrue(reopened.audio.isEmpty)
        XCTAssertTrue(reopened.exportSamples.isEmpty)
        XCTAssertEqual(reopened.state, .idle)
        XCTAssertFalse(reopened.canExport)
        XCTAssertTrue(reopened.canGenerateChangedSections)
    }

    func testDeletionIsIgnoredDuringGenerationAndForUnknownIDs() async throws {
        let model = session()
        await generate(model)
        let take = try XCTUnwrap(model.visibleTakes.first)
        model.deleteTake(UUID())
        XCTAssertEqual(model.visibleTakes.map(\.id), [take.id])
        model.generate()
        XCTAssertTrue(model.state.isGenerating)
        model.deleteTake(take.id)
        XCTAssertTrue(model.visibleTakes.contains { $0.id == take.id })
        model.stopGeneration()
    }
}

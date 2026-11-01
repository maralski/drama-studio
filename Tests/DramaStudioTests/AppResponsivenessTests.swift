// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import AppKit
import DramaEngine
import SwiftUI
import XCTest
@testable import DramaStudio

@MainActor
final class AppResponsivenessTests: XCTestCase {
    private func session() -> SessionModel {
        SessionModel(synthesizer: MockSynthesizer(),
                     library: VoiceLibrary(directory: FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)))
    }

    func testPreparedScriptTracksEditingModeAndSourceReplacement() throws {
        let model = session()
        model.script = "Alex: Hello."
        XCTAssertEqual(try model.prepared.get().text, "Alex: Hello.")
        model.screenplayMode = true
        XCTAssertEqual(try model.prepared.get().text, "<|speaker:0|>Hello.")
        model.editorText = "Sam: Goodbye."
        XCTAssertEqual(try model.prepared.get().speakers.map(\.name), ["Sam"])
        model.source = .plain("Ordinary prose.")
        XCTAssertEqual(model.prepared, .failure(.noDialogue))
        model.screenplayMode = false
        XCTAssertEqual(try model.prepared.get().text, "Ordinary prose.")
        model.clearScriptInput()
        XCTAssertEqual(model.prepared, .failure(.empty))
        XCTAssertFalse(model.canGenerate)
    }

    func testClearSectionPreservesOtherSections() throws {
        let model = session()
        model.open(NarrationDocument(title: "Two sections", source: .pdf, segments: [
            .init(index: 1, label: "First", text: "First line."),
            .init(index: 2, label: "Second", text: "Second line.")
        ]))
        let first = try XCTUnwrap(model.document?.segments.first)
        model.selectSegment(first.id)
        model.clearScriptInput()
        XCTAssertEqual(model.document?.segments.map(\.text), ["", "Second line."])
        XCTAssertNil(model.selection)
    }

    func testTitleSkipsCuesAndTracksEdits() {
        let model = session()
        model.script = "<|speaker:0|>[laughs] <soft>Café 42, go now please.</soft> More words."
        XCTAssertEqual(model.title, "Café 42 go now please")
        model.script = "New words."
        XCTAssertEqual(model.title, "New words")
        XCTAssertEqual(SessionModel.suggestedName(for: "[pause]<soft></soft>"), "Drama Studio Take")
        XCTAssertEqual(SessionModel.suggestedName(for: "Cafe\u{301} 你好 123"), "Cafe\u{301} 你好 123")
    }

    func testClearScriptCanBeUndoneAndRedone() async throws {
        let model = session()
        model.script = "Keep this line."
        let host = NSHostingView(rootView: ScriptEditorView(model: model))
        let window = NSWindow(contentRect: CGRect(x: 0, y: 0, width: 700, height: 500),
                              styleMask: [.titled], backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        window.contentView = host
        defer { window.close() }
        host.layoutSubtreeIfNeeded()
        try await Task.sleep(for: .milliseconds(20))
        model.clearScriptInput()
        host.layoutSubtreeIfNeeded()
        try await Task.sleep(for: .milliseconds(20))
        XCTAssertEqual(model.editorText, "")
        let undo = try XCTUnwrap(window.undoManager)
        XCTAssertTrue(undo.canUndo)
        undo.undo()
        XCTAssertEqual(model.editorText, "Keep this line.")
        undo.redo()
        XCTAssertEqual(model.editorText, "")
    }

    func testScreenplayErrorExplainsFormatAndHelp() {
        let message = ScriptError.noDialogue.localizedDescription
        XCTAssertTrue(message.contains("Alex: Hello."))
        XCTAssertTrue(message.contains("help"))
        XCTAssertTrue(SettingHelp.screenplayMode.text.contains("Sam: Good morning."))
    }

    func testWaveformResponsivenessAudit() throws {
        guard ProcessInfo.processInfo.environment["DRAMA_UI_BENCH"] == "1" else {
            throw XCTSkip("Set DRAMA_UI_BENCH=1 for the release-mode UI audit.")
        }
        let samples = (0..<(44_100 * 180)).map { Float($0 % 200 - 100) / 100 }
        func baseline() -> [Float] {
            let size = (samples.count + 159) / 160
            let raw = stride(from: 0, to: samples.count, by: size).map { start in
                samples[start..<min(start + size, samples.count)].reduce(Float(0)) { max($0, abs($1)) }
            }
            let loudest = raw.max() ?? 0
            return loudest > 0 ? raw.map { $0 / loudest } : raw
        }
        XCTAssertEqual(baseline(), WaveformScrubber.peaks(of: samples, bins: 160))
        for round in 0..<6 {
            for optimized in (round.isMultiple(of: 2) ? [false, true] : [true, false]) {
                let start = ContinuousClock.now
                let peaks = optimized ? WaveformScrubber.peaks(of: samples, bins: 160) : baseline()
                let elapsed = ContinuousClock.now - start
                XCTAssertEqual(peaks.count, 160)
                XCTAssertTrue(peaks.allSatisfy { $0 == 1 })
                if round > 0 {
                    print("WAVEFORM_AUDIT optimized=\(optimized) round=\(round) ms=\(Double(elapsed.components.attoseconds) / 1e15)")
                }
            }
        }
    }

    func testEditingResponsivenessAudit() throws {
        guard ProcessInfo.processInfo.environment["DRAMA_UI_BENCH"] == "1" else {
            throw XCTSkip("Set DRAMA_UI_BENCH=1 for the release-mode UI audit.")
        }
        let model = session()
        model.screenplayMode = true
        let text = String(repeating: "Alex: We should leave before the rain starts.\nSam: I will get the coats.\n", count: 200)
        let host = NSHostingView(rootView: ScriptEditorView(model: model).frame(width: 700, height: 500))
        host.frame = CGRect(x: 0, y: 0, width: 700, height: 500)
        model.script = text
        host.layoutSubtreeIfNeeded()
        for round in 0..<6 {
            let start = ContinuousClock.now
            for edit in 0..<20 {
                model.editorText = text + "Alex: Ending \(round) \(edit)."
                // These consumers are read by the footer, Cast, transport and take controls.
                XCTAssertTrue(model.canGenerate)
                XCTAssertEqual(try model.prepared.get().speakers.count, 2)
                _ = model.casting
                _ = model.canRetake
                _ = model.castSpeakers
                _ = model.projectName
                host.layoutSubtreeIfNeeded()
                XCTAssertEqual(try model.prepared.get().speakers.map(\.name), ["Alex", "Sam"])
            }
            let duration = ContinuousClock.now - start
            let elapsed = Double(duration.components.attoseconds) / 1e15
                + Double(duration.components.seconds) * 1000
            if round > 0 { print("UI_AUDIT round=\(round) edits=20 ms=\(elapsed)") }
        }
    }
}

// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import AppKit
import DramaEngine
import SwiftUI
import XCTest
@testable import DramaStudio

@MainActor
final class AppLayoutTests: XCTestCase {
    func testMinimumSizeLayouts() async throws {
        guard let output = ProcessInfo.processInfo.environment["DRAMA_LAYOUT_AUDIT"] else {
            throw XCTSkip("Set DRAMA_LAYOUT_AUDIT to a directory to render the minimum-size layout audit.")
        }
        guard AppHome.override != nil else {
            throw XCTSkip("Set DRAMA_STUDIO_HOME to an isolated directory for the layout audit.")
        }
        let directory = URL(filePath: output)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let scratch = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: scratch) }
        let model = SessionModel(synthesizer: MockSynthesizer(), library: VoiceLibrary(directory: scratch),
                                 pronunciationURL: scratch.appending(path: "pronunciation.json"))
        model.playsWhileGenerating = false
        let mac = FakeMac()
        mac.facts = SetupFacts(termsAccepted: true, weightsReady: true, codecReady: true, tuned: true)
        let setup = SetupCoordinator(services: mac.services)
        let server = APIServerController(session: model)

        try await render(ContentView(model: model, setup: setup).environment(server),
                         name: "studio-empty", size: ContentView.minimumSize, in: directory)
        model.script = "A short line for the window layout audit."
        model.generate()
        for _ in 0..<500 where model.state.isGenerating { try await Task.sleep(for: .milliseconds(10)) }
        XCTAssertFalse(model.state.isGenerating)
        try await render(ContentView(model: model, setup: setup).environment(server),
                         name: "studio-script", size: ContentView.minimumSize, in: directory)
        try await render(TransportBar(model: model).environment(server), name: "transport-narrow", size: CGSize(width: 480, height: 180), in: directory)
        try await render(InspectorView(model: model), name: "inspector-inline", size: CGSize(width: 280, height: 640), in: directory)
        try await render(AddVoiceSheet(model: model), name: "add-voice", size: CGSize(width: 480, height: 560), in: directory)
        try await render(DesignVoiceSheet(model: model, speaker: Speaker(index: 0, name: "Narrator")),
                         name: "design-voice", size: CGSize(width: 520, height: 580), in: directory)
        try await render(SettingsView(server: server, setup: setup),
                         name: "settings", size: CGSize(width: 560, height: 460), in: directory)
        try await render(PronunciationSettingsView(), name: "pronunciation", size: CGSize(width: 560, height: 460), in: directory)
        try await render(HelpWindowView(), name: "help", size: CGSize(width: 600, height: 400), in: directory)
        try await render(InspectorSheet(model: model, dismiss: {}), name: "inspector", size: CGSize(width: 380, height: 500), in: directory)
        model.notice = String(repeating: "A long validation message needs room to wrap. ", count: 6)
        try await render(ContentView(model: model, setup: setup).environment(server), name: "studio-message", size: ContentView.minimumSize, in: directory)
        model.notice = nil
        model.open(NarrationDocument(title: "Layout audit", source: .pptx, segments: [
            .init(index: 1, label: "A section with a long title", text: "Short narration."),
            .init(index: 2, label: "Second section", text: "Another line.")
        ]))
        try await render(ContentView(model: model, setup: setup).environment(server), name: "studio-document", size: ContentView.minimumSize, in: directory)
        try await render(SectionsView(model: model, document: try XCTUnwrap(model.document), playingID: nil), name: "sections-narrow", size: CGSize(width: 200, height: 580), in: directory)
        try await render(TransportBar(model: model).environment(server), name: "transport-document-narrow", size: CGSize(width: 480, height: 200), in: directory)
        let recorder = VoiceRecorder(capture: LayoutCapture(), directory: scratch, requestAccess: { true })
        await recorder.record()
        recorder.stop()
        try await render(RecorderPanel(recorder: recorder, use: { _ in }).padding(10), name: "recorded", size: CGSize(width: 420, height: 140), in: directory)
        let denied = VoiceRecorder(capture: LayoutCapture(), directory: scratch, requestAccess: { false })
        await denied.record()
        try await render(RecorderPanel(recorder: denied, use: { _ in }).padding(10), name: "recorder-denied", size: CGSize(width: 420, height: 160), in: directory)
        let downloadingMac = FakeMac()
        downloadingMac.facts.termsAccepted = true
        downloadingMac.holdWeights = true
        let downloading = SetupCoordinator(services: downloadingMac.services)
        try await render(SetupView(setup: downloading, model: model), name: "setup-download", size: CGSize(width: 600, height: 480), in: directory)
        downloading.pause()
        try await render(SetupView(setup: downloading, model: model), name: "setup-paused", size: CGSize(width: 600, height: 480), in: directory)
        downloadingMac.holdWeights = false
        downloading.resume()
        for _ in 0..<100 where downloading.stage != .firstLine { try await Task.sleep(for: .milliseconds(10)) }
        try await render(SetupView(setup: downloading, model: model), name: "setup-first-line", size: CGSize(width: 600, height: 480), in: directory)
        let freshMac = FakeMac()
        try await render(SetupView(setup: SetupCoordinator(services: freshMac.services), model: model),
                         name: "setup-terms", size: CGSize(width: 600, height: 480), in: directory)
        freshMac.facts.requirementGaps = ["This Mac has 16 GB of memory. Drama Studio requires at least 64 GB."]
        try await render(SetupView(setup: SetupCoordinator(services: freshMac.services), model: model),
                         name: "setup-requirements", size: CGSize(width: 600, height: 480), in: directory)
        let failureMac = FakeMac()
        failureMac.facts.termsAccepted = true
        failureMac.blocker = String(repeating: "The model folder could not be read. Choose another location or try again. ", count: 15)
        try await render(SetupView(setup: SetupCoordinator(services: failureMac.services), model: model), name: "setup-error", size: CGSize(width: 600, height: 480), in: directory)
        let conversionMac = FakeMac()
        conversionMac.facts = SetupFacts(termsAccepted: true, weightsReady: true)
        conversionMac.conversionCommand = String(repeating: "python convert.py --model /a/long/model/path\n", count: 15)
        try await render(SetupView(setup: SetupCoordinator(services: conversionMac.services), model: model), name: "setup-conversion", size: CGSize(width: 600, height: 480), in: directory)
    }

    private func render<V: View>(_ view: V, name: String, size: CGSize, in directory: URL) async throws {
        let host = NSHostingView(rootView: view.frame(width: size.width, height: size.height)
            .background(Color(nsColor: .windowBackgroundColor)))
        host.sizingOptions = []
        let window = NSWindow(contentRect: CGRect(origin: .zero, size: size), styleMask: [.borderless],
                              backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        window.contentView = host
        window.setContentSize(size)
        window.center()
        window.makeKeyAndOrderFront(nil)
        defer { window.close() }
        for _ in 0..<5 {
            host.layoutSubtreeIfNeeded()
            try await Task.sleep(for: .milliseconds(30))
        }
        XCTAssertEqual(host.bounds.size.width, size.width, accuracy: 1, name)
        XCTAssertEqual(host.bounds.size.height, size.height, accuracy: 1, name)
        let capture = Process()
        capture.executableURL = URL(filePath: "/usr/sbin/screencapture")
        capture.arguments = ["-x", "-o", "-l", String(window.windowNumber), directory.appending(path: "\(name).png").path]
        try capture.run()
        capture.waitUntilExit()
        XCTAssertEqual(capture.terminationStatus, 0, name)
    }
}

private final class LayoutCapture: AudioCapturing {
    func start(writingTo url: URL, level: @escaping @MainActor (Float) -> Void) throws {}
    func stop() -> TimeInterval { 8 }
}

// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import AppKit
import DramaEngine
import SwiftUI
import XCTest
@testable import DramaStudio

@Observable @MainActor
private final class Width { var points: CGFloat = 740 }

@MainActor
final class TakeGridLayoutTests: XCTestCase {
    private func settle(_ host: NSView) async throws {
        for _ in 0..<8 {
            host.layoutSubtreeIfNeeded()
            try await Task.sleep(for: .milliseconds(30))
        }
    }

    private func window(_ host: NSView) -> NSWindow {
        let window = NSWindow(contentRect: CGRect(x: 0, y: 0, width: 760, height: 640), styleMask: [.borderless], backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        window.contentView = host
        return window
    }

    /// Catches the grid keeping rows it arranged for another width: a
    /// narrower window must reflow 300 tiles from three columns to two.
    func testTheGridReflowsWhenItsWidthChanges() async throws {
        let width = Width()
        let host = NSHostingView(rootView: ReflowingGrid(width: width))
        let window = window(host)
        defer { window.close() }
        try await settle(host)
        XCTAssertEqual(host.fittingSize.height, 100 * 40 + 99 * 8, "300 tiles in three columns")
        width.points = 480
        try await settle(host)
        XCTAssertEqual(host.fittingSize.height, 150 * 40 + 149 * 8, "300 tiles in two columns")
    }

    /// Catches the grid jumping back to the selected take whenever its size
    /// changes, here because takes were deleted, while the user has scrolled
    /// away to compare other takes. A new column count still brings the
    /// selected take back.
    func testOnlyANewColumnCountScrollsBackToTheSelectedTake() async throws {
        let scratch = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: scratch) }
        let model = SessionModel(synthesizer: MockSynthesizer(pace: .instant), library: VoiceLibrary(directory: scratch),
                                 pronunciationURL: scratch.appending(path: "pronunciation.json"))
        model.playsWhileGenerating = false
        func finish() async throws {
            for _ in 0..<500 where model.state.isGenerating { try await Task.sleep(for: .milliseconds(10)) }
            XCTAssertFalse(model.state.isGenerating)
        }
        model.script = "A short line for comparing takes."
        model.generate()
        try await finish()
        for _ in 0..<23 {
            model.retake(note: "Softer and warmer")
            try await finish()
        }
        let width = Width()
        let host = NSHostingView(rootView: StripHost(model: model, width: width))
        let window = window(host)
        defer { window.close() }
        try await settle(host)
        model.selectTake(try XCTUnwrap(model.visibleTakes.first).id)
        model.stopPreview()
        try await settle(host)
        let scroll = try XCTUnwrap(Self.scrollView(in: host))
        func offset() -> CGFloat { scroll.contentView.bounds.minY }
        XCTAssertLessThan(offset(), 10, "the first take is at the top")

        let bottom = try XCTUnwrap(scroll.documentView).frame.height - scroll.contentView.bounds.height
        scroll.contentView.scroll(to: CGPoint(x: 0, y: bottom))
        scroll.reflectScrolledClipView(scroll.contentView)
        try await settle(host)
        XCTAssertGreaterThan(offset(), 200, "scrolled away to the newest takes")

        for take in model.visibleTakes.suffix(3) { model.deleteTake(take.id) }
        try await settle(host)
        XCTAssertEqual(model.visibleSelectedTakeID, model.visibleTakes.first?.id)
        XCTAssertGreaterThan(offset(), 100, "a row of tiles went, and the grid stayed near where the user left it")

        width.points = 480
        try await settle(host)
        XCTAssertLessThan(offset(), 10, "two columns instead of three bring the selected take back")
    }

    private static func scrollView(in view: NSView) -> NSScrollView? {
        (view as? NSScrollView) ?? view.subviews.lazy.compactMap(scrollView(in:)).first
    }
}

private struct ReflowingGrid: View {
    let width: Width
    var body: some View {
        TakeGrid {
            ForEach(0..<300, id: \.self) { index in
                Text("Take \(index)").frame(maxWidth: .infinity).frame(height: 40)
            }
        }
        .frame(width: width.points)
        .fixedSize(horizontal: false, vertical: true)
    }
}

private struct StripHost: View {
    let model: SessionModel
    let width: Width
    var body: some View {
        EditorOverTakes(model: model) { Color.clear }
            .frame(width: width.points, height: 640)
    }
}

// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import AppKit
import DramaEngine
import SwiftUI
import XCTest
@testable import DramaStudio

/// The take grid at rest shows whole rows, however it came to rest.
@MainActor
final class TakeGridRestTests: XCTestCase {
    private var scratch: URL!

    override func setUp() async throws {
        scratch = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
    }

    override func tearDown() async throws {
        try? FileManager.default.removeItem(at: scratch)
    }

    private let editor = EditorHeight()

    private func strip(_ model: SessionModel, width: CGFloat, height: CGFloat = 640, shown: Bool = false) async throws -> TakeGridWindow {
        let window = TakeGridWindow(EditorOverTakesHost(model: model, editor: editor).frame(width: width, height: height), size: CGSize(width: width, height: height), shown: shown)
        try await window.settle()
        return window
    }

    private func assertWholeRows(_ window: TakeGridWindow, _ step: String, file: StaticString = #filePath, line: UInt = #line) throws {
        let (cut, offset, viewport, content) = (try window.cutEdges(), try window.offset, try window.viewport, try window.content)
        XCTAssertEqual(cut, [], "\(step): offset \(offset), grid \(viewport) of \(content)", file: file, line: line)
    }

    /// Every way of scrolling freely, from the top to the end and back, in two and three columns.
    private func scrollEverywhere(_ model: SessionModel, _ name: String) async throws {
        for width in [520.0, 760] {
            let window = try await strip(model, width: width)
            defer { window.close() }
            let name = "\(name), \(Int(width)) wide"
            try await window.scroller(to: 0)
            try assertWholeRows(window, "\(name), at the top")
            XCTAssertGreaterThan(try window.content, try window.viewport + 1, "\(name): the grid scrolls")

            try await window.wheel(lines: -1)
            XCTAssertGreaterThan(try window.offset, 1, "\(name): one slow turn of the wheel moves the grid")
            try assertWholeRows(window, "\(name), one line by wheel")
            try await window.wheel(lines: -3)
            try assertWholeRows(window, "\(name), three lines by wheel")
            try await window.trackpad(points: [-7, -7, -7, -7, -7])
            try assertWholeRows(window, "\(name), 35 points by trackpad")
            try await window.trackpad(points: [-12, -12, -12], momentum: [-10, -8, -6, -4, -2, -1])
            try assertWholeRows(window, "\(name), a trackpad flick")
            try await window.scroller(to: try window.offset + 23)
            try assertWholeRows(window, "\(name), 23 points by scroller")

            for _ in 0..<12 { try await window.wheel(lines: -40) }
            let end = try window.offset
            try await window.wheel(lines: -40)
            XCTAssertEqual(try window.offset, end, accuracy: 0.5, "\(name): the end is as far as the grid scrolls")
            XCTAssertEqual(try window.content - end, try window.viewport, accuracy: 0.5, "\(name): the end shows the last row")
            try assertWholeRows(window, "\(name), at the end")
            try await window.scroller(to: try window.content)
            try assertWholeRows(window, "\(name), at the end by scroller")

            try await window.wheel(lines: 1)
            XCTAssertLessThan(try window.offset, end - 1, "\(name): one slow turn of the wheel moves the grid back")
            try assertWholeRows(window, "\(name), one line back by wheel")
            try await window.trackpad(points: [9, 9, 9])
            try assertWholeRows(window, "\(name), 27 points back by trackpad")
        }
    }

    func testFreeScrollingRestsOnWholeRowsOfOneHeight() async throws {
        let model = TakeFixture.model(in: scratch)
        try await TakeFixture.plain(9, in: model)
        try await scrollEverywhere(model, "three rows")
        try await TakeFixture.plain(15, in: model)
        try await scrollEverywhere(model, "many rows")
    }

    func testFreeScrollingRestsOnWholeRowsOfMixedHeights() async throws {
        let model = TakeFixture.model(in: scratch)
        try await TakeFixture.mixed(in: model)
        try await scrollEverywhere(model, "mixed rows")
    }

    func testOneAndTwoRowsShowWholeWithNothingToScroll() async throws {
        let model = TakeFixture.model(in: scratch)
        for (takes, rows) in [(2, "one row"), (2, "two rows")] {
            try await TakeFixture.plain(takes, in: model)
            let window = try await strip(model, width: 520)
            defer { window.close() }
            XCTAssertGreaterThanOrEqual(try window.viewport, try window.content - 0.5, rows)
            try assertWholeRows(window, rows)
            try await window.wheel(lines: -3)
            XCTAssertEqual(try window.offset, 0, accuracy: 0.5, "\(rows): nothing to scroll")
        }
    }

    /// The pointer path of the top edge: press, move, release.
    func testDraggingTheTopEdgeRestsOnWholeRows() async throws {
        let model = TakeFixture.model(in: scratch)
        try await TakeFixture.mixed(in: model)
        for (height, name) in [(760.0, "760 tall"), (620, "620 tall")] {
            let window = try await strip(model, width: 520, height: height, shown: true)
            defer { window.close() }
            try await window.scroller(to: 0)
            let automatic = try window.viewport
            func handle() -> CGPoint { CGPoint(x: 260, y: height - editor.points - 4) }
            try assertWholeRows(window, "\(name), before any drag")

            let held = try await window.drag(from: handle(), rise: 50)
            XCTAssertGreaterThan(held, automatic + 20, "\(name): the grid grows with the pointer")
            XCTAssertLessThanOrEqual(held, automatic + 50.5, "\(name): no further than the pointer")
            XCTAssertGreaterThan(try window.viewport, automatic, "\(name): released, the grid keeps more rows")
            try assertWholeRows(window, "\(name), dragged up 50")
            let chosen = try window.viewport

            try await window.wheel(lines: -3)
            try assertWholeRows(window, "\(name), dragged up 50, then scrolled")
            try await window.scroller(to: 0)

            let back = try await window.drag(from: handle(), rise: -20)
            XCTAssertLessThan(back, chosen - 5, "\(name): the grid shrinks with the pointer")
            try assertWholeRows(window, "\(name), dragged down 20")

            try await window.drag(from: handle(), rise: 600)
            XCTAssertGreaterThanOrEqual(editor.points, 219.5, "\(name): the editor keeps its floor")
            XCTAssertGreaterThanOrEqual(try window.viewport, chosen, "\(name): dragged as far as it goes")
            try assertWholeRows(window, "\(name), dragged past the editor's floor")
            try await window.wheel(lines: -3)
            try assertWholeRows(window, "\(name), at the limit, then scrolled")
            try await window.scroller(to: 0)

            try await window.drag(from: handle(), rise: -600)
            XCTAssertEqual(try window.viewport, automatic, accuracy: 0.5, "\(name): dragged back down, the grid is as tall as it started")
            try assertWholeRows(window, "\(name), dragged back down")
        }
    }

    /// The top edge takes the pointer over its eight points and nowhere else.
    func testOnlyTheTopEdgeResizes() async throws {
        let model = TakeFixture.model(in: scratch)
        try await TakeFixture.plain(8, in: model)
        let window = try await strip(model, width: 520, shown: true)
        defer { window.close() }
        let automatic = try window.viewport
        func top() -> CGFloat { 640 - editor.points }
        for (inset, grows) in [(-6.0, false), (1, true), (7, true), (12, false)] {
            let held = try await window.drag(from: CGPoint(x: 260, y: top() - inset), rise: 80)
            XCTAssertEqual(held > automatic + 20, grows, "a drag from \(inset) points under the strip's top edge")
            try await window.drag(from: CGPoint(x: 260, y: top() - 4), rise: -600)
            XCTAssertEqual(try window.viewport, automatic, accuracy: 0.5)
        }
    }
}

@MainActor
private final class EditorHeight {
    var points: CGFloat = 0
}

/// The editor's floor above the strip, as the workspace lays them out.
private struct EditorOverTakesHost: View {
    let model: SessionModel
    let editor: EditorHeight
    var body: some View {
        VStack(spacing: 0) {
            Color.clear.frame(minHeight: 220)
                .onGeometryChange(for: CGFloat.self, of: \.size.height) { editor.points = $0 }
            TakesStrip(model: model).layoutPriority(1)
        }
    }
}

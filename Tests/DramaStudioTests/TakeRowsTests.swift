// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import XCTest
@testable import DramaStudio

final class TakeRowsTests: XCTestCase {
    /// Rows of two tiles with the given heights, 8 points apart, inside a 2-point inset.
    private func rows(_ heights: CGFloat...) -> TakeRows {
        var top: CGFloat = 2
        var tiles: [CGRect] = []
        for height in heights {
            tiles += [CGRect(x: 2, y: top, width: 220, height: height), CGRect(x: 230, y: top, width: 220, height: height - 9)]
            top += height + 8
        }
        return TakeRows(tiles: tiles, inset: 2)
    }

    func testRowsComeFromTheTilesAsPlaced() {
        let grid = rows(64, 73, 73)
        XCTAssertEqual(grid.rows.map(\.top), [0, 72, 153])
        XCTAssertEqual(grid.rows.map(\.bottom), [68, 149, 230], "A row is as tall as its tallest tile, with the inset")
        XCTAssertEqual(grid.rows.map(\.tiles), [0..<2, 2..<4, 4..<6])
        XCTAssertEqual(grid.columns, 2)
        XCTAssertEqual(grid.contentHeight, 230)
        XCTAssertEqual(grid.row(ofTile: 3), 1)
        XCTAssertNil(grid.row(ofTile: 6))
        XCTAssertTrue(TakeRows(tiles: [], inset: 2).isEmpty)
    }

    func testTheHeightHoldsTheWholeRowsThatFitFromTheRestingRow() {
        let grid = rows(64, 64, 73, 98, 98, 64)
        XCTAssertEqual(grid.height(from: 0, within: 160), 140, "Two plain rows")
        XCTAssertEqual(grid.height(from: 1, within: 160), 149, "A plain row and a taller one")
        XCTAssertEqual(grid.height(from: 2, within: 160), 77, "The next row would not fit, so one row shows")
        XCTAssertEqual(grid.height(from: 3, within: 160), 102)
        XCTAssertEqual(grid.height(from: 5, within: 160), 68, "The last row alone")
        XCTAssertEqual(grid.height(from: 0, within: 240), 221, "A larger budget holds a third row")
        XCTAssertEqual(grid.height(from: 3, within: 60), 60, "A row taller than the budget fills it and scrolls")
        XCTAssertEqual(rows(64).height(from: 0, within: 160), 68)
        XCTAssertEqual(TakeRows().height(from: 0, within: 160), 0)
    }

    func testTheGridRestsNoLowerThanTheRowThatShowsTheEnd() {
        let grid = rows(64, 64, 73, 98, 98, 64)
        XCTAssertEqual(grid.lastStop(within: 160), 5, "98 and 64 together pass 160")
        XCTAssertEqual(grid.lastStop(within: 180), 4)
        XCTAssertEqual(grid.lastStop(within: 1000), 0, "Everything fits, so nothing scrolls")
        XCTAssertEqual(grid.lastRest(within: 160), grid.rows[5].top)
        XCTAssertEqual(grid.lastRest(within: 1000), 0)
        XCTAssertEqual(grid.lastRest(within: 40), grid.contentHeight - 40, "A last row taller than the budget scrolls to its end")
    }

    func testScrollingRestsOnARowsTopEdge() {
        let grid = rows(64, 64, 73, 98, 98, 64)
        let tops = grid.rows.map(\.top)
        func rest(_ target: CGFloat, from origin: CGFloat) -> CGFloat { grid.rest(for: target, from: origin, within: 160) }
        XCTAssertEqual(rest(0, from: 0), 0)
        XCTAssertEqual(rest(10, from: 0), tops[1], "One slow turn of a wheel moves a whole row")
        XCTAssertEqual(rest(tops[1] - 10, from: tops[1]), 0, "And back")
        XCTAssertEqual(rest(tops[2] + 60, from: 0), tops[3], "A longer scroll rests on the nearest row")
        XCTAssertEqual(rest(tops[2] + 20, from: 0), tops[2])
        XCTAssertEqual(rest(40, from: 30), tops[1], "A second turn while the first still glides keeps its row")
        XCTAssertEqual(rest(tops[1] + 2, from: tops[1] - 20), tops[1])
        XCTAssertEqual(rest(5000, from: tops[2]), tops[5], "The end is the last row's top edge")
        XCTAssertEqual(rest(-300, from: tops[2]), 0, "Past the top is the top")
        XCTAssertEqual(rest(tops[4] + 3, from: tops[4] + 3), tops[4], "A grid left between rows rests on the nearest")
        XCTAssertEqual(rest(tops[5] - 3, from: tops[5] - 3), tops[5])
    }

    func testARowTallerThanTheGridScrollsWithinItself() {
        let grid = rows(64, 240, 64)
        let tall = grid.rows[1]
        func rest(_ target: CGFloat, from origin: CGFloat) -> CGFloat { grid.rest(for: target, from: origin, within: 160) }
        XCTAssertEqual(rest(tall.top + 30, from: tall.top), tall.top + 30, "Anywhere that keeps the tall row filling the grid")
        XCTAssertEqual(rest(tall.bottom - 160 + 10, from: tall.top + 30), tall.bottom - 160, "Its end, before the next row")
        XCTAssertEqual(rest(tall.bottom - 150, from: tall.bottom - 160), grid.rows[2].top)
        XCTAssertEqual(grid.row(restingAt: tall.top + 30), 1)
        XCTAssertEqual(grid.offset(of: 1, near: tall.top + 30, within: 160), tall.top + 30, "A rest inside the tall row stays")
        XCTAssertEqual(grid.offset(of: 1, near: 0, within: 160), tall.top)
        XCTAssertEqual(grid.offset(of: 2, near: 0, within: 160), grid.rows[2].top)
    }

    func testASelectedRowShowsWholeWithTheLeastScrolling() {
        let grid = rows(64, 64, 73, 98, 98, 64)
        XCTAssertEqual(grid.top(showing: 1, from: 0, within: 160), 0, "Already whole")
        XCTAssertEqual(grid.top(showing: 2, from: 0, within: 160), 1, "One row down brings it in")
        XCTAssertEqual(grid.top(showing: 3, from: 0, within: 160), 3, "A row that fits with no other rests at the top")
        XCTAssertEqual(grid.top(showing: 0, from: 4, within: 160), 0, "A row above rests at the top")
        XCTAssertEqual(grid.top(showing: 5, from: 0, within: 180), 4, "The last row comes with the row that fits above it")
        XCTAssertEqual(grid.top(showing: 1, from: 0, within: 1000), 0)
    }

    func testAHandChosenHeightTakesWholeRowsBelowThenAbove() {
        let grid = rows(64, 64, 73, 98)
        XCTAssertEqual(grid.wholeHeights(around: 0), [68, 140, 221, 327])
        XCTAssertEqual(grid.wholeHeights(around: 2), [77, 183, 255, 327], "Past the last row, the rows above join")
        XCTAssertEqual(TakeRows().wholeHeights(around: 0), [])
    }

    func testRestingStaysCheapWithHundredsOfRows() {
        var top: CGFloat = 2
        let tiles = (0..<900).map { index -> CGRect in
            defer { if index % 3 == 2 { top += (index % 2 == 0 ? 64 : 81) + 8 } }
            return CGRect(x: CGFloat(index % 3) * 228 + 2, y: top, width: 220, height: index % 2 == 0 ? 64 : 81)
        }
        let grid = TakeRows(tiles: tiles, inset: 2)
        XCTAssertEqual(grid.rows.count, 300)
        let start = ContinuousClock.now
        var rests: CGFloat = 0
        for step in 0..<1000 {
            rests += grid.rest(for: CGFloat(step) * 23, from: CGFloat(step) * 17, within: 160)
            rests += grid.height(from: step % 300, within: 160)
        }
        XCTAssertGreaterThan(rests, 0)
        XCTAssertLessThan(ContinuousClock.now - start, .milliseconds(500), "A thousand rests among 300 rows")
    }
}

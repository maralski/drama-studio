// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import DramaEngine
import XCTest
@testable import DramaStudio

final class AppExamplesTests: XCTestCase {
    func testEachOpeningStyleOpensOneExampleLineThatAlsoUsesATag() {
        let styles = PerformanceCue.cues(of: .style).map(\.opening)
        let tags = PerformanceCue.all.filter { $0.kind != .style }.map(\.opening)
        XCTAssertEqual(Examples.lines.count, 8)
        XCTAssertEqual(Set(Examples.lines.compactMap { line in styles.first { line.text.hasPrefix($0) } }), Set(styles), "one line per style")
        for line in Examples.lines {
            XCTAssertTrue(tags.contains { line.text.contains($0) }, "\(line.title) uses a point or passage tag")
        }
    }

    func testEightCastPresetsEachWithASampleLine() {
        XCTAssertEqual(Examples.cast.count, 8)
        XCTAssertEqual(Set(Examples.cast.map(\.id)).count, 8)
        for preset in Examples.cast {
            XCTAssertFalse(DirectorsNote.spokenWords(preset.line).isEmpty, "\(preset.id) has words to speak")
            XCTAssertFalse(preset.design.description.isEmpty)
        }
        let captain = Examples.cast.first { $0.id == "Captain" }
        XCTAssertEqual(captain?.design.scene, "a storm-battered tavern at the edge of the harbor")
        XCTAssertEqual(captain?.line, "[low and menacing] Ye want to know what's out past the reef? [pause] Pray ye never find out.")
        XCTAssertEqual(Examples.notes.count, 9)
    }

    func testExampleLineGoesOnItsOwnLineAtTheCursor() {
        let line = ExampleLine(title: "T", text: "[icy calm] Go.")
        let mid = line.insert(into: "First. Second.", at: 7..<7)
        XCTAssertEqual(mid.text, "First. \n[icy calm] Go.\nSecond.")
        XCTAssertEqual(mid.selection, 22..<22)
        XCTAssertEqual(line.insert(into: "", at: 0..<0).text, "[icy calm] Go.")
        XCTAssertEqual(line.insert(into: "Replace me", at: 0..<10).text, "[icy calm] Go.")
    }
}

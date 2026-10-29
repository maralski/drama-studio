// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import XCTest
@testable import DramaStudio

final class AppCuePaletteTests: XCTestCase {
    func testPaletteCoversTheSpaceTagsInThreeSections() {
        XCTAssertEqual(Set(PerformanceCue.cues(of: .reaction).map(\.opening)), [
            "[laughs]", "[chuckles]", "[sighs]", "[gasps]", "[pause]", "[clears throat]", "[hesitates]", "[coughs]", "[whispers]",
        ])
        XCTAssertEqual(PerformanceCue.cues(of: .passage).map { $0.opening + $0.closing }, [
            "<whisper></whisper>", "<slow></slow>", "<emphasis></emphasis>",
        ])
        XCTAssertTrue(Set(PerformanceCue.cues(of: .style).map(\.opening)).isSuperset(of: [
            "[with bright eager excitement]", "[icy calm]", "[heavy, deflated sadness]", "[sarcastic and deadpan]",
            "[breathless and terrified]", "[furious, barely holding it together]", "[warm, gentle bedtime voice]",
        ]))
        XCTAssertEqual(PerformanceCue.Kind.allCases.flatMap(PerformanceCue.cues(of:)).count, PerformanceCue.all.count, "every cue sits in one section")
        XCTAssertEqual(Set(PerformanceCue.all.map(\.id)).count, PerformanceCue.all.count, "tags are unique")
    }

    func testOpeningStyleInsertsAsABracketCue() {
        let icy = PerformanceCue.all.first { $0.tag == "icy calm" }!
        XCTAssertEqual(icy.insert(into: "Leave.", at: 0..<0).text, "[icy calm] Leave.")
    }

    func testImporterKeepsEveryPassageTag() {
        let markdown = "A <slow>long</slow> and <emphasis>loud</emphasis> <b>bold</b> line."
        XCTAssertEqual(ScriptImporter.plainText(fromMarkdown: markdown), "A <slow>long</slow> and <emphasis>loud</emphasis> bold line.")
    }
}

// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import XCTest
import DramaEngine
@testable import DramaStudio

final class AppCuePaletteTests: XCTestCase {
    func testExportCueReliabilityInventory() throws {
        guard let output = ProcessInfo.processInfo.environment["DRAMA_CUE_AUDIT_MANIFEST"] else {
            throw XCTSkip("Set DRAMA_CUE_AUDIT_MANIFEST to export the actual cue and voice catalogue.")
        }
        let first = "The garden gate is closed."
        let second = "The meeting begins on Monday."
        let plain = first + " " + second
        var cases: [[String: Any]] = []
        for cue in PerformanceCue.all {
            for position in ["opening", "sentence_boundary"] {
                let start = position == "opening" ? 0 : first.count + 1
                let end = cue.form == .paired ? (position == "opening" ? first.count : plain.count) : start
                cases.append(["cue": cue.tag, "kind": cue.kind.rawValue, "position": position,
                              "input": cue.insert(into: plain, at: start..<end).text,
                              "expected": plain])
            }
        }
        let designs: [[String: Any]] = Examples.cast.map { preset in
            var design: [String: Any] = ["name": preset.design.name, "description": preset.design.description]
            if let scene = preset.design.scene { design["scene"] = scene }
            return ["id": preset.design.name, "label": preset.design.name, "parameters": ["character": design], "sample": preset.line]
        }
        let manifest: [String: Any] = ["cases": cases, "designs": designs,
                                      "cues": PerformanceCue.all.map { ["tag": $0.tag, "opening": $0.opening, "closing": $0.closing, "kind": $0.kind.rawValue] }]
        try JSONSerialization.data(withJSONObject: manifest, options: [.prettyPrinted, .sortedKeys])
            .write(to: URL(filePath: output), options: .atomic)
        XCTAssertEqual(cases.count, PerformanceCue.all.count * 2)
    }

    func testEnglishPaletteAndPresetScriptsHaveOneVerifiableLanguage() throws {
        for cue in PerformanceCue.all {
            let line = cue.insert(into: "The garden gate is closed. The meeting begins on Monday.", at: 0..<0).text
            XCTAssertEqual(try ScriptLanguage.identify(line), "en", cue.tag)
        }
        for preset in Examples.cast {
            XCTAssertEqual(try ScriptLanguage.identify(preset.line), "en", preset.design.name)
        }
    }

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

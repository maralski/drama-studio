// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import DramaEngine
import XCTest
@testable import DramaStudio

final class AppScreenplayTests: XCTestCase {
    func testNamesBecomeSpeakerTagsInOrderOfFirstAppearance() throws {
        let script = """
        The rain fell.
        ALICE: Who's there?
        Bob (whispering): Only me.
        alice: Bob! [laughs]
        It's late.
        BOB:
        I know.
        """
        let prepared = try Screenplay.prepare(script, asScreenplay: true).get()
        XCTAssertEqual(prepared.speakers.map(\.name), ["Narrator", "ALICE", "Bob"])
        XCTAssertEqual(prepared.text, """
        <|speaker:0|>The rain fell.
        <|speaker:1|>Who's there?
        <|speaker:2|>Only me.
        <|speaker:1|>Bob! [laughs] It's late.
        <|speaker:2|>I know.
        """)
    }

    func testImportedDialogueIsDetectedButProseIsNot() {
        XCTAssertTrue(Screenplay.looksLikeScreenplay("Night.\nANN: Hi.\nBEN: Hello.\nANN: Bye."))
        XCTAssertFalse(Screenplay.looksLikeScreenplay("Note: one name only.\nNote: still one."))
        XCTAssertFalse(Screenplay.looksLikeScreenplay("A: x\nB: y\nprose\nprose\nprose"))
    }

    func testURLsAndTimesAreNotDialogue() {
        XCTAssertEqual(Screenplay.prepare("See http://fish.audio at 10:30 today.", asScreenplay: true), .failure(.noDialogue))
    }

    func testMoreThanSixteenSpeakersIsRejected() {
        let script = (1...17).map { "Person\($0): Hello." }.joined(separator: "\n")
        guard case .failure(.tooManySpeakers(let names)) = Screenplay.prepare(script, asScreenplay: true) else {
            return XCTFail("expected tooManySpeakers")
        }
        XCTAssertEqual(names.count, 17)
        let sixteen = (1...16).map { "Person\($0): Hello." }.joined(separator: "\n")
        XCTAssertEqual(try Screenplay.prepare(sixteen, asScreenplay: true).get().speakers.count, 16)
    }

    func testPlainModeReadsExistingTagsOrDefaultsToOneNarrator() throws {
        XCTAssertEqual(try Screenplay.prepare("Just prose.", asScreenplay: false).get().speakers.map(\.index), [0])
        let tagged = try Screenplay.prepare("<|speaker:2|>Hi <|speaker:0|>Yo <|speaker:2|>Hey", asScreenplay: false).get()
        XCTAssertEqual(tagged.speakers.map(\.index), [0, 2])
        XCTAssertEqual(Screenplay.prepare("  \n ", asScreenplay: false), .failure(.empty))
    }

    func testCastingSendsReferencesInSpeakerOrderAndFlagsGaps() {
        let a = ReferenceVoice(name: "A", transcript: "a", codes: [[1]])
        let b = ReferenceVoice(name: "B", transcript: "b", codes: [[2]])
        let speakers = (0..<3).map { Speaker(index: $0, name: "S\($0)") }

        let full = Casting(speakers: speakers, assignments: [0: .library(a.id), 1: .library(b.id)], library: [a, b])
        XCTAssertEqual(full.references.map(\.name), ["A", "B"])
        XCTAssertEqual(full.ignored, [])

        let gap = Casting(speakers: speakers, assignments: [0: .library(a.id), 2: .library(b.id)], library: [a, b])
        XCTAssertEqual(gap.references.map(\.name), ["A"])
        XCTAssertEqual(gap.ignored.map(\.speaker.index), [2])
    }

    func testCueInsertionAtCursorAndAroundSelection() {
        let laughs = PerformanceCue.all.first { $0.tag == "laughs" }!
        let at = laughs.insert(into: "Hello world", at: 5..<5)
        XCTAssertEqual(at.text, "Hello [laughs] world")
        XCTAssertEqual(at.selection, 14..<14)

        let whisper = PerformanceCue.all.first { $0.form == .paired }!
        let wrapped = whisper.insert(into: "Say it softly now", at: 4..<13)
        XCTAssertEqual(wrapped.text, "Say <whisper>it softly</whisper> now")
        XCTAssertEqual(Array(wrapped.text)[wrapped.selection].map(String.init).joined(), "it softly")
    }
}

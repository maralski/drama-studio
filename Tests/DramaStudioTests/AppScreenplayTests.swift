// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import DramaEngine
import XCTest
@testable import DramaStudio

final class AppScreenplayTests: XCTestCase {
    func testHeraldAndEngineerKeepTheirCastVoicesAcrossSentences() throws {
        let prepared = try Screenplay.prepare("""
        Herald: [grandly] Hear ye, developers. Your next app has a voice.
        Engineer: [curious] Does the speech run on this Mac?
        Herald: [confident] Speech runs locally on Apple silicon.
        """, asScreenplay: true).get()
        let herald = ReferenceVoice(name: "Herald", transcript: "A herald speaks.", encodings: [.drama3(codes: Array(repeating: [1], count: 10))])
        let newsreader = ReferenceVoice(name: "Newsreader", transcript: "A newsreader speaks.", encodings: [.drama3(codes: Array(repeating: [2], count: 10))])
        let casting = Casting(speakers: prepared.speakers,
                              assignments: [0: .library(herald.id), 1: .library(newsreader.id)], library: [herald, newsreader])
        let plan = try GenerationPlan(SpeechRequest(text: prepared.text, references: casting.references))
        XCTAssertEqual(plan.chunks.map(\.line), [
            "<|speaker:0|> [grandly] Hear ye, developers.",
            "<|speaker:0|> Your next app has a voice.",
            "<|speaker:1|> [curious] Does the speech run on this Mac?",
            "<|speaker:0|> [confident] Speech runs locally on Apple silicon."
        ])
        for chunk in plan.chunks {
            let prompt = try Drama3Prompt.render(text: chunk.turn, references: plan.references)
            XCTAssertTrue(prompt.contains("Text: <|speaker:0|> A herald speaks.\nSpeech: <codes:1>"))
            XCTAssertTrue(prompt.contains("Text: <|speaker:1|> A newsreader speaks.\nSpeech: <codes:1>"))
        }
    }

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

    func testNamesAndDialogueRespectUnicodeAndFormatBoundaries() throws {
        for name in ["Élodie", "E\u{301}lodie", String(repeating: "E\u{301}", count: 40), "李", "O’Neil", "Jean-Luc", String(repeating: "A", count: 40)] {
            let prepared = try Screenplay.prepare("\(name): Hello.\n\(name): Goodbye.", asScreenplay: true).get()
            XCTAssertEqual(prepared.speakers.map(\.name), [name])
            XCTAssertEqual(prepared.text, "<|speaker:0|>Hello.\n<|speaker:0|>Goodbye.")
        }
        for line in ["Alex:Hello.", "1Alex: Hello.", String(repeating: "A", count: 41) + ": Hello."] {
            XCTAssertEqual(Screenplay.prepare(line, asScreenplay: true), .failure(.noDialogue))
        }
    }

    func testNamesWithoutSpokenWordsAreRejected() {
        XCTAssertEqual(Screenplay.prepare("Alex:\nSam:", asScreenplay: true), .failure(.noDialogue))
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
        let a = ReferenceVoice(name: "A", transcript: "a", encodings: [.drama3(codes: [[1]])])
        let b = ReferenceVoice(name: "B", transcript: "b", encodings: [.drama3(codes: [[2]])])
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

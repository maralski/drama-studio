import Foundation
import XCTest
@testable import DramaStudio

final class AppCaptionsTests: XCTestCase {
    private let takes = [
        NarrationTake(segmentID: UUID(), sampleRange: 0..<(30 * 2048), sentences: [
            .init(text: "<|speaker:0|> Hello there.", frames: 10),
            .init(text: "[laughs] How are <whisper>you</whisper>?", frames: 20),
        ]),
        NarrationTake(segmentID: UUID(), sampleRange: 88_200..<(88_200 + 50 * 2048), sentences: [
            .init(text: "<|speaker:1|>", frames: 7),
            .init(text: "Q&A\nnext.", frames: 43),
        ]),
    ]

    func testTimestampsUseCommaForSRTAndDotForWebVTT() {
        XCTAssertEqual(Captions.timestamp(2_749_415, .srt), "00:01:02,345")
        XCTAssertEqual(Captions.timestamp(2_749_415, .webVTT), "00:01:02.345")
        XCTAssertEqual(Captions.timestamp(0, .srt), "00:00:00,000")
        XCTAssertEqual(Captions.timestamp(164_184_476, .webVTT), "01:02:03.004")
        XCTAssertEqual(Captions.timestamp(22_050, .srt), "00:00:00,500")
        XCTAssertEqual(Captions.timestamp(44_078, .srt), "00:00:01,000", "999.501 ms rounds up")
    }

    func testOneCuePerSentenceTimedFromFramesWithTakeOffsets() {
        XCTAssertEqual(Captions.cues(for: takes), [
            .init(start: 0, end: 20_480, text: "Hello there."),
            .init(start: 20_480, end: 61_440, text: "[laughs] How are you?"),
            .init(start: 88_200 + 7 * 2048, end: 88_200 + 50 * 2048, text: "Q&A next."),
        ])
    }

    func testSRTDocument() {
        XCTAssertEqual(Captions.render(takes, as: .srt), """
        1
        00:00:00,000 --> 00:00:00,464
        Hello there.

        2
        00:00:00,464 --> 00:00:01,393
        [laughs] How are you?

        3
        00:00:02,325 --> 00:00:04,322
        Q&A next.

        """)
    }

    func testWebVTTDocumentEscapesAmpersands() {
        XCTAssertEqual(Captions.render(takes, as: .webVTT), """
        WEBVTT

        00:00:00.000 --> 00:00:00.464
        Hello there.

        00:00:00.464 --> 00:00:01.393
        [laughs] How are you?

        00:00:02.325 --> 00:00:04.322
        Q&amp;A next.

        """)
    }

    func testSentencesNeverRunPastTheirTake() {
        let take = NarrationTake(segmentID: UUID(), sampleRange: 1000..<5000, sentences: [.init(text: "Long.", frames: 3), .init(text: "Lost.", frames: 2)])
        XCTAssertEqual(Captions.cues(for: [take]), [.init(start: 1000, end: 5000, text: "Long.")])
        XCTAssertEqual(Captions.render(cues: [], as: .webVTT), "WEBVTT\n")
        XCTAssertEqual(Captions.render(cues: [], as: .srt), "")
    }
}

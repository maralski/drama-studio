// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import DramaEngine
import XCTest

final class SpeechNoticeTests: XCTestCase {
    private func messages(_ text: String, note: String? = nil, expressiveness: Double = GenerationConfig.defaultExpressiveness,
                          on engine: SpeechEngineID = .ming) throws -> [String] {
        let request = SpeechRequest(text: text, note: note, config: .make(expressiveness: expressiveness))
        return try SpeechOutline(request, engine: engine).notices.map { $0.message(on: engine) }
    }

    func testADroppedCueAndAPassageTagSayDifferentThingsAndNameTheEngine() throws {
        XCTAssertEqual(try messages("[pause] Wait. <whisper>Quiet now.</whisper>"), [
            "Ming can't perform [pause], so it was left out.",
            "Ming can't perform <whisper>, so the words inside it are spoken plainly.",
        ])
        XCTAssertEqual(try messages("[laughs nervously] Fine."), ["Ming performs [laughs nervously] as [Laughter]."])
        XCTAssertEqual(try messages("Fine.", note: "slowly"), ["The note \"slowly\" names no emotion Ming performs, so it was not used."])
    }

    func testExpressivenessIsReportedOnlyWhereTheEngineIgnoresIt() throws {
        XCTAssertEqual(try messages("Fine.", expressiveness: 0.9), ["Ming ignores the Expressiveness setting."])
        XCTAssertEqual(try messages("Fine."), [], "the default is no change to report")
        XCTAssertEqual(try messages("Fine.", expressiveness: 0.9, on: .drama3), [])
    }

    func testAOneRecognizerCheckNamesTheLanguageInEnglish() {
        XCTAssertEqual(SpeechNotice.wordsCheckedByOneRecognizer(language: "es", missing: .model).message(on: .ming),
                       "This Mac has no Apple speech model for Spanish, so only Whisper checked the words.")
        XCTAssertEqual(SpeechNotice.wordsCheckedByOneRecognizer(language: "zh-Hans", missing: .model).message(on: .drama3),
                       "This Mac has no Apple speech model for Chinese, so only Whisper checked the words.")
    }

    func testWithoutApplesRecognizerTheNoticeBlamesTheRecognizerNotTheLanguage() {
        XCTAssertEqual(SpeechNotice.wordsCheckedByOneRecognizer(language: "en", missing: .recognizer).message(on: .ming),
                       "Apple speech recognition isn't available on this Mac, so only Whisper checked the words.")
    }
}

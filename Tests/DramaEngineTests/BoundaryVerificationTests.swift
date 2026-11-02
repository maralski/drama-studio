import XCTest
@testable import DramaEngine

final class BoundaryVerificationTests: XCTestCase {
    typealias Recognition = LocalWordVerifier.Recognition
    private let english = Locale(identifier: "en_US")
    private let left = "Wait here."
    private let right = "Please close it quietly."
    private var clean: Recognition {
        .init(text: "Wait here. Please close it quietly.", spans: [
            .init(text: "Wait", confidence: 0.99), .init(text: " here.", confidence: 0.99),
            .init(text: " Please", confidence: 0.189), .init(text: " close", confidence: 0.99),
            .init(text: " it", confidence: 0.99), .init(text: " quietly.", confidence: 0.99)])
    }
    func testOnlyFirstCurrentOccurrenceCanBeEligible() {
        XCTAssertFalse(clean.matches(line: left + " " + right, locale: english))
        XCTAssertEqual(clean.boundaryWordSpan(previousLine: left, currentLine: right, locale: english), 2)
        for index in [0, 1, 3, 4, 5] {
            var recognition = clean
            recognition.spans[2].confidence = 0.99
            recognition.spans[index].confidence = 0.189
            XCTAssertNil(recognition.boundaryWordSpan(previousLine: left, currentLine: right, locale: english))
        }
        var repeated = Recognition(text: "Please wait. Please stay.", spans: [
            .init(text: "Please", confidence: 0.189), .init(text: " wait.", confidence: 0.99),
            .init(text: " Please", confidence: 0.99), .init(text: " stay.", confidence: 0.99)])
        XCTAssertNil(repeated.boundaryWordSpan(previousLine: "Please wait.", currentLine: "Please stay.", locale: english))
        repeated.spans[0].confidence = 0.99; repeated.spans[2].confidence = 0.189
        XCTAssertEqual(repeated.boundaryWordSpan(previousLine: "Please wait.", currentLine: "Please stay.", locale: english), 2)
    }
    func testAmbiguousRequiredReactionWordCannotQualify() {
        let recognition = Recognition(text: "Wait here. He he stays.", spans: [
            .init(text: "Wait here.", confidence: 0.99), .init(text: " He", confidence: 0.189),
            .init(text: " he stays.", confidence: 0.99)])
        XCTAssertNil(recognition.boundaryWordSpan(previousLine: "[laughs] Wait here.", currentLine: "He stays.", locale: english))
    }
    func testAnnotationAndConfidenceAlignmentShareTheSameTokens() {
        let recognition = Recognition(text: "(gasps) Wait here. Please stay.", spans: [
            .init(text: "(gasps)", confidence: 0.1), .init(text: " Wait here.", confidence: 0.99),
            .init(text: " Please", confidence: 0.189), .init(text: " stay.", confidence: 0.99)])
        XCTAssertEqual(recognition.boundaryWordSpan(previousLine: "[gasps] Wait here.", currentLine: "Please stay.", locale: english), 2)
    }

    func testUnprovedConfidenceAndPartitionsCannotQualify() {
        var missing = clean; missing.spans[2].hasReportedConfidence = false
        XCTAssertNil(missing.boundaryWordSpan(previousLine: left, currentLine: right, locale: english))
        var zero = clean; zero.spans[2].confidence = 0
        XCTAssertNil(zero.boundaryWordSpan(previousLine: left, currentLine: right, locale: english))
        var second = clean; second.spans[5].confidence = 0.189
        XCTAssertNil(second.boundaryWordSpan(previousLine: left, currentLine: right, locale: english))
        var multi = clean; multi.spans[2].text = " Please close"; multi.spans.remove(at: 3)
        XCTAssertNil(multi.boundaryWordSpan(previousLine: left, currentLine: right, locale: english))
        var partition = clean; partition.spans[2].text = "Please"
        XCTAssertNil(partition.boundaryWordSpan(previousLine: left, currentLine: right, locale: english))
        let terminal = Recognition(text: "Wait here. Please.", spans: [
            .init(text: "Wait here.", confidence: 0.99), .init(text: " Please.", confidence: 0.189)])
        XCTAssertNil(terminal.boundaryWordSpan(previousLine: left, currentLine: "Please.", locale: english))
    }
    func testReactionsDoNotDiscardRequiredWordsOrChangeTheirPositions() {
        let reaction = Recognition(text: "Ha! Wait here. Please stay.", spans: [
            .init(text: "Ha!", confidence: 0.189), .init(text: " Wait here.", confidence: 0.99),
            .init(text: " Please", confidence: 0.189), .init(text: " stay.", confidence: 0.99)])
        XCTAssertEqual(reaction.boundaryWordSpan(previousLine: "[laughs] Wait here.", currentLine: "Please stay.", locale: english), 2)
        let missing = Recognition(text: "He! Wait here. Stay.", spans: [
            .init(text: "He!", confidence: 0.189), .init(text: " Wait here.", confidence: 0.99),
            .init(text: " Stay.", confidence: 0.99)])
        XCTAssertNil(missing.boundaryWordSpan(previousLine: "[laughs] Wait here.", currentLine: "He stays.", locale: english))
        for heard in ["Wait here. Close it quietly.", "Wait here. Please it quietly.", "Wait here. Please do not close it quietly."] {
            let changed = Recognition(text: heard, spans: [.init(text: heard, confidence: 0.189)])
            XCTAssertNil(changed.boundaryWordSpan(previousLine: left, currentLine: right, locale: english))
        }
    }
    func testExpandedContractionsAndNumbersDoNotInventSpanAlignment() {
        for (right, heard) in [("I'll stay here.", "I'll stay here."), ("24 tickets remain.", "24 tickets remain."),
                               ("Twenty first door is open.", "21st door is open.")] {
            let recognition = Recognition(text: left + " " + heard, spans: [
                .init(text: left, confidence: 0.99), .init(text: " " + heard.components(separatedBy: " ")[0], confidence: 0.189),
                .init(text: " " + heard.components(separatedBy: " ").dropFirst().joined(separator: " "), confidence: 0.99)])
            XCTAssertNil(recognition.boundaryWordSpan(previousLine: left, currentLine: right, locale: english))
        }
    }

    func testFormattingCannotCreateConfidenceSpanAlignment() {
        let recognition = Recognition(text: "Meet at 9.30. Please stay here.", spans: [
            .init(text: "Meet at ", confidence: 0.99), .init(text: "9.30.", confidence: 0.99),
            .init(text: " Please", confidence: 0.189), .init(text: " stay here.", confidence: 0.99)])
        XCTAssertNil(recognition.boundaryWordSpan(previousLine: "Meet at nine thirty.", currentLine: "Please stay here.", locale: english))
        XCTAssertFalse(recognition.matches(line: "Meet at nine thirty. Please stay here.", locale: english))
    }
}

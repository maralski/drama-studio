import XCTest
@testable import DramaEngine

final class BoundaryVerificationAudioTests: XCTestCase {
    private func samples(_ name: String) throws -> [Float] {
        let url = try XCTUnwrap(Bundle.module.url(forResource: name, withExtension: "wav", subdirectory: "Fixtures"))
        return try ReferenceAudio.load(url: url)
    }
    private func verifier(_ text: String) throws -> LocalWordVerifier {
        let config = GenerationConfig()
        return try XCTUnwrap(LocalWordVerifier.prepare(plan: GenerationPlan(text: text, references: [], config: config),
                                                        config: config, shouldContinue: { true }))
    }
    func testNightOwlPairUsesOnlyThePlanBoundExperiment() throws {
        let line = "[gasps] <whisper>The blue window is open. Please close it quietly.</whisper>"
        let checker = try verifier(line)
        let original = try samples("night-owl-boundary")
        var caller = Array(original[..<94_208])
        let left = try XCTUnwrap(checker.checkUnit(samples: caller, at: 0, shouldContinue: { true }))
        caller[0] = 1
        XCTAssertEqual(left.verifiedSamples, Array(original[..<94_208]))
        let right = try XCTUnwrap(checker.checkUnit(samples: Array(original[94_208...]), at: 1, shouldContinue: { true }))
        XCTAssertTrue(try checker.checkAdjacent(left, right, shouldContinue: { true }))
        XCTAssertFalse(try checker.check(samples: original, line: line, shouldContinue: { true }),
                       "Arbitrary complete-file checks must not acquire the generated-boundary exception")
        XCTAssertThrowsError(try checker.checkAdjacent(right, left, shouldContinue: { true }))
        let other = try verifier(line)
        XCTAssertThrowsError(try other.checkAdjacent(left, right, shouldContinue: { true }))
        XCTAssertThrowsError(try checker.checkAdjacent(left, right, shouldContinue: { false }))
        XCTAssertThrowsError(try checker.checkAdjacent(left, right, shouldContinue: { true }),
                             "Cancellation retires the session instead of reviving earlier proofs")
    }
    func testOriginalHumanRejectedCaptainPairStillFails() throws {
        let line = "[low and menacing] Ye want to know what's out past the reef? [pause] Pray ye never find out."
        let checker = try verifier(line)
        let original = try samples("captain-joined-refray")
        let left = try XCTUnwrap(checker.checkUnit(samples: Array(original[..<206_848]), at: 0, shouldContinue: { true }))
        let right = try XCTUnwrap(checker.checkUnit(samples: Array(original[206_848...]), at: 1, shouldContinue: { true }))
        XCTAssertFalse(try checker.checkAdjacent(left, right, shouldContinue: { true }),
                       "Raw proofs cannot excuse the joined refray substitution")
    }
    func testWrongWordAndClippedEndingCannotAcquireRawProofs() throws {
        let garden = try verifier("The garden gate is closed.")
        XCTAssertNil(try garden.checkUnit(samples: samples("golden-instead-of-garden"),
                                         at: 0, shouldContinue: { true }))
        let monday = try verifier("The meeting begins on Monday.")
        let original = try samples("terminal-monday")
        XCTAssertNotNil(try monday.checkUnit(samples: original, at: 0, shouldContinue: { true }))
        XCTAssertNil(try monday.checkUnit(samples: Array(original.prefix(90_846)), at: 0, shouldContinue: { true }))
        XCTAssertNil(try monday.checkUnit(samples: Array(original.prefix(107_604)), at: 0, shouldContinue: { true }))
    }
    func testWrongWordComponentsCannotHideInsideEitherMixedOrder() throws {
        let miaLine = "[laughs] I found a frog in the garden and his name is Mister Hops."
        let gardenLine = "The garden gate is closed."
        let mia = try samples("mia-hops"), golden = try samples("golden-instead-of-garden")
        for goldenFirst in [false, true] {
            let line = goldenFirst ? gardenLine + " " + miaLine : miaLine + " " + gardenLine
            let checker = try verifier(line)
            XCTAssertNil(try checker.checkUnit(samples: golden, at: goldenFirst ? 0 : 1, shouldContinue: { true }))
            XCTAssertFalse(try checker.check(samples: goldenFirst ? golden + mia : mia + golden,
                                             line: line, shouldContinue: { true }))
        }
    }

    func testTextOnlyDiagnosticCannotMintRawProofs() throws {
        let checker = try LocalWordVerifier.prepare(text: "The meeting begins on Monday.", shouldContinue: { true })
        XCTAssertThrowsError(try checker.checkUnit(samples: samples("terminal-monday"),
                                                  at: 0, shouldContinue: { true }))
    }
}

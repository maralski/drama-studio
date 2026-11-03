// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import XCTest
@testable import DramaEngine

final class LocalWordVerifierTests: XCTestCase {
    func testRetainedOrdinalAndClockAudioWithBothRecognizers() throws {
        guard #available(macOS 26, *) else { throw XCTSkip("Native word checking requires macOS 26.") }
        for (fixture, line, wrong) in [
            ("ordinal-first-door", "The first door is open.", "The second door is open."),
            ("clock-nine-thirty", "The meeting begins on Monday at nine thirty.", "The meeting begins on Monday at nine forty five.")
        ] {
            let verifier = try LocalWordVerifier.prepare(text: line, shouldContinue: { true })
            let url = try XCTUnwrap(Bundle.module.url(forResource: fixture, withExtension: "wav", subdirectory: "Fixtures"))
            let samples = try DramaCodec.loadAudio(url: url)
            XCTAssertTrue(try verifier.check(samples: samples, line: line, shouldContinue: { true }), fixture)
            XCTAssertFalse(try verifier.check(samples: samples, line: wrong, shouldContinue: { true }), fixture)
        }
    }

    func testRejectsHumanConfirmedGoldenInsteadOfGarden() throws {
        guard #available(macOS 26, *) else { throw XCTSkip("Native word checking requires macOS 26.") }
        let line = "The garden gate is closed."
        let verifier: LocalWordVerifier
        verifier = try LocalWordVerifier.prepare(text: line, shouldContinue: { true })
        let url = try XCTUnwrap(Bundle.module.url(forResource: "golden-instead-of-garden", withExtension: "wav", subdirectory: "Fixtures"))
        let samples = try DramaCodec.loadAudio(url: url)
        XCTAssertFalse(try verifier.check(samples: samples, line: line, shouldContinue: { true }),
                       "The listener heard golden; a native garden transcript must not certify this recording.")
    }

    func testRetainedEndpointConfidenceDecisions() throws {
        struct Probe: Decodable { var transcript: String; var spans: [Span] }
        struct Span: Decodable { var text: String; var confidence: Double }
        struct Recording: Decodable {
            var key: String; var line: String; var primary: Probe; var confirmation: Probe
            var expectedPrimary: Bool; var expectedEligible: Bool; var expectedCandidate: Bool
        }
        let url = try XCTUnwrap(Bundle.module.url(forResource: "endpoint-spans", withExtension: "json", subdirectory: "Fixtures"))
        let recordings = try JSONDecoder().decode([Recording].self, from: Data(contentsOf: url))
        XCTAssertEqual(recordings.count, 55)
        let locale = Locale(identifier: "en_AU")
        for recording in recordings {
            func recognition(_ probe: Probe) -> LocalWordVerifier.Recognition {
                .init(text: probe.transcript, spans: probe.spans.map { .init(text: $0.text, confidence: $0.confidence) })
            }
            let primary = recognition(recording.primary), confirmation = recognition(recording.confirmation)
            let accepted = primary.matches(line: recording.line, locale: locale)
            let eligible = primary.canConfirmEndOfAudio(line: recording.line, locale: locale)
            XCTAssertEqual(accepted, recording.expectedPrimary, recording.key)
            XCTAssertEqual(eligible, recording.expectedEligible, recording.key)
            XCTAssertEqual(accepted || (eligible && confirmation.matches(line: recording.line, locale: locale, confidenceFloor: 0.9)),
                           recording.expectedCandidate, recording.key)
        }
    }

    func testIndividuallyMatchedSentencesCanFailAfterJoining() throws {
        guard #available(macOS 26, *) else { throw XCTSkip("Native word checking requires macOS 26.") }
        let first = "[low and menacing] Ye want to know what's out past the reef?"
        let second = "[pause] Pray ye never find out."
        let verifier = try LocalWordVerifier.prepare(text: first + " " + second, shouldContinue: { true })
        let url = try XCTUnwrap(Bundle.module.url(forResource: "captain-joined-refray", withExtension: "wav", subdirectory: "Fixtures"))
        let samples = try DramaCodec.loadAudio(url: url)
        XCTAssertTrue(try verifier.check(samples: Array(samples[..<206_848]), line: first, shouldContinue: { true }))
        XCTAssertTrue(try verifier.check(samples: Array(samples[206_848...]), line: second, shouldContinue: { true }))
        XCTAssertFalse(try verifier.check(samples: samples, line: first + " " + second, shouldContinue: { true }))
    }

    func testNativeRecognitionDistinguishesCompleteEndingFromClippedControls() throws {
        guard #available(macOS 26, *) else { throw XCTSkip("Native word checking requires macOS 26.") }
        let line = "The meeting begins on Monday."
        let verifier: LocalWordVerifier
        verifier = try LocalWordVerifier.prepare(text: line, shouldContinue: { true })
        let url = try XCTUnwrap(Bundle.module.url(forResource: "terminal-monday", withExtension: "wav", subdirectory: "Fixtures"))
        let samples = try DramaCodec.loadAudio(url: url)
        XCTAssertTrue(try verifier.check(samples: samples, line: line, shouldContinue: { true }))
        // Fixed sample boundaries from the retained failure: before the final
        // word, and inside its voiced portion. Neither contains the full ending.
        XCTAssertFalse(try verifier.check(samples: Array(samples.prefix(90_846)), line: line, shouldContinue: { true }))
        XCTAssertFalse(try verifier.check(samples: Array(samples.prefix(107_604)), line: line, shouldContinue: { true }))
    }

    func testEndOfAudioConfirmationRequiresOnlyOneUncertainTerminalWord() {
        let locale = Locale(identifier: "en_AU")
        typealias Recognition = LocalWordVerifier.Recognition
        let ending = Recognition(text: "Go on Monday.", spans: [.init(text: "Go on", confidence: 0.99), .init(text: " Monday.", confidence: 0.33)])
        XCTAssertFalse(ending.matches(line: "Go on Monday.", locale: locale))
        XCTAssertTrue(ending.canConfirmEndOfAudio(line: "Go on Monday.", locale: locale))
        XCTAssertFalse(ending.canConfirmEndOfAudio(line: "Do not go on Monday.", locale: locale))
        let earlier = Recognition(text: "Monday, Monday.", spans: [.init(text: "Monday", confidence: 0.3), .init(text: "Monday.", confidence: 0.99)])
        XCTAssertFalse(earlier.canConfirmEndOfAudio(line: "Monday, Monday.", locale: locale))
        let multiple = Recognition(text: "Go on Monday.", spans: [.init(text: "Go", confidence: 0.3), .init(text: "on Monday.", confidence: 0.3)])
        XCTAssertFalse(multiple.canConfirmEndOfAudio(line: "Go on Monday.", locale: locale))
        let partial = Recognition(text: "Go on Monday.", spans: [.init(text: "Go on", confidence: 0.99), .init(text: " Monday.", confidence: 0.709)])
        XCTAssertFalse(partial.matches(line: "Go on Monday.", locale: locale, confidenceFloor: 0.9))
        let confirmed = Recognition(text: "Go on Monday.", spans: [.init(text: "Go on", confidence: 0.99), .init(text: " Monday.", confidence: 0.96)])
        XCTAssertTrue(confirmed.matches(line: "Go on Monday.", locale: locale, confidenceFloor: 0.9))
    }

    func testNativeGateAcceptsInteriorWordsDownToTheFloorButNotAnUncertainEnding() {
        let locale = Locale(identifier: "en_AU")
        func recognition(_ middle: Double, _ last: Double) -> LocalWordVerifier.Recognition {
            .init(text: "Go on Monday.", spans: [.init(text: "Go", confidence: 0.99), .init(text: " on", confidence: middle),
                                                .init(text: " Monday.", confidence: last)])
        }
        XCTAssertTrue(recognition(0.35, 0.99).matches(line: "Go on Monday.", locale: locale))
        XCTAssertFalse(recognition(0.29, 0.99).matches(line: "Go on Monday.", locale: locale))
        XCTAssertFalse(recognition(0.99, 0.45).matches(line: "Go on Monday.", locale: locale))
        XCTAssertTrue(recognition(0.99, 0.45).canConfirmEndOfAudio(line: "Go on Monday.", locale: locale))
    }

    func testEndOfAudioConfirmationAcceptsInteriorWordsDownToTheFloor() {
        let locale = Locale(identifier: "en_AU")
        func recognition(_ middle: Double) -> LocalWordVerifier.Recognition {
            .init(text: "Go on Monday.", spans: [.init(text: "Go", confidence: 0.99), .init(text: " on", confidence: middle),
                                                .init(text: " Monday.", confidence: 0.45)])
        }
        XCTAssertTrue(recognition(0.4).canConfirmEndOfAudio(line: "Go on Monday.", locale: locale))
        XCTAssertFalse(recognition(0.29).canConfirmEndOfAudio(line: "Go on Monday.", locale: locale))
    }

    func testOnlyTheNativeTranscriptHasAppleFormattingForgiven() {
        let locale = Locale(identifier: "en_AU")
        let line = "The train leaves at 1:30 this afternoon."
        let heard = "The train leaves at 130 this afternoon."
        let native = LocalWordVerifier.Recognition(text: heard, spans: [.init(text: heard, confidence: 0.99)])
        XCTAssertTrue(native.matches(line: line, locale: locale))
        XCTAssertFalse(LocalWordVerifier.independentMatches(line: line, transcript: heard, locale: locale))
        XCTAssertTrue(LocalWordVerifier.independentMatches(line: line, transcript: "The train leaves at 1.30 this afternoon.", locale: locale))
    }

    func testTimeoutCancelsTheRecognitionWorker() {
        let stopped = DispatchSemaphore(value: 0)
        XCTAssertThrowsError(try LocalWordVerifier.blocking(timeout: .milliseconds(20), shouldContinue: { true }) {
            defer { stopped.signal() }
            try await Task.sleep(for: .seconds(5))
            return "late recognition"
        }) { error in
            guard case SpeechVerificationError.timedOut = error else { return XCTFail("Unexpected \(error)") }
        }
        XCTAssertEqual(stopped.wait(timeout: .now() + 2), .success)
    }

    func testAlreadyCancelledDoesNotStartRecognition() {
        let stopped = DispatchSemaphore(value: 0)
        XCTAssertThrowsError(try LocalWordVerifier.blocking(shouldContinue: { false }) {
            defer { stopped.signal() }
            try await Task.sleep(for: .seconds(5))
            return "late recognition"
        }) { XCTAssertTrue($0 is CancellationError) }
        XCTAssertEqual(stopped.wait(timeout: .now() + .milliseconds(20)), .timedOut)
    }

    func testRetainedCueRecordingsAgainstCheckedRecognition() throws {
        guard let path = ProcessInfo.processInfo.environment["DRAMA_WORD_CHECK_RECORDINGS"] else {
            throw XCTSkip("Set DRAMA_WORD_CHECK_RECORDINGS to a reviewed local recording manifest.")
        }
        struct Recording: Decodable { var audio: String; var line: String; var matches: Bool }
        let records = try JSONDecoder().decode([Recording].self, from: Data(contentsOf: URL(filePath: path)))
        for record in records {
            let verifier = try LocalWordVerifier.prepare(text: record.line, shouldContinue: { true })
            XCTAssertEqual(try verifier.check(samples: DramaCodec.loadAudio(url: URL(filePath: record.audio)),
                                              line: record.line, shouldContinue: { true }), record.matches, record.audio)
        }
    }
}

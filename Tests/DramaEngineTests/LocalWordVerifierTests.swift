// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Speech
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
            let samples = try ReferenceAudio.load(url: url)
            XCTAssertTrue(try verifier.check(samples: samples, line: line, shouldContinue: { true }), fixture)
            XCTAssertFalse(try verifier.check(samples: samples, line: wrong, shouldContinue: { true }), fixture)
        }
    }

    /// A Mac without Apple's model for the language still checks every word,
    /// with Whisper alone, and the result says so.
    func testWhisperAloneAcceptsTheRightWordsAndRejectsTheWrongOnes() throws {
        guard #available(macOS 26, *) else { throw XCTSkip("Native word checking requires macOS 26.") }
        let line = "The first door is open."
        let independent = try IndependentWordRecognizer(language: "en", deadline: .now + .seconds(30), shouldContinue: { true })
        let verifier = LocalWordVerifier(apple: .missing(.model), language: "en", independent: independent, plannedLines: nil)
        XCTAssertEqual(verifier.notices, [.wordsCheckedByOneRecognizer(language: "en", missing: .model)])
        let url = try XCTUnwrap(Bundle.module.url(forResource: "ordinal-first-door", withExtension: "wav", subdirectory: "Fixtures"))
        let samples = try ReferenceAudio.load(url: url)
        XCTAssertTrue(try verifier.check(samples: samples, line: line, shouldContinue: { true }))
        XCTAssertFalse(try verifier.check(samples: samples, line: "The second door is open.", shouldContinue: { true }))
    }

    /// The Whisper-only path must not swallow English on a Mac that has
    /// Apple's English model: both recognizers keep checking it.
    func testEnglishWithAnInstalledAppleModelUsesBothRecognizers() async throws {
        guard #available(macOS 26, *) else { throw XCTSkip("Native word checking requires macOS 26.") }
        let installed = await SpeechTranscriber.installedLocales
        guard SpeechTranscriber.isAvailable, installed.contains(where: { $0.language.languageCode == .english }) else {
            throw XCTSkip("This Mac has no Apple English speech model installed.")
        }
        let verifier = try LocalWordVerifier.prepare(text: "The first door is open.", shouldContinue: { true })
        XCTAssertEqual(verifier.notices, [], "no one-recognizer notice, so Apple's model is in the check")
        let notices = try await LocalWordVerifier.notices(text: "The first door is open.", language: "en", config: GenerationConfig())
        XCTAssertEqual(notices, [])
    }

    func testRejectsHumanConfirmedGoldenInsteadOfGarden() throws {
        guard #available(macOS 26, *) else { throw XCTSkip("Native word checking requires macOS 26.") }
        let line = "The garden gate is closed."
        let verifier: LocalWordVerifier
        verifier = try LocalWordVerifier.prepare(text: line, shouldContinue: { true })
        let url = try XCTUnwrap(Bundle.module.url(forResource: "golden-instead-of-garden", withExtension: "wav", subdirectory: "Fixtures"))
        let samples = try ReferenceAudio.load(url: url)
        XCTAssertFalse(try verifier.check(samples: samples, line: line, shouldContinue: { true }),
                       "The listener heard golden; a native garden transcript must not certify this recording.")
    }

    /// Apple's spans for 11 Drama 3 takes, complete and cut inside their final
    /// word, heard raw (`primary`) and with 0.75 s of silence appended
    /// (`endpoint`), as the check hears a take that ends in silence.
    func testRetainedEndpointConfidenceDecisions() throws {
        struct Probe: Decodable { var transcript: String; var spans: [Span] }
        struct Span: Decodable { var text: String; var confidence: Double }
        struct Recording: Decodable {
            var key: String; var line: String; var primary: Probe; var endpoint: Probe
            var expectedPrimary: Bool; var expectedEndpoint: Bool
        }
        let url = try XCTUnwrap(Bundle.module.url(forResource: "endpoint-spans", withExtension: "json", subdirectory: "Fixtures"))
        let recordings = try JSONDecoder().decode([Recording].self, from: Data(contentsOf: url))
        XCTAssertEqual(recordings.count, 55)
        let locale = Locale(identifier: "en_AU")
        for recording in recordings {
            func recognition(_ probe: Probe) -> LocalWordVerifier.Recognition {
                .init(text: probe.transcript, spans: probe.spans.map { .init(text: $0.text, confidence: $0.confidence) })
            }
            XCTAssertEqual(recognition(recording.primary).matches(line: recording.line, locale: locale), recording.expectedPrimary, recording.key)
            XCTAssertEqual(recognition(recording.endpoint).matches(line: recording.line, locale: locale), recording.expectedEndpoint, recording.key)
        }
    }

    func testIndividuallyMatchedSentencesCanFailAfterJoining() throws {
        guard #available(macOS 26, *) else { throw XCTSkip("Native word checking requires macOS 26.") }
        let first = "[low and menacing] Ye want to know what's out past the reef?"
        let second = "[pause] Pray ye never find out."
        let verifier = try LocalWordVerifier.prepare(text: first + " " + second, shouldContinue: { true })
        let url = try XCTUnwrap(Bundle.module.url(forResource: "captain-joined-refray", withExtension: "wav", subdirectory: "Fixtures"))
        let samples = try ReferenceAudio.load(url: url)
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
        let samples = try ReferenceAudio.load(url: url)
        XCTAssertTrue(try verifier.check(samples: samples, line: line, shouldContinue: { true }))
        // Fixed sample boundaries from the retained failure: before the final
        // word, and inside its voiced portion. Neither contains the full ending.
        XCTAssertFalse(try verifier.check(samples: Array(samples.prefix(90_846)), line: line, shouldContinue: { true }))
        XCTAssertFalse(try verifier.check(samples: Array(samples.prefix(107_604)), line: line, shouldContinue: { true }))
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
        XCTAssertTrue(recognition(0.99, 0.5).matches(line: "Go on Monday.", locale: locale))
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
            XCTAssertEqual(try verifier.check(samples: ReferenceAudio.load(url: URL(filePath: record.audio)),
                                              line: record.line, shouldContinue: { true }), record.matches, record.audio)
        }
    }
}

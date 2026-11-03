// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import XCTest
@testable import DramaEngine

/// Real recognizer output for generated clips: `pos` and `spell` clips speak
/// the line, `neg` and `negspell` clips speak a different line, and `drop`
/// clips leave a word out.
final class WordCheckCorpusTests: XCTestCase {
    private struct Clip: Decodable {
        struct Span: Decodable { var text: String; var confidence: Double }
        var id: String; var line: String; var native: String; var nativeSpans: [Span]; var independent: String
        var speaksLine: Bool { ["pos", "spell"].contains(id.split(separator: ".")[1]) }
        var recognition: LocalWordVerifier.Recognition {
            .init(text: native, spans: nativeSpans.map { .init(text: $0.text, confidence: $0.confidence) })
        }
    }
    private let locale = Locale(identifier: "en_AU")

    private func corpus(_ name: String = "word-check-corpus") throws -> [Clip] {
        let url = try XCTUnwrap(Bundle.module.url(forResource: name, withExtension: "json", subdirectory: "Fixtures"))
        return try JSONDecoder().decode([Clip].self, from: Data(contentsOf: url))
    }

    private func accepted(_ clip: Clip) -> Bool {
        clip.recognition.matches(line: clip.line, locale: locale)
            && LocalWordVerifier.independentMatches(line: clip.line, transcript: clip.independent, locale: locale)
    }

    func testCorpusAcceptsCorrectSpeechAndRejectsEveryWrongClip() throws {
        let clips = try corpus()
        XCTAssertEqual(clips.count, 200)
        let wrongAccepted = clips.filter { !$0.speaksLine && accepted($0) }.map(\.id)
        XCTAssertEqual(wrongAccepted, [])
        let correct = clips.filter(\.speaksLine)
        let correctAccepted = correct.filter(accepted).count
        print("word-check corpus: \(correctAccepted)/\(correct.count) correct clips accepted")
        XCTAssertGreaterThanOrEqual(correctAccepted, 132)
    }

    /// Lines with every kind of cue Drama Studio offers. About a dozen correct
    /// clips performed only the cue's sound or cut the line short.
    func testCuedLinesAcceptCorrectSpeechAndRejectEveryWrongClip() throws {
        let clips = try corpus("word-check-cues")
        XCTAssertEqual(clips.count, 132)
        let wrongAccepted = clips.filter { !$0.speaksLine && accepted($0) }.map(\.id)
        XCTAssertEqual(wrongAccepted, [])
        let correct = clips.filter(\.speaksLine)
        let correctAccepted = correct.filter(accepted).count
        print("word-check cues: \(correctAccepted)/\(correct.count) correct clips accepted")
        XCTAssertGreaterThanOrEqual(correctAccepted, 76)
    }

    func testMatcherSpeed() throws {
        guard ProcessInfo.processInfo.environment["DRAMA_WORD_CHECK_BENCH"] != nil else {
            throw XCTSkip("Set DRAMA_WORD_CHECK_BENCH=1 to time the matcher.")
        }
        let clips = try corpus()
        try WordCoverage.prepare()
        _ = clips.map(accepted)
        let rounds = 20
        var calls = 0
        let start = ContinuousClock.now
        for _ in 0..<rounds {
            for clip in clips {
                _ = WordCoverage.matches(line: clip.line, transcript: clip.native, locale: locale, profile: .apple)
                _ = WordCoverage.matches(line: clip.line, transcript: clip.independent, locale: locale, profile: .exact)
                calls += 2
            }
        }
        let elapsed = ContinuousClock.now - start
        let micros = Double(elapsed.components.attoseconds) / 1e12 + Double(elapsed.components.seconds) * 1e6
        print(String(format: "word-check matcher: %d calls, %.1f µs per call", calls, micros / Double(calls)))
    }
}

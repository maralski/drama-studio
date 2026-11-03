// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import XCTest
@testable import DramaEngine

/// Real recognizer output for generated clips, one fixture per set listed in
/// word-check-sets.json. `pos` and `spell` clips speak the line, `neg` and
/// `negspell` clips speak a different line, and `drop` clips leave a word out.
/// tools/word_check_eval/README.md explains how the sets were made.
final class WordCheckCorpusTests: XCTestCase {
    private struct ClipSet: Decodable {
        var name: String; var clips: Int; var replayFloor: Int
    }
    private struct Clip: Decodable {
        struct Span: Decodable { var text: String; var confidence: Double }
        var id: String; var line: String; var native: String; var nativeSpans: [Span]; var independent: String
        var speaksLine: Bool { ["pos", "spell"].contains(id.split(separator: ".")[1]) }
        var recognition: LocalWordVerifier.Recognition {
            .init(text: native, spans: nativeSpans.map { .init(text: $0.text, confidence: $0.confidence) })
        }
    }
    private let locale = Locale(identifier: "en_AU")

    private func fixture<T: Decodable>(_ name: String) throws -> T {
        let url = try XCTUnwrap(Bundle.module.url(forResource: name, withExtension: "json", subdirectory: "Fixtures"))
        return try JSONDecoder().decode(T.self, from: Data(contentsOf: url))
    }

    private func accepted(_ clip: Clip) -> Bool {
        clip.recognition.matches(line: clip.line, locale: locale)
            && LocalWordVerifier.independentMatches(line: clip.line, transcript: clip.independent, locale: locale)
    }

    /// The replay cannot confirm the end of the audio, so its floors sit a
    /// little below what the full check accepts on the same clips.
    func testEverySetAcceptsCorrectSpeechAndRejectsEveryWrongClip() throws {
        let sets: [ClipSet] = try fixture("word-check-sets")
        XCTAssertFalse(sets.isEmpty)
        for set in sets {
            let clips: [Clip] = try fixture("word-check-\(set.name)")
            XCTAssertEqual(clips.count, set.clips, "\(set.name) clip count")
            let wrongAccepted = clips.filter { !$0.speaksLine && accepted($0) }.map(\.id)
            XCTAssertEqual(wrongAccepted, [], "\(set.name) accepted wrong clips")
            let correct = clips.filter(\.speaksLine)
            let correctAccepted = correct.filter(accepted).count
            print("word-check \(set.name): \(correctAccepted)/\(correct.count) correct, "
                  + "\(wrongAccepted.count)/\(clips.count - correct.count) wrong accepted")
            XCTAssertGreaterThanOrEqual(correctAccepted, set.replayFloor, "\(set.name) correct clips accepted")
        }
    }

    func testMatcherSpeed() throws {
        guard ProcessInfo.processInfo.environment["DRAMA_WORD_CHECK_BENCH"] != nil else {
            throw XCTSkip("Set DRAMA_WORD_CHECK_BENCH=1 to time the matcher.")
        }
        let clips: [Clip] = try fixture("word-check-tuning")
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

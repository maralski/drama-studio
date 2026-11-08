// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import DramaEngine
import XCTest
@testable import DramaStudio

/// The frozen word-check reliability harness for the Ming engine. One command
/// runs a fixed corpus (six voices, eight lines, three seeds) through the
/// engine registry the app's session uses, counts the attempts each sentence
/// took and prints the metric: first-attempt word-check pass rate, hard
/// failures, mean attempts, real-time factor including retries, and the error
/// count. Every take and every sentence land as wavs in `MING_HILL_OUT`, with
/// `sentences.tsv` and `lines.tsv`. Run under tools/with-gpu-lock, release:
///
///     DRAMA_APP_CHECK_HOME=<home> MING_WEIGHTS=<checkpoint> MING_HILL_OUT=<dir> \
///       swift test -c release -Xswiftc -enable-testing --filter MingReliabilityHarness
///
/// `MING_HILL_VOICES`, `MING_HILL_SEEDS` and `MING_HILL_LINES` (comma lists)
/// narrow a probe run; the frozen metric is the full corpus.
@MainActor
final class MingReliabilityHarness: XCTestCase {
    static let lines = [
        "The lamp was lit before the fog rolled in.",
        "Fog swallowed the harbor by midnight, and nobody on the pier noticed.",
        "Captain Reyes logged 47 crates at pier 12 before dawn.",
        "[laughs] You should have seen his face.",
        "[sighs] Then it was over, and we went home.",
        "Well, [laughs] that was not the plan, was it?",
        "Nobody answered the door. So I knocked again, louder this time. The hallway light flickered twice and went out.",
        "You want to know what happened? Then sit down and listen.",
    ]
    static let seeds: [UInt64] = [11, 22, 33]
    static let clipTranscript = "The lighthouse keeper climbed the stairs every evening to light the lamp."
    static let design = VoiceDesign(name: "Design", description: "gravelly old male voice, weathered and slow")

    enum Source {
        case library(ReferenceVoice.ID)
        case designed
        case builtIn
    }

    struct Voice {
        var name: String
        var source: Source
    }

    struct SentenceRow {
        var voice: String
        var seed: UInt64
        var line: Int
        var sentence: Int
        var text: String
        var attempts: Int
        var hardFailed: Bool
        var audioSeconds: Double
        var firstPass: Bool { attempts == 1 && !hardFailed }
    }

    struct LineRow {
        var voice: String
        var seed: UInt64
        var line: Int
        var sentences: Int
        var steps: Int
        var audioSeconds: Double
        var wallSeconds: Double
        var decodeSeconds: Double
        var verifySeconds: Double
        var outcome: String
        /// When the line started and finished, so audit records (which carry only a time) can be attributed to it.
        var started: Date
        var finished: Date
    }

    func testCorpus() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let path = environment["DRAMA_APP_CHECK_HOME"] else { throw XCTSkip("set DRAMA_APP_CHECK_HOME") }
        guard let mingPath = environment["MING_WEIGHTS"], case let ming = MingLocation(directory: URL(filePath: mingPath)), ming.isComplete else {
            throw XCTSkip("set MING_WEIGHTS to a complete Ming checkpoint")
        }
        guard let outPath = environment["MING_HILL_OUT"] else { throw XCTSkip("set MING_HILL_OUT") }
        let home = URL(filePath: path, directoryHint: .isDirectory)
        let out = URL(filePath: outPath, directoryHint: .isDirectory)
        let work = out.appending(path: "work")
        try? FileManager.default.removeItem(at: work)
        try FileManager.default.createDirectory(at: work.appending(path: "Voices"), withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: out.appending(path: "wav"), withIntermediateDirectories: true)
        print("HILL bundle \(Bundle.main.bundleIdentifier ?? "nil") audit \(environment["DS_WORD_CHECK_AUDIT_DIR"] ?? "off")")

        let voicesDirectory = work.appending(path: "Voices")
        for file in try FileManager.default.contentsOfDirectory(at: home.appending(path: "Voices"), includingPropertiesForKeys: nil)
            where file.pathExtension == "json" {
            try FileManager.default.copyItem(at: file, to: voicesDirectory.appending(path: file.lastPathComponent))
        }
        let library = VoiceLibrary(directory: voicesDirectory, codecDirectory: home.appending(path: "Codec"))
        let engines = try SpeechEngines(selected: .ming, factories: [.ming: .ming(location: ming)], voices: library)
        let clipped = try await engines.cloneVoice(audioURL: home.appending(path: "clip.aiff"), transcript: Self.clipTranscript, name: "Clip")
        try library.add(clipped.voice, clip: clipped.clip)

        var voices: [Voice] = library.voices.sorted { $0.name < $1.name }.map { Voice(name: $0.name, source: .library($0.id)) }
        voices += [Voice(name: "Design", source: .designed), Voice(name: "Default", source: .builtIn)]
        let wanted = Self.list("MING_HILL_VOICES")
        if !wanted.isEmpty { voices = voices.filter { wanted.contains($0.name) } }
        let seeds = Self.list("MING_HILL_SEEDS").compactMap { UInt64($0) }.ifEmpty(Self.seeds)
        let lineIndices = Self.list("MING_HILL_LINES").compactMap { Int($0) }.ifEmpty(Array(Self.lines.indices))

        var sentenceRows: [SentenceRow] = []
        var lineRows: [LineRow] = []
        var errors: [String] = []
        let corpusStart = Date()
        for voice in voices {
            for seed in seeds {
                var sessionVoice: Take?
                for lineIndex in lineIndices {
                    let line = Self.lines[lineIndex]
                    var config = GenerationConfig.make(expressiveness: GenerationConfig.defaultExpressiveness)
                    config.streaming = .defaultLowLatency
                    config.seed = seed
                    var references: [ReferenceVoice] = []
                    var character: VoiceDesign?
                    switch voice.source {
                    case .library(let id): references = [try XCTUnwrap(library.voice(id))]
                    case .designed: character = Self.design
                    case .builtIn: break
                    }
                    let request = SpeechRequest(text: line, references: references, character: character,
                                                previousTakes: sessionVoice.map { [$0] } ?? [], config: config)
                    let planned = try SpeechOutline(request, engine: .ming).lines
                    let tag = "\(voice.name)-s\(seed)-l\(lineIndex)"
                    var retries: [Int: Int] = [:]
                    var result: SpeechResult?
                    var outcome = "ok"
                    var failedSentence: Int?
                    let start = Date()
                    do {
                        for try await event in engines.synthesize(request) {
                            switch event {
                            case .retry(let index, _): retries[index, default: 0] += 1
                            case .finished(let finished): result = finished
                            default: break
                            }
                        }
                    } catch SpeechVerificationError.exhausted(let sentence, let attempts) {
                        outcome = "hard-fail sentence \(sentence) after \(attempts) attempts"
                        failedSentence = sentence - 1
                    } catch {
                        outcome = "error: \(error)"
                        errors.append("\(tag): \(error)")
                    }
                    let wall = Date().timeIntervalSince(start)
                    let spoken = result?.sentences.map(\.samples) ?? []
                    for (index, text) in planned.enumerated() {
                        if let failedSentence, index > failedSentence { break }
                        if result == nil, failedSentence == nil, outcome != "ok" { break }
                        let seconds = index < spoken.count ? Double(spoken[index]) / Double(ModelConstants.sampleRate) : 0
                        sentenceRows.append(SentenceRow(voice: voice.name, seed: seed, line: lineIndex, sentence: index, text: text,
                                                        attempts: retries[index, default: 0] + 1, hardFailed: failedSentence == index,
                                                        audioSeconds: seconds))
                    }
                    let timings = result?.timings ?? SpeechTimings()
                    let audioSeconds = Double(result?.samples.count ?? 0) / Double(ModelConstants.sampleRate)
                    lineRows.append(LineRow(voice: voice.name, seed: seed, line: lineIndex, sentences: planned.count, steps: timings.framesGenerated,
                                            audioSeconds: audioSeconds, wallSeconds: wall, decodeSeconds: timings.decodeSeconds,
                                            verifySeconds: timings.verificationSeconds, outcome: outcome, started: start, finished: Date()))
                    print("HILL \(tag): \(outcome), attempts \(planned.indices.map { retries[$0, default: 0] + 1 }), "
                          + "\(timings.framesGenerated) steps, \(String(format: "%.2f", audioSeconds)) s audio in \(String(format: "%.2f", wall)) s")
                    if let result {
                        try ReferenceAudio.write(result.samples, to: out.appending(path: "wav/\(tag).wav"))
                        var cursor = 0
                        for (index, count) in spoken.enumerated() {
                            try ReferenceAudio.write(Array(result.samples[cursor ..< cursor + count]), to: out.appending(path: "wav/\(tag)-\(index).wav"))
                            cursor += count
                        }
                        if sessionVoice == nil, voice.source.isSessionVoice { sessionVoice = result.take }
                    }
                }
            }
        }
        let corpusWall = Date().timeIntervalSince(corpusStart)
        try Self.write(sentences: sentenceRows, lines: lineRows, to: out)
        Self.summarize(sentences: sentenceRows, lines: lineRows, errors: errors, voices: voices.map(\.name), corpusWall: corpusWall)
    }

    static func summarize(sentences: [SentenceRow], lines: [LineRow], errors: [String], voices: [String], corpusWall: Double) {
        func rate(_ rows: [SentenceRow]) -> String {
            rows.isEmpty ? "n/a" : String(format: "%.1f%% (%d/%d)", 100 * Double(rows.filter(\.firstPass).count) / Double(rows.count),
                                          rows.filter(\.firstPass).count, rows.count)
        }
        let hard = sentences.filter(\.hardFailed).count
        let attempts = sentences.isEmpty ? 0 : Double(sentences.map(\.attempts).reduce(0, +)) / Double(sentences.count)
        let audio = lines.map(\.audioSeconds).reduce(0, +)
        let wall = lines.map(\.wallSeconds).reduce(0, +)
        print("HILL-SUMMARY sentences \(sentences.count) lines \(lines.count) errors \(errors.count) hard-failures \(hard)")
        print("HILL-SUMMARY first-attempt-pass \(rate(sentences)) mean-attempts \(String(format: "%.2f", attempts)) "
              + "rtf \(String(format: "%.2f", audio > 0 ? wall / audio : 0)) audio \(String(format: "%.1f", audio)) s wall \(String(format: "%.1f", wall)) s "
              + "corpus-wall \(String(format: "%.1f", corpusWall)) s")
        for voice in voices {
            let rows = sentences.filter { $0.voice == voice }
            print("HILL-SUMMARY voice \(voice) first-attempt-pass \(rate(rows)) hard-failures \(rows.filter(\.hardFailed).count) "
                  + "mean-attempts \(String(format: "%.2f", rows.isEmpty ? 0 : Double(rows.map(\.attempts).reduce(0, +)) / Double(rows.count)))")
        }
        let cued = sentences.filter { $0.text.contains("[") }
        print("HILL-SUMMARY cue-lines first-attempt-pass \(rate(cued)) plain-lines first-attempt-pass \(rate(sentences.filter { !$0.text.contains("[") }))")
        for error in errors { print("HILL-ERROR \(error)") }
    }

    static func write(sentences: [SentenceRow], lines: [LineRow], to out: URL) throws {
        var text = "voice\tseed\tline\tsentence\tattempts\thard_failed\tfirst_pass\taudio_seconds\ttext\n"
        for row in sentences {
            text += "\(row.voice)\t\(row.seed)\t\(row.line)\t\(row.sentence)\t\(row.attempts)\t\(row.hardFailed)\t\(row.firstPass)\t"
                + "\(String(format: "%.3f", row.audioSeconds))\t\(row.text)\n"
        }
        try text.write(to: out.appending(path: "sentences.tsv"), atomically: true, encoding: .utf8)
        let clock = ISO8601DateFormatter()
        clock.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        var lineText = "voice\tseed\tline\tsentences\tsteps\taudio_seconds\twall_seconds\tdecode_seconds\tverify_seconds\toutcome\tstarted\tfinished\n"
        for row in lines {
            lineText += "\(row.voice)\t\(row.seed)\t\(row.line)\t\(row.sentences)\t\(row.steps)\t\(String(format: "%.3f", row.audioSeconds))\t"
                + "\(String(format: "%.3f", row.wallSeconds))\t\(String(format: "%.3f", row.decodeSeconds))\t\(String(format: "%.3f", row.verifySeconds))\t\(row.outcome)\t"
                + "\(clock.string(from: row.started))\t\(clock.string(from: row.finished))\n"
        }
        try lineText.write(to: out.appending(path: "lines.tsv"), atomically: true, encoding: .utf8)
    }

    static func list(_ variable: String) -> [String] {
        (ProcessInfo.processInfo.environment[variable] ?? "").split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
    }
}

private extension MingReliabilityHarness.Source {
    /// A designed or built-in voice speaks once, then later lines clone that take, as the session does.
    var isSessionVoice: Bool {
        switch self {
        case .library: false
        case .designed, .builtIn: true
        }
    }
}

private extension Array {
    func ifEmpty(_ fallback: [Element]) -> [Element] { isEmpty ? fallback : self }
}

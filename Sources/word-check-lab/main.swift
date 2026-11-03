// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
// Probe tool for the spoken-word check. Not part of the product.
// tools/word_check_eval/README.md explains the evaluation built on it.
//
//   word-check-lab gen <items.json> <outdir>          unchecked generation, one WAV per item
//   word-check-lab recog <items.json> <dir>           native + Whisper transcripts and the production decision
//   word-check-lab decide <items.json> <dir> <out>    the production decision and its time per item
//   word-check-lab tokens                             stdin: JSON [String] -> production tokens per string
//   word-check-lab match                              stdin: JSON [[line, transcript, ("apple")?]] -> en_US, en_AU
//   word-check-lab corpus <fixture.json>              replay a recorded fixture through the matcher
//
// An item's voice is a name; gen reads its library voice JSON from
// WORD_CHECK_VOICE_<NAME>, for example WORD_CHECK_VOICE_NEWSREADER.
import AVFoundation
import DramaEngine
import Foundation
import Speech

struct Item: Codable {
    var id: String
    var text: String        // what is spoken
    var check: String?      // what the audio is checked against (defaults to text)
    var seed: UInt64?
    var voice: String?      // named library voice, resolved by voicePath
    var audio: String?      // reuse another item's WAV instead of generating
}

func readItems(_ path: String) throws -> [Item] {
    try JSONDecoder().decode([Item].self, from: Data(contentsOf: URL(fileURLWithPath: path)))
}

func blocking<T: Sendable>(_ work: @escaping @Sendable () async throws -> T) throws -> T {
    let box = LockedBox<Result<T, Error>>()
    let sem = DispatchSemaphore(value: 0)
    Task.detached {
        do { box.set(.success(try await work())) } catch { box.set(.failure(error)) }
        sem.signal()
    }
    sem.wait()
    return try box.get()!.get()
}

final class LockedBox<T>: @unchecked Sendable {
    private var value: T?
    private let lock = NSLock()
    func set(_ v: T) { lock.lock(); value = v; lock.unlock() }
    func get() -> T? { lock.lock(); defer { lock.unlock() }; return value }
}

func log(_ s: String) { FileHandle.standardError.write((s + "\n").data(using: .utf8)!) }

func voicePath(_ name: String) -> String {
    let variable = "WORD_CHECK_VOICE_" + name.uppercased()
    guard let path = ProcessInfo.processInfo.environment[variable] else {
        log("set \(variable) to the voice JSON exported from the Drama Studio library"); exit(1)
    }
    return path
}

/// The recorded clip for an item: a WAV from gen, or a FLAC from the release asset.
func audioURL(_ item: Item, in dir: URL) -> URL? {
    ["wav", "flac"].lazy.map { dir.appendingPathComponent((item.audio ?? item.id) + "." + $0) }
        .first { FileManager.default.fileExists(atPath: $0.path) }
}

// MARK: - gen

func gen(_ items: [Item], out: URL) throws {
    let voices = Dictionary(uniqueKeysWithValues: Set(items.compactMap(\.voice)).map { ($0, voicePath($0)) })
    guard let loc = ModelLocation.huggingFaceCache() ?? {
        let own = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Application Support/DramaStudio/Models/drama-3-e0a9218")
        return FileManager.default.fileExists(atPath: own.path) ? ModelLocation(directory: own) : nil
    }() else { log("no model"); exit(1) }
    let model = try DramaModel(location: loc, precision: .exact, capacity: DramaModel.defaultCapacity, schedule: nil)
    let tokenizer = try Tokenizer(url: model.location.directory.appendingPathComponent("tokenizer.json"))
    let gen = FrameGenerator(model: model, tokenizer: tokenizer)
    let codec = try blocking { try await DramaCodec(directory: DramaCodec.defaultDirectory) }
    gen.codec = { codes, window in try DramaSynthesizer.decodeBlocking(codec, codes: codes, window: window) }
    for item in items where item.audio == nil {
        let wav = out.appendingPathComponent(item.id + ".wav")
        if FileManager.default.fileExists(atPath: wav.path) { continue }
        var refs: [Drama3Prompt.Reference] = []
        if let voice = item.voice {
            struct V: Decodable { var transcript: String; var codes: [[Int32]] }
            let v = try JSONDecoder().decode(V.self, from: Data(contentsOf: URL(fileURLWithPath: voices[voice]!)))
            refs.append(.init(transcript: v.transcript, codes: v.codes))
        }
        var config = GenerationConfig()
        config.wordVerification = .uncheckedPreview
        config.chunking = .sentences
        config.minSpokenRatio = 0          // raw attempts: no duration-based retries
        if let seed = item.seed { config.seed = seed }
        var audio: [Float] = []
        let t0 = Date()
        _ = try gen.generate(text: item.text, references: refs, config: config) { event in
            if case .chunkFinished(let r) = event { audio += r.samples }
        }
        try PCM16.wav(audio).write(to: wav)
        log(String(format: "%@ %.1fs audio in %.1fs", item.id, Double(audio.count) / 44_100, Date().timeIntervalSince(t0)))
    }
}

// MARK: - recog

@available(macOS 26, *)
func nativeTranscript(_ url: URL, locale: Locale) async throws -> (String, [[String: Any]]) {
    let file = try AVAudioFile(forReading: url)
    let transcriber = SpeechTranscriber(locale: locale, transcriptionOptions: [], reportingOptions: [],
                                        attributeOptions: [.transcriptionConfidence])
    let analyzer = SpeechAnalyzer(modules: [transcriber])
    let reader = Task { () -> (String, [[String: Any]]) in
        var text = ""
        var spans: [[String: Any]] = []
        for try await result in transcriber.results {
            text += String(result.text.characters)
            for run in result.text.runs {
                spans.append(["text": String(result.text[run.range].characters),
                              "confidence": Double(run.transcriptionConfidence ?? -1)])
            }
        }
        return (text, spans)
    }
    _ = try await analyzer.analyzeSequence(from: file)
    try await analyzer.finalizeAndFinishThroughEndOfInput()
    return try await reader.value
}

func recog(_ items: [Item], dir: URL) throws {
    guard #available(macOS 26, *) else { log("needs macOS 26"); exit(1) }
    let locale = try blocking { () async throws -> Locale in
        let installed = await SpeechTranscriber.installedLocales
        let requested = ScriptLanguage.preferredLocale(language: "en", installed: installed, preferences: Locale.preferredLanguages)
            ?? Locale(identifier: "en")
        return await SpeechTranscriber.supportedLocale(equivalentTo: requested) ?? requested
    }
    log("native locale: \(locale.identifier)")
    let deadline = { ContinuousClock.now + .seconds(120) }
    let whisper = try IndependentWordRecognizer(language: "en", deadline: deadline(), shouldContinue: { true })
    var results: [[String: Any]] = []
    for item in items {
        guard let wav = audioURL(item, in: dir) else { continue }
        let line = item.check ?? item.text
        let samples = try DramaCodec.loadAudio(url: wav)
        let (native, spans) = try blocking { try await nativeTranscript(wav, locale: locale) }
        let independent = try whisper.transcript(pcm16: PCM16.encode(samples), deadline: deadline(), shouldContinue: { true })
        let verifier = try LocalWordVerifier.prepare(text: line, shouldContinue: { true })
        let checkStart = ContinuousClock.now
        let accepted = try verifier.check(samples: samples, line: line, shouldContinue: { true })
        let checkSeconds = (ContinuousClock.now - checkStart) / .seconds(1)
        results.append(["id": item.id, "line": line, "native": native, "native_spans": spans,
                         "whisper_small_en": independent, "production_accepted": accepted, "check_seconds": checkSeconds,
                         "seconds": Double(samples.count) / 44_100])
        log("\(item.id) accepted=\(accepted) native=\(native) | whisper=\(independent)")
    }
    let data = try JSONSerialization.data(withJSONObject: results, options: [.prettyPrinted])
    try data.write(to: dir.appendingPathComponent("recog.json"))
}

// MARK: - decide

func decide(_ items: [Item], dir: URL, out: URL) throws {
    var results: [[String: Any]] = []
    for item in items {
        guard let wav = audioURL(item, in: dir) else { continue }
        let line = item.check ?? item.text
        let samples = try DramaCodec.loadAudio(url: wav)
        let verifier = try LocalWordVerifier.prepare(text: line, shouldContinue: { true })
        let start = ContinuousClock.now
        let accepted = try verifier.check(samples: samples, line: line, shouldContinue: { true })
        let seconds = (ContinuousClock.now - start) / .seconds(1)
        results.append(["id": item.id, "accepted": accepted, "check_seconds": seconds])
    }
    try JSONSerialization.data(withJSONObject: results, options: [.prettyPrinted]).write(to: out)
}

// MARK: - tokens

func tokens() throws {
    let input = FileHandle.standardInput.readDataToEndOfFile()
    let texts = try JSONDecoder().decode([String].self, from: input)
    let locale = Locale(identifier: "en_US")
    try WordCoverage.prepare()
    let out = texts.map { WordCoverage.expectedTokens($0, locale: locale) }
    FileHandle.standardOutput.write(try JSONEncoder().encode(out))
}

// stdin: JSON [[line, transcript, ("apple")?]] -> [Bool], production matcher, en_US and en_AU
func matchPairs() throws {
    let pairs = try JSONDecoder().decode([[String]].self, from: FileHandle.standardInput.readDataToEndOfFile())
    try WordCoverage.prepare()
    let out = pairs.map { p in ["en_US", "en_AU"].map {
        WordCoverage.matches(line: p[0], transcript: p[1], locale: Locale(identifier: $0),
                             profile: p.count > 2 && p[2] == "apple" ? .apple : .exact)
    } }
    FileHandle.standardOutput.write(try JSONEncoder().encode(out))
}

// corpus fixture -> ids the matcher accepts (native at Apple profile with the 0.3/0.5 floors, independent exact)
func corpus(_ path: String) throws {
    struct Span: Decodable { var text: String; var confidence: Double }
    struct Clip: Decodable { var id: String; var line: String; var native: String; var nativeSpans: [Span]; var independent: String }
    let clips = try JSONDecoder().decode([Clip].self, from: Data(contentsOf: URL(fileURLWithPath: path)))
    try WordCoverage.prepare()
    let locale = Locale(identifier: "en_AU")
    for clip in clips {
        let last = clip.nativeSpans.lastIndex { $0.text.contains { $0.isLetter || $0.isNumber } }
        let uncertain = clip.nativeSpans.indices.filter { clip.nativeSpans[$0].confidence < ($0 == last ? 0.5 : 0.3) }.map { clip.nativeSpans[$0].text }
        let native = WordCoverage.matches(line: clip.line, transcript: clip.native, locale: locale, profile: .apple, uncertainSpans: uncertain)
        let independent = WordCoverage.matches(line: clip.line, transcript: clip.independent, locale: locale, profile: .exact)
        print(clip.id, native ? "N" : "-", independent ? "W" : "-")
    }
}

let args = CommandLine.arguments
switch args.count > 1 ? args[1] : "" {
case "gen": try gen(readItems(args[2]), out: URL(fileURLWithPath: args[3]))
case "recog": try recog(readItems(args[2]), dir: URL(fileURLWithPath: args[3]))
case "tokens": try tokens()
case "match": try matchPairs()
case "corpus": try corpus(args[2])
case "decide": try decide(readItems(args[2]), dir: URL(fileURLWithPath: args[3]), out: URL(fileURLWithPath: args[4]))
default: log("usage: word-check-lab gen|recog|decide|tokens|match|corpus"); exit(1)
}

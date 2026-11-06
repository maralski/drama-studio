// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import XCTest
import CryptoKit
@testable import DramaEngine

/// The app's engine path end to end: text in, streamed events and 44.1 kHz
/// audio out, through the Metal engine and the Core ML codec.
final class SynthesizerIntegrationTests: XCTestCase {
    func testJoinedCaptainCannotCompleteWhenWholeRecordingLosesWords() async throws {
        guard let loc = ModelLocation.huggingFaceCache(), loc.isComplete else { throw XCTSkip("drama-3 not downloaded") }
        guard DramaCodec.modelsExist() else { throw XCTSkip("codec models not converted") }
        let tuning = FileManager.default.temporaryDirectory.appendingPathComponent("drama-join-tuning-\(UUID().uuidString)")
        addTeardownBlock { try? FileManager.default.removeItem(at: tuning) }
        let synth = DramaSynthesizer(location: loc, codecDirectory: DramaCodec.defaultDirectory, tuningDirectory: tuning)
        var config = GenerationConfig()
        config.seed = 1
        config.maxRetries = 0
        let request = SpeechRequest(
            text: "[low and menacing] Ye want to know what's out past the reef? [pause] Pray ye never find out.",
            character: Self.captainJoinVoice,
            note: "Speak slowly with careful clear diction", config: config)
        var completed = false
        var chunks: [[Float]] = []
        var failure: Error?
        do {
            for try await event in synth.synthesize(request) {
                switch event {
                case .audio(let samples, _): chunks.append(samples)
                case .finished: completed = true
                default: break
                }
            }
        } catch { failure = error }
        XCTAssertEqual(chunks.count, 1, "The rejected second packet must not expose its bad join to playback")
        let digest = SHA256.hash(data: PCM16.encode(chunks.flatMap { $0 })).map { String(format: "%02x", $0) }.joined()
        XCTAssertEqual(digest, "e5c8d722e5163eeeaf35299c94b48621e0e754987d6f5c247a941f44ee8e3976",
                       "The accepted first sentence remains byte-identical to the retained control")
        XCTAssertFalse(completed, "Individually accepted sentences must not certify the joined recording")
        XCTAssertEqual(failure?.localizedDescription,
                       "Sentence 2 did not pass the spoken-word check after 1 attempts. Generation stopped without creating a completed take.")
    }

    func testDefaultCaptainRecoversWithoutChangingTheScript() async throws {
        guard let loc = ModelLocation.huggingFaceCache(), loc.isComplete else { throw XCTSkip("drama-3 not downloaded") }
        guard DramaCodec.modelsExist() else { throw XCTSkip("codec models not converted") }
        let tuning = FileManager.default.temporaryDirectory.appendingPathComponent("drama-captain-tuning-\(UUID().uuidString)")
        addTeardownBlock { try? FileManager.default.removeItem(at: tuning) }
        let synth = DramaSynthesizer(location: loc, codecDirectory: DramaCodec.defaultDirectory, tuningDirectory: tuning)
        var config = GenerationConfig()
        config.seed = 1
        let line = "[low and menacing] Ye want to know what's out past the reef? [pause] Pray ye never find out."
        var delivered: [Float] = []
        var stream = StreamedAudio()
        var completed: SpeechResult?
        for try await event in synth.synthesize(SpeechRequest(text: line, character: Self.captainJoinVoice, config: config)) {
            delivered += stream.newSamples(from: event)
            if case .finished(let result) = event { completed = result }
        }
        let result = try XCTUnwrap(completed)
        XCTAssertEqual(result.wordCoverage, .recognized)
        XCTAssertEqual(result.samples, delivered)
        XCTAssertEqual(result.samples.count, result.take.sampleCount)
        XCTAssertEqual(result.sentences.map(\.text), [
            "[low and menacing] Ye want to know what's out past the reef?", "[pause] Pray ye never find out."
        ])
        XCTAssertEqual(result.take.text, "Captain: " + line)
        XCTAssertGreaterThan(result.timings.framesGenerated, result.take.dramaCodes[0].count,
                             "The known rejected first attempt must be replaced before completion")
    }

    func testDefaultMiaRecoversWithoutChangingTheScript() async throws {
        guard let loc = ModelLocation.huggingFaceCache(), loc.isComplete else { throw XCTSkip("drama-3 not downloaded") }
        guard DramaCodec.modelsExist() else { throw XCTSkip("codec models not converted") }
        let tuning = FileManager.default.temporaryDirectory.appendingPathComponent("drama-mia-tuning-\(UUID().uuidString)")
        addTeardownBlock { try? FileManager.default.removeItem(at: tuning) }
        let synth = DramaSynthesizer(location: loc, codecDirectory: DramaCodec.defaultDirectory, tuningDirectory: tuning)
        var config = GenerationConfig()
        config.seed = 1
        let line = "[with bright eager excitement] Guess what, guess what! [laughs] I found a frog in the garden and his name is Mister Hops!"
        let voice = VoiceDesign(name: "Mia", description: "excited little girl, about seven, talks fast, giggly and breathless")
        var delivered: [Float] = []
        var stream = StreamedAudio()
        var completed: SpeechResult?
        for try await event in synth.synthesize(SpeechRequest(text: line, character: voice, config: config)) {
            delivered += stream.newSamples(from: event)
            if case .finished(let result) = event { completed = result }
        }
        let result = try XCTUnwrap(completed)
        XCTAssertEqual(result.wordCoverage, .recognized)
        XCTAssertEqual(result.samples, delivered)
        XCTAssertEqual(result.samples.count, result.take.sampleCount)
        XCTAssertEqual(result.sentences.map(\.text), [
            "[with bright eager excitement] Guess what, guess what!",
            "[laughs] I found a frog in the garden and his name is Mister Hops!"
        ])
        XCTAssertEqual(result.take.text, "Mia: " + line)
        XCTAssertGreaterThan(result.timings.framesGenerated, result.take.dramaCodes[0].count)
    }

    private static let captainJoinVoice = VoiceDesign(
        name: "Captain", description: "gravelly old male voice, about seventy, weathered and slow, salt in every word",
        scene: "a storm-battered tavern at the edge of the harbor")

    func testShortSentenceProducesSpeech() async throws {
        guard let loc = ModelLocation.huggingFaceCache(), loc.isComplete else { throw XCTSkip("drama-3 not downloaded") }
        guard DramaCodec.modelsExist() else { throw XCTSkip("codec models not converted") }
        // A temp tuning directory keeps the test from writing the user's cache
        // with a debug-process measurement; the first load tunes into it.
        let tuning = FileManager.default.temporaryDirectory.appendingPathComponent("drama-tuning-\(UUID().uuidString)")
        addTeardownBlock { try? FileManager.default.removeItem(at: tuning) }
        let synth = DramaSynthesizer(location: loc, codecDirectory: DramaCodec.defaultDirectory, tuningDirectory: tuning)
        var config = GenerationConfig()
        config.seed = 2
        let request = SpeechRequest(text: "Hello, my name is Drama and I live inside a computer.", config: config)
        var audioChunks = 0
        var frames = 0
        var result: SpeechResult?
        for try await event in synth.synthesize(request) {
            switch event {
            case .audio(let samples, _): audioChunks += 1; XCTAssertGreaterThan(samples.count, 44_100)
            case .frame: frames += 1
            case .finished(let r): result = r
            default: break
            }
        }
        let r = try XCTUnwrap(result)
        XCTAssertEqual(audioChunks, 1)
        XCTAssertEqual(r.take.dramaCodes[0].count, frames)
        XCTAssertEqual(r.samples.count, r.take.sampleCount)
        XCTAssertEqual(r.sentences, [SpokenSentence(text: request.text, samples: frames * 2048)])
        let rms = (r.samples.reduce(0) { $0 + $1 * $1 } / Float(r.samples.count)).squareRoot()
        XCTAssertGreaterThan(rms, 0.02, "decoded audio should contain speech")
        XCTAssertGreaterThan(Double(r.take.dramaCodes[0].count) * 2048 / 44100, 2.0)
        print("integration: \(r.take.dramaCodes[0].count) frames, \(String(format: "%.1f", Double(r.samples.count) / 44100))s audio, rms \(rms), \(String(format: "%.1f", r.timings.framesPerSecond)) frames/s, codec \(String(format: "%.2f", r.timings.codecSeconds))s")
    }

    /// Low latency: each chunk's partials, in order, are exactly its audio, and
    /// a chunk's audio is complete before the next chunk's first partial.
    func testLowLatencyStreamsEachChunkInOrder() async throws {
        guard let loc = ModelLocation.huggingFaceCache(), loc.isComplete else { throw XCTSkip("drama-3 not downloaded") }
        guard DramaCodec.modelsExist() else { throw XCTSkip("codec models not converted") }
        let tuning = FileManager.default.temporaryDirectory.appendingPathComponent("drama-tuning-\(UUID().uuidString)")
        addTeardownBlock { try? FileManager.default.removeItem(at: tuning) }
        let synth = DramaSynthesizer(location: loc, codecDirectory: DramaCodec.defaultDirectory, tuningDirectory: tuning)
        var config = GenerationConfig()
        config.seed = 5
        config.streaming = .defaultLowLatency
        config.maxRetries = 0
        config.wordVerification = .uncheckedPreview
        let request = SpeechRequest(text: "The storm had passed by morning, leaving the harbor strangely quiet. Old Tom walked the pier, counting the boats that had survived the night.", config: config)
        var partials: [Int: [Float]] = [:]
        var finals: [Int: [Float]] = [:]
        var order: [String] = []
        var result: SpeechResult?
        for try await event in synth.synthesize(request) {
            switch event {
            case .partialAudio(let samples, let chunk, let frames):
                XCTAssertEqual(frames.lowerBound, (partials[chunk]?.count ?? 0) / ModelConstants.samplesPerFrame, "chunk \(chunk): contiguous")
                partials[chunk, default: []] += samples
                if order.last != "partial \(chunk)" { order.append("partial \(chunk)") }
            case .audio(let samples, let chunk):
                finals[chunk] = samples
                order.append("audio \(chunk)")
            case .finished(let r): result = r
            default: break
            }
        }
        XCTAssertEqual(order, ["partial 0", "audio 0", "partial 1", "audio 1"])
        XCTAssertEqual(partials, finals)
        XCTAssertEqual(try XCTUnwrap(result).samples, finals[0]! + finals[1]!)
    }
}

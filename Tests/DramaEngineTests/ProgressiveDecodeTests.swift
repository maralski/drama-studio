// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation
import XCTest
@testable import DramaEngine

final class ProgressiveDecodeTests: XCTestCase {
    /// A causal stand-in for the codec whose output, like a real decoder's,
    /// depends on how much left context the call was given.
    private static func contextSensitiveDecode(_ codes: [[Int32]], _ w: DecodeWindow) -> [Float] {
        var state = 0.0
        var out: [Float] = []
        for t in w.start ..< w.end {
            state = state * 0.99 + Double(codes[0][t] % 97) + Double(codes[3][t] % 13) * 0.1
            guard t >= w.keep else { continue }
            out += (0 ..< ModelConstants.samplesPerFrame).map { Float(sin(state + Double($0) * 0.01)) }
        }
        return out
    }

    private static func frames(_ count: Int) -> [Frame] {
        (0 ..< count).map { (t: Int) -> Frame in
            let codes: [Int32] = (0 ..< ModelConstants.numCodebooks).map { (book: Int) -> Int32 in
                let mixed: Int = t * 31 + book * 17 + t * t
                return Int32(mixed % 1024)
            }
            return [codes[0] + ModelConstants.semanticStart] + codes
        }
    }

    private static func codes(_ frames: [Frame]) -> [[Int32]] {
        (0 ..< ModelConstants.numCodebooks).map { c in frames.map { $0[1 + c] } }
    }

    func testCoveringReusesTheFullDecodeWindows() {
        XCTAssertEqual(DecodeWindow.covering(256 ..< 320), [DecodeWindow(start: 0, keep: 256, end: 320)])
        XCTAssertEqual(DecodeWindow.covering(448 ..< 576), [
            DecodeWindow(start: 0, keep: 448, end: 512),
            DecodeWindow(start: 256, keep: 512, end: 576),
        ])
        XCTAssertEqual(DecodeWindow.covering(0 ..< 0), [])
    }

    func testPiecesJoinIntoTheFullChunkDecode() throws {
        let all = Self.frames(600)
        let full = DecodeWindow.windows(frames: all.count).flatMap { Self.contextSensitiveDecode(Self.codes(all), $0) }
        for steps in [[64], [37, 100, 1, 300, 162], [600]] {
            let decoder = ChunkDecoder(Self.contextSensitiveDecode)
            var landed = 0
            var pieces: [ChunkDecoder.Piece] = []
            var cursor = 0
            while landed < all.count {
                landed = min(all.count, landed + steps[cursor % steps.count])
                cursor += 1
                _ = try decoder.advance(to: Array(all.prefix(landed)))
                pieces += decoder.takePieces()
            }
            XCTAssertEqual(pieces.map(\.frames.lowerBound), [0] + pieces.dropLast().map(\.frames.upperBound), "steps \(steps): pieces are contiguous")
            XCTAssertEqual(pieces.last?.frames.upperBound, all.count)
            XCTAssertEqual(pieces.flatMap(\.samples), full, "steps \(steps): partials join into the full decode")
            XCTAssertEqual(decoder.samples, full)
        }
    }

    func testAdvanceReportsTheTrailingSilenceOfTheDecodedAudio() throws {
        let silentTail: ChunkDecoder.Decode = { _, w in
            (w.keep ..< w.end).flatMap { t in [Float](repeating: t < 100 ? 0.5 : 0, count: ModelConstants.samplesPerFrame) }
        }
        let decoder = ChunkDecoder(silentTail)
        let all = Self.frames(186)
        XCTAssertEqual(try decoder.advance(to: Array(all.prefix(100))), 0)
        let tail = try decoder.advance(to: all)
        XCTAssertEqual(tail, Double(86 * ModelConstants.samplesPerFrame) / Double(ModelConstants.sampleRate), accuracy: 1e-9)
    }

    func testMockStreamsPartialsThatJoinIntoEachChunk() async throws {
        var config = GenerationConfig()
        config.seed = 1
        config.streaming = .lowLatency(stepFrames: 16)
        config.maxRetries = 0
        let request = SpeechRequest(text: "The first sentence runs a while. The second one is here too.", config: config)
        var partials: [Int: [Float]] = [:]
        var finals: [Int: [Float]] = [:]
        var ranges: [Range<Int>] = []
        for try await event in MockSynthesizer(pace: .instant).synthesize(request) {
            switch event {
            case .partialAudio(let samples, let chunk, let range):
                partials[chunk, default: []] += samples
                ranges.append(range)
                XCTAssertEqual(samples.count, range.count * ModelConstants.samplesPerFrame)
            case .audio(let samples, let chunk): finals[chunk] = samples
            default: break
            }
        }
        XCTAssertFalse(finals.isEmpty)
        XCTAssertEqual(partials, finals)
        XCTAssertTrue(ranges.contains { $0.count == 16 })
    }

    func testSentenceModeSendsNoPartials() async throws {
        var config = GenerationConfig()
        config.seed = 1
        let request = SpeechRequest(text: "Only whole sentences arrive here.", config: config)
        for try await event in MockSynthesizer(pace: .instant).synthesize(request) {
            if case .partialAudio = event { XCTFail("sentence mode streamed a partial") }
        }
    }

    /// `DRAMA_CODEC_SEAM=1` runs the seam on the Core ML codec.
    func testSeamOnTheRealCodec() async throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["DRAMA_CODEC_SEAM"] == "1", "set DRAMA_CODEC_SEAM=1 to decode on the Core ML codec")
        try XCTSkipUnless(DramaCodec.modelsExist(), "Core ML codec models not converted")
        let codec = try await DramaCodec(directory: DramaCodec.defaultDirectory)
        struct Fixture: Decodable { let codes: [[Int32]] }
        let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Fixtures/codec_codes.json")
        let clip = try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: url)).codes
        let length = 600
        let codes = clip.map { row in (0 ..< length).map { row[$0 % row.count] } }
        let all: [Frame] = (0 ..< length).map { t in [codes[0][t] + ModelConstants.semanticStart] + codes.map { $0[t] } }
        let full = try await codec.decode(codes: codes)

        let decoder = ChunkDecoder { codes, window in try DramaSynthesizer.decodeBlocking(codec, codes: codes, window: window) }
        var joined: [Float] = []
        for landed in stride(from: 64, through: length + 63, by: 64).map({ min($0, length) }) {
            _ = try decoder.advance(to: Array(all.prefix(landed)))
            joined += decoder.takePieces().flatMap(\.samples)
        }
        let aligned = zip(joined, full).map { abs($0 - $1) }.max() ?? .infinity

        var bare: [Float] = []
        for landed in stride(from: 64, through: length + 63, by: 64).map({ min($0, length) }) {
            let new = min(64, landed - bare.count / ModelConstants.samplesPerFrame)
            let start = max(0, landed - DramaCodec.contextFrames - new)
            let window = DecodeWindow(start: start, keep: landed - new, end: landed)
            bare += try await codec.decode(codes: codes, window: window)
        }
        let bareError = zip(bare, full).map { abs($0 - $1) }.max() ?? .infinity
        print(String(format: "codec seam over %d frames in 64-frame steps: max abs error %.3g with the full decode's windows, %.3g with a bare 256-frame context", length, aligned, bareError))
        XCTAssertEqual(joined.count, full.count)
        XCTAssertLessThan(aligned, 1e-3)
    }
}

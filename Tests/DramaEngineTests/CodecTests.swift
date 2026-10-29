// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import AVFoundation
import Foundation
import XCTest
@testable import DramaEngine

final class CodecTests: XCTestCase {
    private struct CodesFixture: Decodable {
        let codes: [[Int32]]
        let reencoded_codebook0: [Int32]
        let tone_codes: [[Int32]]
    }

    private struct AudioFixture: Decodable {
        let samples: [Float]
    }

    private static let fixtures = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        .appendingPathComponent("Fixtures", isDirectory: true)

    private func load<T: Decodable>(_ name: String) throws -> T {
        try JSONDecoder().decode(T.self, from: Data(contentsOf: Self.fixtures.appendingPathComponent(name)))
    }

    private func makeCodec() async throws -> DramaCodec {
        let dir = DramaCodec.defaultDirectory
        try XCTSkipUnless(DramaCodec.modelsExist(in: dir),
                          "Core ML codec models not found in \(dir.path); run Sources/DramaEngine/CodecConversion/convert_codec.py to create them")
        return try await DramaCodec(directory: dir)
    }

    func testDecodeMatchesReferenceAndReencodes() async throws {
        let codec = try await makeCodec()
        let fixture: CodesFixture = try load("codec_codes.json")
        let reference: AudioFixture = try load("codec_audio_ref.json")
        let frames = fixture.codes[0].count

        let audio = try await codec.decode(codes: fixture.codes)
        XCTAssertEqual(audio.count, frames * DramaCodec.samplesPerFrame)

        let head = Array(audio.prefix(reference.samples.count))
        var signal = 0.0, noise = 0.0
        for (r, x) in zip(reference.samples, head) {
            signal += Double(r) * Double(r)
            noise += Double(r - x) * Double(r - x)
        }
        let snr = 10 * log10(signal / max(noise, .leastNonzeroMagnitude))
        XCTAssertGreaterThan(snr, 30, "decoded audio SNR \(snr) dB against the PyTorch reference")

        let codes = try await codec.encode(samples: audio)
        XCTAssertEqual(codes.count, DramaCodec.numCodebooks)
        XCTAssertEqual(codes[0].count, frames)
        let matches = zip(codes[0], fixture.reencoded_codebook0).filter { $0 == $1 }.count
        let rate = Double(matches) / Double(frames)
        // Codebook 0 flips easily: PyTorch's own decode-then-encode keeps only 86% of the
        // fixture codes, and the fp16 decoder's 35 dB error moves the re-encode further
        // (72% measured). Exact-input encoder parity is checked on the tone at 90%.
        XCTAssertGreaterThanOrEqual(rate, 0.6, "codebook 0 re-encode agreement with PyTorch \(rate)")
    }

    func testWindowsCoverEveryFrameOnce() {
        typealias W = DecodeWindow
        XCTAssertEqual(DecodeWindow.windows(frames: 0), [])
        XCTAssertEqual(DecodeWindow.windows(frames: 300), [W(start: 0, keep: 0, end: 300)])
        XCTAssertEqual(DecodeWindow.windows(frames: 512), [W(start: 0, keep: 0, end: 512)])
        XCTAssertEqual(DecodeWindow.windows(frames: 1000), [
            W(start: 0, keep: 0, end: 512),
            W(start: 256, keep: 512, end: 768),
            W(start: 512, keep: 768, end: 1000),
        ])
    }

    func testEncoderMatchesReferenceOnTone() async throws {
        let codec = try await makeCodec()
        let fixture: CodesFixture = try load("codec_codes.json")
        let frames = fixture.tone_codes[0].count
        let rate = Double(DramaCodec.sampleRate)
        let tone = (0 ..< frames * DramaCodec.samplesPerFrame).map { i -> Float in
            let t = Double(i) / rate
            let twoPi = 2 * Double.pi
            let phase: Double = 220 * t + 3 * sin(twoPi * 5 * t)
            let envelope: Double = 0.6 + 0.4 * sin(twoPi * 2 * t)
            let vibrato: Double = 0.3 * sin(twoPi * phase) * envelope
            let overtone: Double = 0.15 * sin(twoPi * 660 * t)
            return Float(vibrato + overtone)
        }
        let codes = try await codec.encode(samples: tone)
        for book in 0 ..< DramaCodec.numCodebooks {
            let matches = zip(codes[book], fixture.tone_codes[book]).filter { $0 == $1 }.count
            let floor = book == 0 ? 0.9 : 0.5
            XCTAssertGreaterThanOrEqual(Double(matches) / Double(frames), floor, "codebook \(book) against the PyTorch encoder")
        }
    }

    func testPaddingKeepsPrefixAndLengths() async throws {
        let codec = try await makeCodec()
        let fixture: CodesFixture = try load("codec_codes.json")
        let prefix = fixture.codes.map { Array($0.prefix(20)) }

        let full = try await codec.decode(codes: fixture.codes)
        let short = try await codec.decode(codes: prefix)
        XCTAssertEqual(short.count, 20 * DramaCodec.samplesPerFrame)
        let maxDiff = zip(short, full).map { abs($0 - $1) }.max() ?? 0
        XCTAssertLessThan(maxDiff, 1e-3, "right padding must not change earlier frames")

        let long = fixture.codes.map { row in (0 ..< 10).flatMap { _ in row } }
        let longAudio = try await codec.decode(codes: long)
        XCTAssertEqual(longAudio.count, 640 * DramaCodec.samplesPerFrame)
        let headDiff = zip(longAudio, full).map { abs($0 - $1) }.max() ?? 0
        XCTAssertLessThan(headDiff, 1e-3, "the first window of a long decode is a plain padded decode")

        let encoded = try await codec.encode(samples: Array(full.prefix(5000)))
        XCTAssertEqual(encoded.map(\.count), Array(repeating: 3, count: DramaCodec.numCodebooks))
        let empty = try await codec.encode(samples: [])
        XCTAssertEqual(empty.map(\.count), Array(repeating: 0, count: DramaCodec.numCodebooks))
    }

    func testLoadAudioResamplesAndMixesToMono() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("codec-test-\(UUID().uuidString).wav")
        defer { try? FileManager.default.removeItem(at: url) }
        let rate = 22_050.0
        let format = AVAudioFormat(standardFormatWithSampleRate: rate, channels: 2)!
        let frames = AVAudioFrameCount(rate)
        let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frames)!
        buffer.frameLength = frames
        for i in 0 ..< Int(frames) {
            let s = Float(sin(2 * Double.pi * 440 * Double(i) / rate))
            buffer.floatChannelData![0][i] = 0.5 * s
            buffer.floatChannelData![1][i] = 0.1 * s
        }
        do {
            let file = try AVAudioFile(forWriting: url, settings: format.settings)
            try file.write(from: buffer)
        }

        let samples = try DramaCodec.loadAudio(url: url)
        XCTAssertEqual(Double(samples.count), Double(DramaCodec.sampleRate), accuracy: 64)
        let middle = samples[10_000 ..< 30_000]
        let peak = middle.map(abs).max() ?? 0
        XCTAssertEqual(peak, 0.3, accuracy: 0.01, "stereo 0.5 and 0.1 should average to 0.3")
    }
}

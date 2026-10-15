import AVFoundation
import Foundation
import XCTest
@testable import DramaEngine

final class CodecTests: XCTestCase {
    private struct CodesFixture: Decodable {
        let codes: [[Int32]]
        let reencoded_codebook0: [Int32]
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
                          "Core ML codec models not found in \(dir.path); run tools/convert_codec.py to create them")
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
        XCTAssertGreaterThanOrEqual(rate, 0.9, "codebook 0 re-encode agreement with PyTorch \(rate)")
    }

    func testPaddingKeepsPrefixAndLengths() async throws {
        let codec = try await makeCodec()
        let fixture: CodesFixture = try load("codec_codes.json")
        let prefix = fixture.codes.map { Array($0.prefix(20)) }

        let full = try await codec.decode(codes: fixture.codes)
        let short = try await codec.decode(codes: prefix)
        XCTAssertEqual(short.count, 20 * DramaCodec.samplesPerFrame)
        let maxDiff = zip(short, full).map { abs($0 - $1) }.max() ?? 0
        XCTAssertLessThan(maxDiff, 1e-2, "a 20-frame decode padded to 32 should match the first 20 frames of the 64-frame decode")

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

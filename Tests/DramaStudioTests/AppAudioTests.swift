// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import AVFoundation
import XCTest
@testable import DramaStudio

final class AppAudioTests: XCTestCase {
    func testTimelineHoldsAtChunkEndWhileStarvedThenResumes() {
        var timeline = PlaybackTimeline(origin: 1000)
        XCTAssertEqual(timeline.append(length: 500, earliest: 0), 0)
        XCTAssertEqual(timeline.append(length: 300, earliest: 100), 500, "queued chunks play back to back")
        XCTAssertEqual(timeline.audioPosition(atNodeSample: 650), 1650)

        XCTAssertEqual(timeline.append(length: 200, earliest: 2000), 2000, "a late chunk starts after the stall")
        XCTAssertEqual(timeline.audioPosition(atNodeSample: 1500), 1800, "position waits at the end of available audio")
        XCTAssertEqual(timeline.audioPosition(atNodeSample: 2100), 1900)
        XCTAssertEqual(timeline.audioPosition(atNodeSample: 9999), 2000)
        XCTAssertEqual(timeline.audioEnd, 2000)
    }

    func testExportedFilesReadBackAt44kMono() throws {
        let samples = (0..<44_100).map { Float(sin(Double($0) * 0.05) * 0.5) }
        let dir = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        for format in ExportFormat.allCases {
            let url = dir.appending(path: "take.\(format.fileExtension)")
            try AudioExporter.write(samples, to: url, as: format)
            let file = try AVAudioFile(forReading: url)
            XCTAssertEqual(file.fileFormat.sampleRate, 44_100, "\(format)")
            XCTAssertEqual(file.fileFormat.channelCount, 1, "\(format)")
            let read = try readAll(file)
            switch format {
            case .wav16:
                XCTAssertEqual(file.fileFormat.settings[AVLinearPCMBitDepthKey] as? Int, 16)
                XCTAssertEqual(read.count, samples.count)
                XCTAssertEqual(read[1234], samples[1234], accuracy: 1e-4)
            case .wavFloat:
                XCTAssertEqual(file.fileFormat.commonFormat, .pcmFormatFloat32)
                XCTAssertEqual(read, samples)
            case .aac:
                XCTAssertEqual(file.fileFormat.settings[AVFormatIDKey] as? UInt32, kAudioFormatMPEG4AAC)
                XCTAssertEqual(Double(read.count), Double(samples.count), accuracy: 2048)
            }
        }
    }

    /// A single `read(into:)` may return fewer frames than asked for.
    private func readAll(_ file: AVAudioFile) throws -> [Float] {
        var samples: [Float] = []
        let buffer = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: 4096)!
        while file.framePosition < file.length {
            try file.read(into: buffer)
            if buffer.frameLength == 0 { break }
            samples += UnsafeBufferPointer(start: buffer.floatChannelData![0], count: Int(buffer.frameLength))
        }
        return samples
    }

    /// Renders the engine offline so the test needs no audio device. At 1.5x
    /// each rendered output frame consumes 1.5 source samples, and the
    /// position the scrubber reads must count those source samples.
    @MainActor
    func testPositionCountsSourceSamplesAtOneAndAHalfSpeed() throws {
        let format = AVAudioFormat(standardFormatWithSampleRate: 44_100, channels: 1)!
        let audio = (0..<44_100 * 6).map { Float(sin(Double($0) * 0.03) * 0.5) }
        var measured: [Float: Double] = [:]
        for rate: Float in [1, 1.5] {
            let player = StreamingPlayer()
            try player.engine.enableManualRenderingMode(.offline, format: format, maximumFrameCount: 4096)
            player.rate = rate
            try player.play(audio, from: 1000)
            let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 4096)!
            var rendered = 0
            while rendered < 44_100 * 2 {
                XCTAssertEqual(try player.engine.renderOffline(4096, to: buffer), .success)
                rendered += Int(buffer.frameLength)
            }
            let position = try XCTUnwrap(player.position)
            measured[rate] = Double(position - 1000) / Double(rendered)
            player.stop()
        }
        XCTAssertEqual(try XCTUnwrap(measured[1]), 1, accuracy: 0.05)
        XCTAssertEqual(try XCTUnwrap(measured[1.5]), 1.5, accuracy: 0.08, "\(measured)")
    }

    func testWaveformPeaksAreNormalisedPerBin() {
        let peaks = WaveformScrubber.peaks(of: [0, 0.5, -1, 0.25, 0.1, 0], bins: 3)
        XCTAssertEqual(peaks, [0.5, 1, 0.1])
    }
}

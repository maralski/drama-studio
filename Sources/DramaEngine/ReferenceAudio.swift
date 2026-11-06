// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
@preconcurrency import AVFoundation
import Foundation

/// Clip audio as engines and the voice library take it: mono 44.1 kHz Float samples.
public enum ReferenceAudio {
    /// Writes mono 32-bit float WAV at 44.1 kHz, which ``load(url:)`` reads back unchanged.
    public static func write(_ samples: [Float], to url: URL) throws {
        let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: Double(ModelConstants.sampleRate), channels: 1, interleaved: false)!
        let file = try AVAudioFile(forWriting: url, settings: format.settings, commonFormat: .pcmFormatFloat32, interleaved: false)
        defer { file.close() }
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(max(samples.count, 1))) else {
            throw CodecError.audio("cannot allocate \(samples.count) frames")
        }
        buffer.frameLength = AVAudioFrameCount(samples.count)
        guard !samples.isEmpty else { return }
        samples.withUnsafeBufferPointer { buffer.floatChannelData![0].update(from: $0.baseAddress!, count: samples.count) }
        try file.write(from: buffer)
    }

    /// Reads any format AVFoundation can open, averages the channels to mono, and
    /// resamples to 44.1 kHz. Reads in blocks: one `read(into:)` can return
    /// fewer frames than the file holds.
    public static func load(url: URL) throws -> [Float] {
        let file = try AVAudioFile(forReading: url)
        let format = file.processingFormat
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 1 << 16) else {
            throw CodecError.audio("cannot allocate a read buffer")
        }
        let channelCount = Int(format.channelCount)
        var mono: [Float] = []
        mono.reserveCapacity(Int(file.length))
        while file.framePosition < file.length {
            try file.read(into: buffer)
            guard buffer.frameLength > 0 else { break }
            guard let channels = buffer.floatChannelData else { throw CodecError.audio("no float channel data") }
            let count = Int(buffer.frameLength)
            var block = [Float](repeating: 0, count: count)
            for c in 0 ..< channelCount {
                let channel = UnsafeBufferPointer(start: channels[c], count: count)
                for i in 0 ..< count { block[i] += channel[i] }
            }
            if channelCount > 1 {
                let scale = 1 / Float(channelCount)
                for i in 0 ..< count { block[i] *= scale }
            }
            mono += block
        }
        if format.sampleRate == Double(ModelConstants.sampleRate) { return mono }
        return try resample(mono, from: format.sampleRate)
    }

    private static func resample(_ samples: [Float], from rate: Double) throws -> [Float] {
        guard let source = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: rate, channels: 1, interleaved: false),
              let target = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: Double(ModelConstants.sampleRate), channels: 1, interleaved: false),
              let converter = AVAudioConverter(from: source, to: target),
              let input = AVAudioPCMBuffer(pcmFormat: source, frameCapacity: AVAudioFrameCount(max(samples.count, 1)))
        else { throw CodecError.audio("cannot build a \(rate) Hz to \(ModelConstants.sampleRate) Hz converter") }
        converter.sampleRateConverterQuality = AVAudioQuality.max.rawValue
        converter.sampleRateConverterAlgorithm = AVSampleRateConverterAlgorithm_Mastering
        input.frameLength = AVAudioFrameCount(samples.count)
        samples.withUnsafeBufferPointer { input.floatChannelData![0].update(from: $0.baseAddress!, count: samples.count) }

        let chunk: AVAudioFrameCount = 65_536
        var out: [Float] = []
        out.reserveCapacity(Int(Double(samples.count) * Double(ModelConstants.sampleRate) / rate) + Int(chunk))
        var fed = false
        while true {
            guard let buffer = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: chunk) else {
                throw CodecError.audio("cannot allocate output buffer")
            }
            var error: NSError?
            let status = converter.convert(to: buffer, error: &error) { _, inputStatus in
                if fed {
                    inputStatus.pointee = .endOfStream
                    return nil
                }
                fed = true
                inputStatus.pointee = .haveData
                return input
            }
            if let error { throw CodecError.audio(error.localizedDescription) }
            out.append(contentsOf: UnsafeBufferPointer(start: buffer.floatChannelData![0], count: Int(buffer.frameLength)))
            if status == .endOfStream || status == .error || (status == .inputRanDry && buffer.frameLength == 0) { break }
        }
        return out
    }
}

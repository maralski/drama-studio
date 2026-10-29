// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import AVFoundation
import DramaEngine
import Foundation

public enum AudioEncoding {
    public static let sampleRate = ModelConstants.sampleRate

    public static func pcm16(_ samples: [Float]) -> Data {
        var data = Data(capacity: samples.count * 2)
        for sample in samples {
            let value = Int16((max(-1, min(1, sample)) * 32767).rounded())
            withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) }
        }
        return data
    }

    public static func wav(_ samples: [Float]) -> Data {
        let pcm = pcm16(samples)
        var header = Data()
        func append<T: FixedWidthInteger>(_ value: T) { withUnsafeBytes(of: value.littleEndian) { header.append(contentsOf: $0) } }
        header.append(Data("RIFF".utf8))
        append(UInt32(36 + pcm.count))
        header.append(Data("WAVEfmt ".utf8))
        append(UInt32(16))
        append(UInt16(1))
        append(UInt16(1))
        append(UInt32(sampleRate))
        append(UInt32(sampleRate * 2))
        append(UInt16(2))
        append(UInt16(16))
        header.append(Data("data".utf8))
        append(UInt32(pcm.count))
        return header + pcm
    }

    public static func m4a(_ samples: [Float]) throws -> Data {
        let url = FileManager.default.temporaryDirectory.appending(path: "drama-\(UUID().uuidString).m4a")
        defer { try? FileManager.default.removeItem(at: url) }
        let settings: [String: Any] = [AVFormatIDKey: kAudioFormatMPEG4AAC, AVSampleRateKey: Double(sampleRate), AVNumberOfChannelsKey: 1, AVEncoderBitRateKey: 192_000]
        let file = try AVAudioFile(forWriting: url, settings: settings, commonFormat: .pcmFormatFloat32, interleaved: false)
        let buffer = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(max(samples.count, 1)))!
        buffer.frameLength = AVAudioFrameCount(samples.count)
        samples.withUnsafeBufferPointer { buffer.floatChannelData![0].update(from: $0.baseAddress!, count: samples.count) }
        try file.write(from: buffer)
        file.close()
        return try Data(contentsOf: url)
    }

    public static func encode(_ samples: [Float], as format: AudioFormat) throws -> Data {
        switch format {
        case .wav: wav(samples)
        case .pcm: pcm16(samples)
        case .m4a: try m4a(samples)
        }
    }

    /// AVFoundation picks a decoder from the file extension, so uploaded bytes
    /// need one that matches their container.
    public static func fileExtension(sniffing data: Data) -> String {
        let bytes = [UInt8](data.prefix(12))
        func ascii(_ range: Range<Int>) -> String { bytes.count >= range.upperBound ? String(decoding: bytes[range], as: UTF8.self) : "" }
        if ascii(0..<4) == "RIFF" { return "wav" }
        if ascii(0..<4) == "FORM" { return "aiff" }
        if ascii(0..<4) == "fLaC" { return "flac" }
        if ascii(0..<4) == "caff" { return "caf" }
        if ascii(4..<8) == "ftyp" { return "m4a" }
        if ascii(0..<3) == "ID3" || (bytes.count >= 2 && bytes[0] == 0xFF && bytes[1] & 0xE0 == 0xE0) { return "mp3" }
        return "wav"
    }
}

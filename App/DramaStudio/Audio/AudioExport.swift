import AVFoundation
import DramaEngine
import UniformTypeIdentifiers

enum ExportFormat: String, CaseIterable, Identifiable {
    case wav16, wavFloat, aac

    var id: String { rawValue }

    var title: String {
        switch self {
        case .wav16: "WAV, 16-bit"
        case .wavFloat: "WAV, 32-bit Float"
        case .aac: "M4A (AAC)"
        }
    }

    var contentType: UTType {
        switch self {
        case .wav16, .wavFloat: .wav
        case .aac: .mpeg4Audio
        }
    }

    var fileExtension: String { self == .aac ? "m4a" : "wav" }

    var settings: [String: Any] {
        let common: [String: Any] = [AVSampleRateKey: Double(ModelConstants.sampleRate), AVNumberOfChannelsKey: 1]
        switch self {
        case .wav16:
            return common.merging([AVFormatIDKey: kAudioFormatLinearPCM, AVLinearPCMBitDepthKey: 16, AVLinearPCMIsFloatKey: false, AVLinearPCMIsBigEndianKey: false]) { $1 }
        case .wavFloat:
            return common.merging([AVFormatIDKey: kAudioFormatLinearPCM, AVLinearPCMBitDepthKey: 32, AVLinearPCMIsFloatKey: true, AVLinearPCMIsBigEndianKey: false]) { $1 }
        case .aac:
            return common.merging([AVFormatIDKey: kAudioFormatMPEG4AAC, AVEncoderBitRateKey: 192_000]) { $1 }
        }
    }
}

enum AudioExporter {
    static func write(_ samples: [Float], to url: URL, as format: ExportFormat) throws {
        let file = try AVAudioFile(forWriting: url, settings: format.settings, commonFormat: .pcmFormatFloat32, interleaved: false)
        defer { file.close() }
        let blockSize = 1 << 16
        var start = 0
        while start < samples.count {
            let count = min(blockSize, samples.count - start)
            let buffer = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(count))!
            buffer.frameLength = AVAudioFrameCount(count)
            samples.withUnsafeBufferPointer { buffer.floatChannelData![0].update(from: $0.baseAddress! + start, count: count) }
            try file.write(from: buffer)
            start += count
        }
    }
}

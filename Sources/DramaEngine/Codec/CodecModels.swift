@preconcurrency import AVFoundation
import CoreML
import Foundation

public enum CodecError: Error, CustomStringConvertible {
    case missingModel(URL)
    case badCodes(String)
    case unexpectedOutput(String)
    case audio(String)

    public var description: String {
        switch self {
        case .missingModel(let url): return "codec model not found at \(url.path); run tools/convert_codec.py"
        case .badCodes(let m): return "bad codec codes: \(m)"
        case .unexpectedOutput(let m): return "unexpected codec output: \(m)"
        case .audio(let m): return "audio: \(m)"
        }
    }
}

/// The Drama 3 DAC codec on Core ML: 10 codebooks per 2048-sample frame at 44.1 kHz.
///
/// Both models take one static window of `windowFrames` frames so the Neural Engine
/// sees a fixed shape. Shorter inputs are right-padded and trimmed; the codec is
/// causal, so padding never changes the frames that are kept. Longer inputs run in
/// windows that carry `contextFrames` of left context and keep only the new frames.
/// With 256 frames of context the windowed decode stays within 0.006 of a full
/// decode (53 dB) and the windowed encode keeps 99.8% of codebook 0.
public actor DramaCodec {
    public static let sampleRate = 44_100
    public static let samplesPerFrame = 2048
    public static let numCodebooks = 10
    public static let windowFrames = 512
    public static let contextFrames = 256
    /// The frame counts the models accept.
    public static let frameSizes = [windowFrames]
    public static let encoderName = "DramaCodecEncoder"
    public static let decoderName = "DramaCodecDecoder"

    public static var defaultDirectory: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("DramaStudio/Codec", isDirectory: true)
    }

    private let encoder: MLModel
    private let decoder: MLModel

    /// Loads both models from `directory`, compiling each `.mlpackage` to a cached
    /// `.mlmodelc` beside it the first time (and again whenever the package is newer).
    /// `computeUnits` applies to the decoder; the GPU returns wrong audio for it, so
    /// keep it to `.cpuOnly` or `.cpuAndNeuralEngine`. The fp32 encoder always runs on
    /// the CPU.
    public init(directory: URL = DramaCodec.defaultDirectory, computeUnits: MLComputeUnits = .cpuOnly) async throws {
        let encoderConfig = MLModelConfiguration()
        encoderConfig.computeUnits = .cpuOnly
        let decoderConfig = MLModelConfiguration()
        decoderConfig.computeUnits = computeUnits
        encoder = try await MLModel.load(contentsOf: Self.compiledModel(named: Self.encoderName, in: directory), configuration: encoderConfig)
        decoder = try await MLModel.load(contentsOf: Self.compiledModel(named: Self.decoderName, in: directory), configuration: decoderConfig)
    }

    public static func modelsExist(in directory: URL = DramaCodec.defaultDirectory) -> Bool {
        [encoderName, decoderName].allSatisfy { name in
            ["mlpackage", "mlmodelc"].contains { ext in
                FileManager.default.fileExists(atPath: directory.appendingPathComponent("\(name).\(ext)").path)
            }
        }
    }

    /// A model call over frames `start ..< end` whose output is kept from `keep` on.
    struct Window: Equatable {
        let start: Int
        let keep: Int
        let end: Int
    }

    static func windows(frames: Int) -> [Window] {
        var out: [Window] = []
        var keep = 0
        while keep < frames {
            let start = keep == 0 ? 0 : keep - contextFrames
            let end = min(frames, start + windowFrames)
            out.append(Window(start: start, keep: keep, end: end))
            keep = end
        }
        return out
    }

    /// `[10][T]` codes to `T * 2048` mono samples at 44.1 kHz.
    public func decode(codes: [[Int32]]) async throws -> [Float] {
        guard codes.count == Self.numCodebooks else {
            throw CodecError.badCodes("expected \(Self.numCodebooks) codebooks, got \(codes.count)")
        }
        let frames = codes[0].count
        guard codes.allSatisfy({ $0.count == frames }) else { throw CodecError.badCodes("codebooks differ in length") }

        let width = Self.windowFrames
        var audio: [Float] = []
        audio.reserveCapacity(frames * Self.samplesPerFrame)
        for w in Self.windows(frames: frames) {
            var flat = [Int32](repeating: 0, count: Self.numCodebooks * width)
            for (book, row) in codes.enumerated() {
                flat.replaceSubrange(book * width ..< book * width + (w.end - w.start), with: row[w.start ..< w.end])
            }
            let input = MLMultiArray(MLShapedArray<Int32>(scalars: flat, shape: [1, Self.numCodebooks, width]))
            let output = try await decoder.prediction(from: MLDictionaryFeatureProvider(dictionary: ["codes": input]))
            guard let samples = output.featureValue(for: "audio")?.multiArrayValue else {
                throw CodecError.unexpectedOutput("decoder returned no `audio`")
            }
            let scalars = MLShapedArray<Float>(converting: samples).scalars
            guard scalars.count == width * Self.samplesPerFrame else {
                throw CodecError.unexpectedOutput("decoder returned \(scalars.count) samples for \(width) frames")
            }
            audio.append(contentsOf: scalars[(w.keep - w.start) * Self.samplesPerFrame ..< (w.end - w.start) * Self.samplesPerFrame])
        }
        return audio
    }

    /// 44.1 kHz mono samples to `[10][ceil(n / 2048)]` codes.
    public func encode(samples: [Float]) async throws -> [[Int32]] {
        let frames = (samples.count + Self.samplesPerFrame - 1) / Self.samplesPerFrame
        let width = Self.windowFrames
        var codes = [[Int32]](repeating: [], count: Self.numCodebooks)
        for w in Self.windows(frames: frames) {
            let lo = w.start * Self.samplesPerFrame
            let hi = min(samples.count, w.end * Self.samplesPerFrame)
            var audio = [Float](repeating: 0, count: width * Self.samplesPerFrame)
            audio.replaceSubrange(0 ..< hi - lo, with: samples[lo ..< hi])
            let input = MLMultiArray(MLShapedArray<Float>(scalars: audio, shape: [1, 1, audio.count]))
            let output = try await encoder.prediction(from: MLDictionaryFeatureProvider(dictionary: ["audio": input]))
            guard let result = output.featureValue(for: "codes")?.multiArrayValue else {
                throw CodecError.unexpectedOutput("encoder returned no `codes`")
            }
            let shaped = MLShapedArray<Int32>(converting: result)
            guard shaped.shape == [1, Self.numCodebooks, width] else {
                throw CodecError.unexpectedOutput("encoder returned shape \(shaped.shape)")
            }
            let scalars = shaped.scalars
            for book in 0 ..< Self.numCodebooks {
                codes[book].append(contentsOf: scalars[book * width + (w.keep - w.start) ..< book * width + (w.end - w.start)])
            }
        }
        return codes
    }


    /// Reads any format AVFoundation can open, averages the channels to mono, and
    /// resamples to 44.1 kHz.
    public static func loadAudio(url: URL) throws -> [Float] {
        let file = try AVAudioFile(forReading: url)
        let format = file.processingFormat
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(file.length)) else {
            throw CodecError.audio("cannot allocate \(file.length) frames")
        }
        try file.read(into: buffer)
        guard let channels = buffer.floatChannelData else { throw CodecError.audio("no float channel data") }

        let count = Int(buffer.frameLength)
        let channelCount = Int(format.channelCount)
        var mono = [Float](repeating: 0, count: count)
        for c in 0 ..< channelCount {
            let channel = UnsafeBufferPointer(start: channels[c], count: count)
            for i in 0 ..< count { mono[i] += channel[i] }
        }
        if channelCount > 1 {
            let scale = 1 / Float(channelCount)
            for i in 0 ..< count { mono[i] *= scale }
        }
        if format.sampleRate == Double(sampleRate) { return mono }
        return try resample(mono, from: format.sampleRate)
    }

    private static func resample(_ samples: [Float], from rate: Double) throws -> [Float] {
        guard let source = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: rate, channels: 1, interleaved: false),
              let target = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: Double(sampleRate), channels: 1, interleaved: false),
              let converter = AVAudioConverter(from: source, to: target),
              let input = AVAudioPCMBuffer(pcmFormat: source, frameCapacity: AVAudioFrameCount(max(samples.count, 1)))
        else { throw CodecError.audio("cannot build a \(rate) Hz to \(sampleRate) Hz converter") }
        converter.sampleRateConverterQuality = AVAudioQuality.max.rawValue
        converter.sampleRateConverterAlgorithm = AVSampleRateConverterAlgorithm_Mastering
        input.frameLength = AVAudioFrameCount(samples.count)
        samples.withUnsafeBufferPointer { input.floatChannelData![0].update(from: $0.baseAddress!, count: samples.count) }

        let chunk: AVAudioFrameCount = 65_536
        var out: [Float] = []
        out.reserveCapacity(Int(Double(samples.count) * Double(sampleRate) / rate) + Int(chunk))
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

    private static func compiledModel(named name: String, in directory: URL) async throws -> URL {
        let fm = FileManager.default
        let package = directory.appendingPathComponent("\(name).mlpackage")
        let compiled = directory.appendingPathComponent("\(name).mlmodelc")
        let hasPackage = fm.fileExists(atPath: package.path)
        if fm.fileExists(atPath: compiled.path) {
            let manifest = package.appendingPathComponent("Manifest.json")
            if !hasPackage || modificationDate(compiled) >= modificationDate(manifest) { return compiled }
        }
        guard hasPackage else { throw CodecError.missingModel(package) }
        let temporary = try await MLModel.compileModel(at: package)
        if fm.fileExists(atPath: compiled.path) { try fm.removeItem(at: compiled) }
        try fm.moveItem(at: temporary, to: compiled)
        return compiled
    }

    private static func modificationDate(_ url: URL) -> Date {
        (try? url.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
    }
}

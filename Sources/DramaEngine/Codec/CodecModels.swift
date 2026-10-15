@preconcurrency import AVFoundation
import CoreML
import Foundation

public enum CodecError: Error, CustomStringConvertible {
    case missingModel(URL)
    case badCodes(String)
    case tooLong(frames: Int, limit: Int)
    case unexpectedOutput(String)
    case audio(String)

    public var description: String {
        switch self {
        case .missingModel(let url): return "codec model not found at \(url.path); run tools/convert_codec.py"
        case .badCodes(let m): return "bad codec codes: \(m)"
        case .tooLong(let frames, let limit): return "\(frames) frames exceeds the codec limit of \(limit) (\(limit * DramaCodec.samplesPerFrame / DramaCodec.sampleRate) s)"
        case .unexpectedOutput(let m): return "unexpected codec output: \(m)"
        case .audio(let m): return "audio: \(m)"
        }
    }
}

/// The Drama 3 DAC codec on Core ML: 10 codebooks per 2048-sample frame at 44.1 kHz.
///
/// The models take a fixed set of frame counts (`frameSizes`) so the Neural Engine
/// sees static shapes. Inputs are right-padded to the next size and outputs trimmed;
/// the codec is causal, so padding never changes the frames that are kept.
public actor DramaCodec {
    public static let sampleRate = 44_100
    public static let samplesPerFrame = 2048
    public static let numCodebooks = 10
    public static let frameSizes = [16, 32, 64, 128, 256, 512, 768]
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
    public init(directory: URL = DramaCodec.defaultDirectory, computeUnits: MLComputeUnits = .all) async throws {
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits
        encoder = try await MLModel.load(contentsOf: Self.compiledModel(named: Self.encoderName, in: directory), configuration: config)
        decoder = try await MLModel.load(contentsOf: Self.compiledModel(named: Self.decoderName, in: directory), configuration: config)
    }

    public static func modelsExist(in directory: URL = DramaCodec.defaultDirectory) -> Bool {
        [encoderName, decoderName].allSatisfy { name in
            ["mlpackage", "mlmodelc"].contains { ext in
                FileManager.default.fileExists(atPath: directory.appendingPathComponent("\(name).\(ext)").path)
            }
        }
    }

    /// `[10][T]` codes to `T * 2048` mono samples at 44.1 kHz.
    public func decode(codes: [[Int32]]) async throws -> [Float] {
        guard codes.count == Self.numCodebooks else {
            throw CodecError.badCodes("expected \(Self.numCodebooks) codebooks, got \(codes.count)")
        }
        let frames = codes[0].count
        guard codes.allSatisfy({ $0.count == frames }) else { throw CodecError.badCodes("codebooks differ in length") }
        if frames == 0 { return [] }
        let padded = try Self.paddedFrames(frames)

        var flat = [Int32](repeating: 0, count: Self.numCodebooks * padded)
        for (book, row) in codes.enumerated() {
            flat.replaceSubrange(book * padded ..< book * padded + frames, with: row)
        }
        let input = MLMultiArray(MLShapedArray<Int32>(scalars: flat, shape: [1, Self.numCodebooks, padded]))
        let output = try await decoder.prediction(from: MLDictionaryFeatureProvider(dictionary: ["codes": input]))
        guard let audio = output.featureValue(for: "audio")?.multiArrayValue else {
            throw CodecError.unexpectedOutput("decoder returned no `audio`")
        }
        let samples = MLShapedArray<Float>(converting: audio).scalars
        guard samples.count == padded * Self.samplesPerFrame else {
            throw CodecError.unexpectedOutput("decoder returned \(samples.count) samples for \(padded) frames")
        }
        return Array(samples.prefix(frames * Self.samplesPerFrame))
    }

    /// 44.1 kHz mono samples to `[10][ceil(n / 2048)]` codes.
    public func encode(samples: [Float]) async throws -> [[Int32]] {
        let frames = (samples.count + Self.samplesPerFrame - 1) / Self.samplesPerFrame
        if frames == 0 { return Array(repeating: [], count: Self.numCodebooks) }
        let padded = try Self.paddedFrames(frames)

        var audio = samples
        audio.append(contentsOf: repeatElement(0, count: padded * Self.samplesPerFrame - samples.count))
        let input = MLMultiArray(MLShapedArray<Float>(scalars: audio, shape: [1, 1, audio.count]))
        let output = try await encoder.prediction(from: MLDictionaryFeatureProvider(dictionary: ["audio": input]))
        guard let codes = output.featureValue(for: "codes")?.multiArrayValue else {
            throw CodecError.unexpectedOutput("encoder returned no `codes`")
        }
        let shaped = MLShapedArray<Int32>(converting: codes)
        guard shaped.shape == [1, Self.numCodebooks, padded] else {
            throw CodecError.unexpectedOutput("encoder returned shape \(shaped.shape)")
        }
        let scalars = shaped.scalars
        return (0 ..< Self.numCodebooks).map { book in
            Array(scalars[book * padded ..< book * padded + frames])
        }
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

    static func paddedFrames(_ frames: Int) throws -> Int {
        guard let size = frameSizes.first(where: { $0 >= frames }) else {
            throw CodecError.tooLong(frames: frames, limit: frameSizes.last!)
        }
        return size
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

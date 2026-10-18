@preconcurrency import AVFoundation
import CoreML
import Foundation

/// Why a codec call failed.
public enum CodecError: Error, CustomStringConvertible {
    /// No `.mlpackage` or `.mlmodelc` at the URL. ``DramaSetup`` builds them.
    case missingModel(URL)
    /// The codes do not have 10 codebooks of one length, or a window does not fit.
    case badCodes(String)
    /// A Core ML model returned something other than the expected array.
    case unexpectedOutput(String)
    /// The audio file could not be read or resampled.
    case audio(String)
    /// A model file takes another frame count than its name promises; it was
    /// built by an older `convert_codec.py`.
    case wrongShape(URL, String)

    /// A message for a person.
    public var description: String {
        switch self {
        case .missingModel(let url): return "codec model not found at \(url.path); build it with DramaSetup.prepare or Sources/DramaEngine/CodecConversion/convert_codec.py"
        case .badCodes(let m): return "bad codec codes: \(m)"
        case .unexpectedOutput(let m): return "unexpected codec output: \(m)"
        case .audio(let m): return "audio: \(m)"
        case .wrongShape(let url, let m): return "codec model \(url.lastPathComponent) \(m); rebuild it with DramaSetup.prepare or Sources/DramaEngine/CodecConversion/convert_codec.py"
        }
    }
}

/// The Drama 3 DAC codec on Core ML: 10 codebooks per 2048-sample frame at 44.1 kHz.
///
/// Use it on its own to encode audio to codes or decode codes to audio
/// without loading the 39.6 GB model; it needs only the two codec models,
/// about 1.2 GB. Both run on the CPU: the encoder in fp32 matches PyTorch's
/// codes exactly and takes about 2.1 s for a 10 s clip on an M3 Max; the
/// fp16 decoder is 35 to 39 dB from the fp32 reference and takes about 1.5 s
/// per 512-frame window (23.8 s of audio).
///
/// Each model takes one static window of frames, because Core ML's enumerated
/// shapes crash the CPU backend here. The encoder and the decoder take
/// `windowFrames`; a second copy of the decoder, the probe, takes `probeFrames`
/// and serves the generator's silence stop, which reads only the newest frames.
/// Shorter inputs are right-padded and trimmed; the codec is causal, so padding
/// never changes the frames that are kept. Longer inputs run in windows that
/// carry `contextFrames` of left context and keep only the new frames. With 256
/// frames of context the windowed decode stays within 0.006 of a full decode
/// (53 dB) and the windowed encode keeps 99.8% of codebook 0.
public actor DramaCodec {
    /// Sample rate in Hz of every buffer in and out.
    public static let sampleRate = 44_100
    /// Samples per frame.
    public static let samplesPerFrame = 2048
    /// Codebooks per frame.
    public static let numCodebooks = 10
    /// Frames per model call, about 23.8 s of audio.
    public static let windowFrames = 512
    /// Frames of left context a window carries when the input is longer than one window.
    public static let contextFrames = 256
    /// Frames per probe call: the shape ``probeName`` takes.
    public static let probeFrames = 64
    /// Frames of left context a probe window carries. The generator probes every
    /// 16 frames, so a probe call usually classifies 16 new frames after 48 of context.
    public static let probeContext = 48
    /// The encoder's file name without extension, `.mlpackage` or `.mlmodelc`.
    public static let encoderName = "DramaCodecEncoder"
    /// The decoder's file name without extension, `.mlpackage` or `.mlmodelc`.
    public static let decoderName = "DramaCodecDecoder"
    /// The probe decoder's file name without extension, `.mlpackage` or `.mlmodelc`.
    public static let probeName = "DramaCodecProbe"
    /// Every model a codec directory must hold.
    public static let modelNames = [encoderName, decoderName, probeName]

    /// `~/Library/Application Support/DramaStudio/Codec`, where
    /// `convert_codec.py` writes the models by default.
    public static var defaultDirectory: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("DramaStudio/Codec", isDirectory: true)
    }

    private let encoder: MLModel
    /// The decoders by the frame count each takes, smallest first.
    private let decoders: [(frames: Int, model: MLModel)]

    /// The frame counts the loaded decoders take, smallest first.
    package var decoderFrames: [Int] { decoders.map(\.frames) }

    /// Loads the three models from `directory`, compiling each `.mlpackage` to a cached
    /// `.mlmodelc` beside it the first time (and again whenever the package is newer).
    /// `computeUnits` applies to the decoders; the GPU returns wrong audio for them, so
    /// keep it to `.cpuOnly` or `.cpuAndNeuralEngine`. The fp32 encoder always runs on
    /// the CPU.
    public init(directory: URL = DramaCodec.defaultDirectory, computeUnits: MLComputeUnits = .cpuOnly) async throws {
        try await self.init(directory: directory, decoders: [Self.decoderName: Self.windowFrames, Self.probeName: Self.probeFrames], computeUnits: computeUnits)
    }

    /// Loads the encoder and the named decoders, each checked against the frame
    /// count its file records.
    package init(directory: URL, decoders names: [String: Int], computeUnits: MLComputeUnits = .cpuOnly) async throws {
        let encoderConfig = MLModelConfiguration()
        encoderConfig.computeUnits = .cpuOnly
        let decoderConfig = MLModelConfiguration()
        decoderConfig.computeUnits = computeUnits
        encoder = try await MLModel.load(contentsOf: Self.compiledModel(named: Self.encoderName, in: directory), configuration: encoderConfig)
        var loaded: [(frames: Int, model: MLModel)] = []
        for (name, frames) in names.sorted(by: { $0.value < $1.value }) {
            let url = try await Self.compiledModel(named: name, in: directory)
            let model = try await MLModel.load(contentsOf: url, configuration: decoderConfig)
            let recorded = (model.modelDescription.metadata[.creatorDefinedKey] as? [String: String])?["window_frames"].flatMap(Int.init)
            guard recorded == frames else {
                throw CodecError.wrongShape(url, "takes \(recorded.map(String.init) ?? "an unrecorded number of") frames, not \(frames)")
            }
            loaded.append((frames, model))
        }
        decoders = loaded
    }

    /// Whether every model is in `directory`, as a package or compiled.
    public static func modelsExist(in directory: URL = DramaCodec.defaultDirectory) -> Bool {
        modelNames.allSatisfy { name in
            ["mlpackage", "mlmodelc"].contains { ext in
                FileManager.default.fileExists(atPath: directory.appendingPathComponent("\(name).\(ext)").path)
            }
        }
    }

    /// `[10][T]` codes to `T * 2048` mono samples at 44.1 kHz.
    public func decode(codes: [[Int32]]) async throws -> [Float] {
        var audio: [Float] = []
        audio.reserveCapacity((codes.first?.count ?? 0) * Self.samplesPerFrame)
        for w in DecodeWindow.windows(frames: codes.first?.count ?? 0) {
            audio += try await decode(codes: codes, window: w)
        }
        return audio
    }

    /// The samples of frames `window.keep ..< window.end`, from one call of the
    /// smallest decoder that fits `window.start ..< window.end` of `codes`.
    package func decode(codes: [[Int32]], window w: DecodeWindow) async throws -> [Float] {
        guard codes.count == Self.numCodebooks else {
            throw CodecError.badCodes("expected \(Self.numCodebooks) codebooks, got \(codes.count)")
        }
        let frames = codes[0].count
        guard codes.allSatisfy({ $0.count == frames }) else { throw CodecError.badCodes("codebooks differ in length") }
        guard w.end <= frames, let (width, decoder) = decoders.first(where: { $0.frames >= w.end - w.start }) else {
            throw CodecError.badCodes("window \(w) does not fit \(frames) frames or any decoder shape \(decoderFrames)")
        }

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
        return Array(scalars[(w.keep - w.start) * Self.samplesPerFrame ..< (w.end - w.start) * Self.samplesPerFrame])
    }

    /// 44.1 kHz mono samples to `[10][ceil(n / 2048)]` codes.
    public func encode(samples: [Float]) async throws -> [[Int32]] {
        let frames = (samples.count + Self.samplesPerFrame - 1) / Self.samplesPerFrame
        let width = Self.windowFrames
        var codes = [[Int32]](repeating: [], count: Self.numCodebooks)
        for w in DecodeWindow.windows(frames: frames) {
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

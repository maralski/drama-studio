// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
@preconcurrency import AVFoundation
import CoreML
import Foundation
package import DramaTTS

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
    /// The voice has Drama 3 codes but no clip, and the codec that decodes
    /// them is not in the directory.
    case noCodecForClip(voiceName: String, directory: URL)

    /// A message for a person.
    public var description: String {
        switch self {
        case .missingModel(let url): return "codec model not found at \(url.path); build it with DramaSetup.prepare, or run the conversion command DramaSetup.check reports"
        case .badCodes(let m): return "bad codec codes: \(m)"
        case .unexpectedOutput(let m): return "unexpected codec output: \(m)"
        case .audio(let m): return "audio: \(m)"
        case .wrongShape(let url, let m): return "codec model \(url.lastPathComponent) \(m); rebuild it with DramaSetup.prepare, or run the conversion command DramaSetup.check reports"
        case .noCodecForClip(let name, let directory):
            return "the voice \(name) was saved without a clip, and recreating it from its Drama 3 codes needs the voice codec, which is not in \(directory.path); set up Drama 3 to get it"
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
    /// ``DramaSetup`` builds the models.
    public static var defaultDirectory: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("DramaStudio/Codec", isDirectory: true)
    }

    /// The engine's codec, for code that generates with it.
    package nonisolated let audioCodec: DramaAudioCodec

    /// Loads the models from `directory`, compiling a `.mlpackage` on first use.
    /// `computeUnits` applies to the decoders; the GPU returns wrong audio for them, so
    /// keep it to `.cpuOnly` or `.cpuAndNeuralEngine`. The fp32 encoder always runs on
    /// the CPU.
    public init(directory: URL = DramaCodec.defaultDirectory, computeUnits: MLComputeUnits = .cpuOnly) async throws {
        audioCodec = try await Drama3Errors.translating { try await DramaAudioCodec(directory: directory, computeUnits: computeUnits) }
    }

    /// The frame counts the loaded decoders take, smallest first.
    package var decoderFrames: [Int] { get async { await audioCodec.decoderFrames } }

    /// Whether every model is in `directory`, as a package or compiled.
    public static func modelsExist(in directory: URL = DramaCodec.defaultDirectory) -> Bool {
        DramaAudioCodec.modelsExist(in: directory)
    }

    /// A clip for a voice saved before the library kept clips, decoded from
    /// its Drama 3 codes; nil when the voice has no Drama 3 encoding. Loads
    /// only the codec, not the model, and fails with
    /// ``CodecError/noCodecForClip(voiceName:directory:)`` when it is missing.
    public static func recreateClip(of voice: ReferenceVoice, directory: URL = DramaCodec.defaultDirectory) async throws -> [Float]? {
        guard case .drama3(let codes)? = voice.encodings[.drama3] else { return nil }
        guard modelsExist(in: directory) else { throw CodecError.noCodecForClip(voiceName: voice.name, directory: directory) }
        return try await DramaCodec(directory: directory).decode(codes: codes)
    }

    /// `[10][T]` codes to `T * 2048` mono samples at 44.1 kHz.
    public func decode(codes: [[Int32]]) async throws -> [Float] {
        try await Drama3Errors.translating { try await audioCodec.decode(codes: codes) }
    }

    /// Mono 44.1 kHz samples to `[10][T]` codes, `T = ceil(count / 2048)`.
    public func encode(samples: [Float]) async throws -> [[Int32]] {
        try await Drama3Errors.translating { try await audioCodec.encode(samples: samples) }
    }
}

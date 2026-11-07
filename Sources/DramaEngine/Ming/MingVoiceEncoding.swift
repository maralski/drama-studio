// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation

/// What Ming needs to clone a voice from a stretch of audio: what that audio
/// says, its VAE latents (rows of 64, one per 80 ms) and its 192-value CAM++
/// speaker embedding. ``MingSynthesizer`` makes one from a clip or from a
/// take's audio. The JSON layout is Drama Studio's own, so a MingTTS release
/// cannot change saved voices or projects: latents and the embedding are
/// base64 little-endian Float32, about 43 KB for a 10 s clip.
public struct MingVoiceEncoding: Codable, Sendable, Hashable {
    public static let latentDim = 64
    public static let speakerEmbeddingDim = 192
    /// Samples one latent row stands for at 44.1 kHz: 80 ms.
    public static let samplesPerLatent = 3528

    /// What the encoded audio says, word for word. A take's excerpt says less
    /// than the take, so the encoding carries its own transcript.
    public let transcript: String
    /// The encoded audio's length at 44.1 kHz, before Ming pads it to whole rows.
    public let sampleCount: Int
    package let latents: [Float]
    package let speakerEmbedding: [Float]

    /// Latent rows, one per 80 ms of audio.
    public var latentRows: Int { latents.count / Self.latentDim }

    /// - Throws: ``EngineError/badVoiceEncoding(_:)`` when a shape is wrong.
    package init(transcript: String, sampleCount: Int, latents: [Float], speakerEmbedding: [Float]) throws {
        guard !latents.isEmpty, latents.count % Self.latentDim == 0 else {
            throw EngineError.badVoiceEncoding("latents are not rows of \(Self.latentDim)")
        }
        guard speakerEmbedding.count == Self.speakerEmbeddingDim else {
            throw EngineError.badVoiceEncoding("speaker embedding has \(speakerEmbedding.count) values, not \(Self.speakerEmbeddingDim)")
        }
        guard sampleCount > 0 else { throw EngineError.badVoiceEncoding("sample count must be positive") }
        self.transcript = transcript
        self.sampleCount = sampleCount
        self.latents = latents
        self.speakerEmbedding = speakerEmbedding
    }

    private enum CodingKeys: String, CodingKey { case transcript, sampleCount, latents, speakerEmbedding }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        func floats(_ key: CodingKeys) throws -> [Float] {
            guard let data = Data(base64Encoded: try container.decode(String.self, forKey: key)), data.count % 4 == 0 else {
                throw DecodingError.dataCorruptedError(forKey: key, in: container, debugDescription: "not base64 Float32")
            }
            return data.withUnsafeBytes { raw in
                raw.bindMemory(to: UInt32.self).map { Float(bitPattern: UInt32(littleEndian: $0)) }
            }
        }
        do {
            try self.init(transcript: try container.decode(String.self, forKey: .transcript),
                          sampleCount: try container.decode(Int.self, forKey: .sampleCount),
                          latents: try floats(.latents), speakerEmbedding: try floats(.speakerEmbedding))
        } catch let error as EngineError {
            throw DecodingError.dataCorruptedError(forKey: .latents, in: container, debugDescription: error.description)
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(transcript, forKey: .transcript)
        try container.encode(sampleCount, forKey: .sampleCount)
        try container.encode(Self.base64(latents), forKey: .latents)
        try container.encode(Self.base64(speakerEmbedding), forKey: .speakerEmbedding)
    }

    private static func base64(_ values: [Float]) -> String {
        var data = Data(capacity: values.count * 4)
        for value in values { withUnsafeBytes(of: value.bitPattern.littleEndian) { data.append(contentsOf: $0) } }
        return data.base64EncodedString()
    }
}

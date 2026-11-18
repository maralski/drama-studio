// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation
internal import DramaTTS

/// Clip audio as engines and the voice library take it: mono 44.1 kHz Float samples.
public enum ReferenceAudio {
    /// Writes mono 32-bit float WAV at 44.1 kHz, which ``load(url:)`` reads back unchanged.
    public static func write(_ samples: [Float], to url: URL) throws {
        try Drama3Errors.translating { try DramaAudioFile.write(samples, to: url) }
    }

    /// Reads any format AVFoundation can open, averages the channels to mono, and
    /// resamples to 44.1 kHz.
    public static func load(url: URL) throws -> [Float] {
        try Drama3Errors.translating { try DramaAudioFile.load(url: url) }
    }
}

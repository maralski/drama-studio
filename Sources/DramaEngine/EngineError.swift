// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import CryptoKit
import Foundation

/// Why the engine could not load or run.
public enum EngineError: Error, CustomStringConvertible {
    /// The Mac has no Metal device the engine can use.
    case noMetalDevice
    /// A kernel, or the resource bundle that holds the kernel sources, is missing.
    case missingKernel(String)
    /// The snapshot's files do not match what the engine expects.
    case badModel(String)
    /// A cached kernel schedule failed validation.
    case badSchedule(String)
    /// A Ming voice encoding's shapes are wrong.
    case badVoiceEncoding(String)
    /// The named voice has no stored clip, so it cannot be encoded for another engine.
    case noClip(voiceName: String)
    /// No engine of this kind is installed here.
    case engineUnavailable(SpeechEngineID)

    /// A message for a person.
    public var description: String {
        switch self {
        case .noMetalDevice: return "no Metal device"
        case .missingKernel(let n): return "kernel \(n) missing from the Metal library"
        case .badModel(let m): return "bad model: \(m)"
        case .badSchedule(let m): return "bad kernel schedule: \(m)"
        case .badVoiceEncoding(let m): return "bad voice encoding: \(m)"
        case .noClip(let name): return "the voice \(name) has no stored clip to encode for this engine"
        case .engineUnavailable(let engine): return "\(engine.capabilities.name) isn't installed"
        }
    }
}

// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import DramaEngine
import Foundation

/// What this Mac has against what each engine needs.
struct MacRequirements: Equatable {
    /// The setting that records "Continue Anyway" for `engine`. Drama 3 keeps
    /// the key from when it was the only engine.
    static func acknowledgedKey(for engine: SpeechEngineID) -> String {
        engine == .drama3 ? "requirementsAcknowledged" : "requirementsAcknowledged.\(engine.rawValue)"
    }

    var appleSilicon: Bool
    var memoryBytes: UInt64

    /// This Mac. `DRAMA_STUDIO_MEMORY_GB` stands in for the memory size, to
    /// see what a Mac with less memory is offered.
    static var current: MacRequirements {
        var arm64: Int32 = 0
        var size = MemoryLayout<Int32>.size
        let isArm = sysctlbyname("hw.optional.arm64", &arm64, &size, nil, 0) == 0 && arm64 == 1
        let simulated = ProcessInfo.processInfo.environment["DRAMA_STUDIO_MEMORY_GB"].flatMap(UInt64.init).map { $0 << 30 }
        return MacRequirements(appleSilicon: isArm, memoryBytes: simulated ?? ProcessInfo.processInfo.physicalMemory)
    }

    private func hasMemory(for engine: SpeechEngineID) -> Bool {
        engine.install.recommendedMemory.map { memoryBytes >= $0 } ?? true
    }

    /// One sentence per shortfall against `engine`; empty when the Mac meets its needs.
    func gaps(for engine: SpeechEngineID) -> [String] {
        var gaps: [String] = []
        let name = engine.capabilities.name
        if !appleSilicon {
            gaps.append("This Mac does not have Apple silicon. \(name) runs on the GPU of Apple silicon Macs and is not expected to load here.")
        }
        if !hasMemory(for: engine), let recommended = engine.install.recommendedMemory {
            gaps.append("This Mac has \(memoryBytes >> 30) GB of memory. \(name) maps about \(engine.install.downloadBytes / 1_000_000_000) GB of model weights and has been verified only on an M3 Max with 128 GB. Below \(recommended >> 30) GB, expect very slow speech, heavy swapping, or a load that fails.")
        }
        return gaps
    }

    /// The first engine, in the app's order of preference, whose memory need
    /// this Mac meets; the last engine when it meets none.
    var recommendedEngine: SpeechEngineID {
        SpeechEngineID.allCases.first(where: hasMemory) ?? SpeechEngineID.allCases.last!
    }

    /// What the setup window's engine step offers on this Mac.
    func offer(downloaded: Set<SpeechEngineID>) -> EngineOffer {
        EngineOffer(recommended: recommendedEngine,
                    gaps: Dictionary(uniqueKeysWithValues: SpeechEngineID.allCases.map { ($0, gaps(for: $0)) }).filter { !$0.value.isEmpty },
                    downloaded: downloaded)
    }
}

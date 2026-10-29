// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation

/// What this Mac has against what the engine was verified on.
struct MacRequirements: Equatable {
    static let recommendedMemory: UInt64 = 64 << 30
    static let acknowledgedKey = "requirementsAcknowledged"

    var appleSilicon: Bool
    var memoryBytes: UInt64

    /// This Mac. `DRAMA_STUDIO_MEMORY_GB` stands in for the memory size, to
    /// see the warning on a Mac that has enough.
    static var current: MacRequirements {
        var arm64: Int32 = 0
        var size = MemoryLayout<Int32>.size
        let isArm = sysctlbyname("hw.optional.arm64", &arm64, &size, nil, 0) == 0 && arm64 == 1
        let simulated = ProcessInfo.processInfo.environment["DRAMA_STUDIO_MEMORY_GB"].flatMap(UInt64.init).map { $0 << 30 }
        return MacRequirements(appleSilicon: isArm, memoryBytes: simulated ?? ProcessInfo.processInfo.physicalMemory)
    }

    /// One sentence per shortfall; empty when the Mac meets both.
    var gaps: [String] {
        var gaps: [String] = []
        if !appleSilicon {
            gaps.append("This Mac does not have Apple silicon. Drama Studio's engine runs on the GPU of Apple silicon Macs and is not expected to load here.")
        }
        if memoryBytes < Self.recommendedMemory {
            gaps.append("This Mac has \(memoryBytes >> 30) GB of memory. The engine maps about 41 GB of model weights and has been verified only on an M3 Max with 128 GB. Below 64 GB, expect very slow speech, heavy swapping, or a load that fails.")
        }
        return gaps
    }
}

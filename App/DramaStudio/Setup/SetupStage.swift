// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import DramaEngine
import Foundation

/// The steps of a first run, in order. A step is skipped when the Mac
/// already has what it produces.
enum SetupStep: String, CaseIterable, Sendable {
    case requirements, terms, weights, codec, tuning

    var title: String {
        switch self {
        case .requirements: "Check this Mac"
        case .terms: "Agree to the terms"
        case .weights: "Download Drama 3"
        case .codec: "Download the voice codec"
        case .tuning: "Tune for this Mac"
        }
    }
}

/// What is already on this Mac, read from disk and settings.
struct SetupFacts: Equatable {
    var requirementGaps: [String] = []
    var requirementsAcknowledged = false
    var termsAccepted = false
    var weightsReady = false
    var codecReady = false
    var tuned = false

    func satisfies(_ step: SetupStep) -> Bool {
        switch step {
        case .requirements: requirementGaps.isEmpty || requirementsAcknowledged
        case .terms: termsAccepted
        case .weights: weightsReady
        case .codec: codecReady
        case .tuning: tuned
        }
    }

    /// The first step still to do; `finished` holds steps done this launch,
    /// such as a tuning that kept the default schedule.
    func firstNeeded(finished: Set<SetupStep>) -> SetupStep? {
        SetupStep.allCases.first { !finished.contains($0) && !satisfies($0) }
    }
}

/// A download under way or paused.
struct Transfer: Equatable {
    var progress: DownloadProgress
    var paused = false
    var secondsLeft: Double?
}

enum CodecStage: Equatable {
    case downloading(Transfer)
    case unpacking
    /// A development build has no codec bundle to download; `command` builds the models with Python.
    case needsConversion(command: String)
}

/// Where the first run stands. The coordinator moves through it; the setup
/// window draws it; generation is refused in every case but the last two.
enum SetupStage: Equatable {
    case requirements(gaps: [String])
    case terms
    case weights(Transfer)
    case codec(CodecStage)
    case tuning(detail: String)
    /// Setup just finished: offer the first line.
    case firstLine
    case ready
    case failed(SetupStep, reason: String)

    var step: SetupStep? {
        switch self {
        case .requirements: .requirements
        case .terms: .terms
        case .weights: .weights
        case .codec: .codec
        case .tuning: .tuning
        case .failed(let step, _): step
        case .firstLine, .ready: nil
        }
    }

    var allowsGeneration: Bool {
        switch self {
        case .firstLine, .ready: true
        default: false
        }
    }
}

/// Bytes per second over the last half minute of a download, for the time estimate.
struct RateEstimate {
    private var samples: [(time: Double, bytes: Int64)] = []
    var window: Double = 30

    mutating func add(bytes: Int64, at time: Double) {
        samples.append((time, bytes))
        samples.removeAll { time - $0.time > window }
    }

    mutating func reset() { samples = [] }

    /// Nil until there are a few seconds of samples.
    func secondsLeft(remaining: Int64) -> Double? {
        guard let first = samples.first, let last = samples.last, last.time - first.time >= 3, last.bytes > first.bytes else { return nil }
        return Double(remaining) / (Double(last.bytes - first.bytes) / (last.time - first.time))
    }
}

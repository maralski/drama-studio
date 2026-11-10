// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import DramaEngine
import Foundation

/// The steps of a first run, in order. Which of them a run takes depends on
/// the engine being set up; a step is skipped when the Mac already has what
/// it produces.
enum SetupStep: String, CaseIterable, Sendable {
    case engine, requirements, terms, usage, weights, codec, tuning

    func title(for engine: SpeechEngineID) -> String {
        switch self {
        case .engine: "Choose a speech engine"
        case .requirements: "Check this Mac"
        case .terms: "Agree to the terms"
        case .usage: "Help improve Drama Studio"
        case .weights: "Download \(engine.capabilities.name)"
        case .codec: "Download the voice codec"
        case .tuning: "Tune for this Mac"
        }
    }

    /// The step's name in the window's step strip.
    var label: String {
        switch self {
        case .engine: "Engine"
        case .requirements: "This Mac"
        case .terms: "Terms"
        case .usage: "Usage"
        case .weights: "Model"
        case .codec: "Voice codec"
        case .tuning: "Tuning"
        }
    }
}

extension SpeechEngineID {
    /// The steps that set this engine up, read from its ``install`` row: the
    /// terms when its license asks for agreement, and the codec and tuning
    /// when it has those components.
    var setupSteps: [SetupStep] {
        let install = install
        return SetupStep.allCases.filter { step in
            switch step {
            case .engine, .requirements, .usage, .weights: true
            case .terms: install.license.requiresAgreement
            case .codec: install.components.contains(.codecModels)
            case .tuning: install.components.contains(.tuning)
            }
        }
    }
}

/// What the engine step offers.
struct EngineOffer: Equatable {
    /// The engine preselected for this Mac.
    var recommended: SpeechEngineID
    /// Why this Mac falls short of an engine, one sentence per shortfall.
    var gaps: [SpeechEngineID: [String]] = [:]
    /// Engines whose files are already on this Mac.
    var downloaded: Set<SpeechEngineID> = []
}

/// What is already on this Mac, read from disk and settings. The download,
/// codec, tuning and acknowledgement facts are those of ``engine``.
struct SetupFacts: Equatable {
    /// The engine this run sets up; nil until the user chooses one.
    var engine: SpeechEngineID?
    var offer = EngineOffer(recommended: .drama3)
    var requirementsAcknowledged = false
    var termsAccepted = false
    var weightsReady = false
    var codecReady = false
    var tuned = false
    var usageChoiceMade = false

    var requirementGaps: [String] { engine.flatMap { offer.gaps[$0] } ?? [] }

    func satisfies(_ step: SetupStep) -> Bool {
        switch step {
        case .engine: engine != nil
        case .requirements: requirementGaps.isEmpty || requirementsAcknowledged
        case .terms: termsAccepted
        case .usage: usageChoiceMade
        case .weights: weightsReady
        case .codec: codecReady
        case .tuning: tuned
        }
    }

    /// The first step still to do; `finished` holds steps done this launch,
    /// such as a tuning that kept the default schedule. Before an engine is
    /// chosen, that is choosing one.
    func firstNeeded(finished: Set<SetupStep>) -> SetupStep? {
        guard let engine else { return .engine }
        return engine.setupSteps.first { !finished.contains($0) && !satisfies($0) }
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
    case engine(EngineOffer)
    case requirements(gaps: [String])
    case terms
    case usage
    case weights(Transfer)
    case codec(CodecStage)
    case tuning(detail: String)
    /// Setup just finished: offer the first line.
    case firstLine
    case ready
    case failed(SetupStep, reason: String)

    var step: SetupStep? {
        switch self {
        case .engine: .engine
        case .requirements: .requirements
        case .terms: .terms
        case .usage: .usage
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

    /// Nil until there are five seconds and 50 MB of samples; small files
    /// at the start of a download are all latency and would overstate the time.
    func secondsLeft(remaining: Int64) -> Double? {
        guard let first = samples.first, let last = samples.last, last.time - first.time >= 5, last.bytes - first.bytes >= 50_000_000 else { return nil }
        return Double(remaining) / (Double(last.bytes - first.bytes) / (last.time - first.time))
    }
}

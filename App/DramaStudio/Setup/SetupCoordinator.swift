// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import DramaEngine
import Foundation

/// What the coordinator does to the world. ``SetupServices/live(session:)``
/// wires the real downloads; tests pass closures.
struct SetupServices {
    var facts: @MainActor () -> SetupFacts
    var acknowledgeRequirements: @MainActor () -> Void
    var acceptTerms: @MainActor () -> Void
    /// Why the weights can't download now, such as too little free disk.
    var weightsBlocker: @MainActor () -> String?
    /// Where the weights download stands before it starts, counting partial files.
    var weightsProgress: @MainActor () -> DownloadProgress
    var downloadWeights: @Sendable (@escaping @Sendable (DownloadProgress) -> Void) async throws -> Void
    /// The Python command that builds the codec models, when this build has no codec bundle to download.
    var codecConversionCommand: @MainActor () -> String?
    /// Downloads and verifies the codec zip, and returns where it is.
    var downloadCodec: @Sendable (@escaping @Sendable (DownloadProgress) -> Void) async throws -> URL
    var installCodec: @Sendable (URL) async throws -> Void
    /// Loads the model once, which tunes the kernels and builds the packed-weight cache.
    var tune: @MainActor (@escaping @Sendable (String) -> Void) async throws -> Void
    /// Puts the real engine in the session once the weights and codec are in place.
    var attachEngine: @MainActor () -> Void
    var prepareFirstLine: @MainActor () -> Void
    var now: @Sendable () -> Double = { Date().timeIntervalSinceReferenceDate }
}

/// Drives the first run through ``SetupStage``. ``reconcile()`` reads what
/// is on disk and resumes from the first step still needed, so a quit, a
/// crash, a pause or a retry all converge on the same next step.
@MainActor @Observable
final class SetupCoordinator {
    private(set) var stage: SetupStage = .terms
    @ObservationIgnored private let services: SetupServices
    /// Steps done this launch that the facts may not show, such as a tuning that kept the default schedule.
    @ObservationIgnored private var finished: Set<SetupStep> = []
    @ObservationIgnored private var work: Task<Void, Never>?
    /// Bumped whenever new work starts, so late callbacks from old work are dropped.
    @ObservationIgnored private var run = 0
    @ObservationIgnored private var rate = RateEstimate()

    init(services: SetupServices) {
        self.services = services
        reconcile()
    }

    /// Nil when generation may run; otherwise the message the window, the API and the MCP server show.
    var generationRefusal: String? {
        switch stage {
        case .firstLine, .ready: nil
        case .requirements, .terms: TermsAgreement.refusal
        case .failed(let step, _) where step == .requirements || step == .terms: TermsAgreement.refusal
        default: "Drama Studio is still setting up (\(stage.step?.title.lowercased() ?? "setup")). Generation starts when its setup window finishes."
        }
    }

    func reconcile() {
        stop()
        let facts = services.facts()
        guard let step = facts.firstNeeded(finished: finished) else {
            services.attachEngine()
            if finished.isDisjoint(with: [.weights, .codec, .tuning]) {
                stage = .ready
            } else {
                services.prepareFirstLine()
                stage = .firstLine
            }
            return
        }
        switch step {
        case .requirements: stage = .requirements(gaps: facts.requirementGaps)
        case .terms: stage = .terms
        case .weights: startWeights()
        case .codec: startCodec()
        case .tuning: startTuning()
        }
    }

    func acknowledgeRequirements() {
        services.acknowledgeRequirements()
        reconcile()
    }

    func acceptTerms() {
        services.acceptTerms()
        reconcile()
    }

    /// Stops the download under way and keeps its partial file.
    func pause() {
        switch stage {
        case .weights(var transfer):
            stop()
            transfer.paused = true
            transfer.secondsLeft = nil
            stage = .weights(transfer)
        case .codec(.downloading(var transfer)):
            stop()
            transfer.paused = true
            transfer.secondsLeft = nil
            stage = .codec(.downloading(transfer))
        default:
            break
        }
    }

    /// Resumes a paused download, retries a failed step, or checks again for codec models built by hand.
    func resume() { reconcile() }

    /// Leaves the first-line offer for the studio.
    func finish() {
        if stage == .firstLine { stage = .ready }
    }

    private func stop() {
        run += 1
        work?.cancel()
        work = nil
        rate.reset()
    }

    /// Runs `body` as the current work. A failure lands on `step`'s failed
    /// stage; a cancellation does nothing, because whoever cancelled has
    /// already set the stage.
    private func perform(_ step: SetupStep, _ body: @escaping @MainActor (_ current: @escaping @MainActor () -> Bool) async throws -> Void) {
        let mine = run
        let current = { [weak self] in self?.run == mine }
        work = Task { [weak self] in
            do {
                try await body(current)
                guard let self, current() else { return }
                self.finished.insert(step)
                self.work = nil
                self.reconcile()
            } catch is CancellationError {
            } catch {
                guard let self, current() else { return }
                self.work = nil
                self.stage = .failed(step, reason: SessionModel.describe(error))
            }
        }
    }

    private func transferUpdate(_ progress: DownloadProgress) -> Transfer {
        rate.add(bytes: progress.doneBytes, at: services.now())
        return Transfer(progress: progress, secondsLeft: rate.secondsLeft(remaining: progress.totalBytes - progress.doneBytes))
    }

    private func startWeights() {
        if let blocker = services.weightsBlocker() {
            stage = .failed(.weights, reason: blocker)
            return
        }
        stage = .weights(Transfer(progress: services.weightsProgress()))
        let download = services.downloadWeights
        perform(.weights) { [weak self] current in
            try await download { progress in
                Task { @MainActor in
                    guard let self, current() else { return }
                    self.stage = .weights(self.transferUpdate(progress))
                }
            }
        }
    }

    private func startCodec() {
        if let command = services.codecConversionCommand() {
            stage = .codec(.needsConversion(command: command))
            return
        }
        stage = .codec(.downloading(Transfer(progress: DownloadProgress(totalBytes: 0, doneBytes: 0, filesDone: 0, fileCount: 1))))
        let download = services.downloadCodec, install = services.installCodec
        perform(.codec) { [weak self] current in
            let archive = try await download { progress in
                Task { @MainActor in
                    guard let self, current() else { return }
                    self.stage = .codec(.downloading(self.transferUpdate(progress)))
                }
            }
            guard let self, current() else { throw CancellationError() }
            self.stage = .codec(.unpacking)
            try await install(archive)
        }
    }

    private func startTuning() {
        stage = .tuning(detail: "Loading Drama 3")
        let tune = services.tune
        perform(.tuning) { [weak self] current in
            try await tune { detail in
                Task { @MainActor in
                    guard let self, current() else { return }
                    self.stage = .tuning(detail: detail)
                }
            }
        }
    }
}

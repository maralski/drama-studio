// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import DramaEngine
import Foundation
import Observation

/// Ming's install for Settings: what setup finds, and the optional download
/// of the pinned revision into ``AppHome/mingModels``. Drama 3 is the engine
/// the setup window requires; Ming is added here when the user wants it.
@MainActor @Observable
final class MingInstall {
    enum Phase: Equatable {
        case idle
        case downloading(DownloadProgress)
        case failed(String)
    }

    /// The last check; nil until the first finishes.
    private(set) var report: SetupReport?
    private(set) var phase = Phase.idle
    @ObservationIgnored private let session: SessionModel
    @ObservationIgnored private var download: Task<Void, Never>?

    init(session: SessionModel) { self.session = session }

    /// The checkpoint is complete; the word checker is the setup window's to report.
    var isReady: Bool {
        guard let report else { return false }
        return ![SetupItem.Component.weights, .tokenizer, .speakerEncoder].contains { report[$0].isMissing }
    }

    /// Checks the files again and, once they are complete, gives the session a Ming factory.
    func refresh() async {
        let path = ModelSetup.mingCustomPath
        report = await Task.detached { DramaSetup.check(ming: ModelSetup.mingCandidate(customPath: path)) }.value
        guard !ModelSetup.isPreviewEngine, let location = ModelSetup.mingLocation() else { return }
        await session.install(.ming, .ming(location: location))
    }

    /// Why the download can't start now: too little disk.
    var blocker: String? {
        let plan = SpeechEngineID.ming.install.plan
        let needed = plan.totalBytes - plan.bytesOnDisk(in: AppHome.mingModels)
        guard let free = ModelSetup.freeBytes(near: AppHome.support), free < needed else { return nil }
        return "Ming needs \(ModelSetup.gigabytes(needed)) free and this disk has \(ModelSetup.gigabytes(free)) free."
    }

    /// Downloads the pinned revision, verifying every file, then attaches it.
    /// Running it again resumes from the partial files.
    func startDownload() {
        guard download == nil else { return }
        if let blocker { return phase = .failed(blocker) }
        let run = PlanDownload(plan: SpeechEngineID.ming.install.plan, directory: AppHome.mingModels, downloader: FileDownloader())
        phase = .downloading(run.progress)
        download = Task {
            do {
                try await run.run { progress in Task { @MainActor in self.show(progress) } }
                phase = .idle
            } catch is CancellationError {
                phase = .idle
            } catch {
                phase = .failed("The Ming download stopped: \(error). Click Download to resume.")
            }
            download = nil
            await refresh()
        }
    }

    func cancelDownload() { download?.cancel() }

    private func show(_ progress: DownloadProgress) {
        if case .downloading = phase { phase = .downloading(progress) }
    }
}

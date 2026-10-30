// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import DramaEngine
import Foundation

enum ModelSetup {
    static let customDirectoryKey = "modelDirectory"
    static let licenseURL = TermsAgreement.licenseWebURL

    /// Set DRAMA_PREVIEW_ENGINE=1 to run the app on the mock instead of the Metal engine.
    static let isPreviewEngine = ProcessInfo.processInfo.environment["DRAMA_PREVIEW_ENGINE"] == "1"

    /// Disk the codec zip, the unpacked and compiled codec models and the packed-weight cache take beside the weights.
    static let reservedBytes: Int64 = 7_000_000_000

    static var customPath: String { AppHome.defaults.string(forKey: customDirectoryKey) ?? "" }

    /// The mock under the preview engine or before any weights are found.
    static func makeSynthesizer() -> any SpeechSynthesizer {
        if isPreviewEngine { return MockSynthesizer() }
        guard let location = location(customPath: customPath) else { return MockSynthesizer() }
        return DramaSynthesizer(location: location, codecDirectory: AppHome.codec, tuningDirectory: AppHome.tuning)
    }

    /// The folder chosen in Settings when there is one; then the setup
    /// window's download once every file is in place; then the newest
    /// Hugging Face cache snapshot, so `hf download` users skip the download.
    static func location(customPath: String) -> ModelLocation? {
        if !customPath.isEmpty { return ModelLocation(directory: URL(filePath: customPath, directoryHint: .isDirectory)) }
        if ModelDownloadPlan.drama3.missingFiles(in: AppHome.models).isEmpty { return ModelLocation(directory: AppHome.models) }
        return AppHome.usesHuggingFaceCache ? .huggingFaceCache() : nil
    }

    enum Status: Equatable {
        case ready(URL)
        case incomplete(URL)
        case missing

        var problem: String? {
            switch self {
            case .ready: nil
            case .incomplete(let folder): "The Drama 3 folder at \(folder.path) is missing files. Choose another folder, or use the default location and let the setup window download them."
            case .missing: "Drama 3 isn't downloaded yet. The setup window downloads it."
            }
        }
    }

    /// The download's state from the engine's own check. The codec models and
    /// the tuning are left out; the setup window handles them.
    static func status(customPath: String) -> Status {
        guard let location = location(customPath: customPath) else { return .missing }
        let report = DramaSetup.check(location: location, codecDirectory: AppHome.codec, tuningDirectory: AppHome.tuning)
        return [.weights, .tokenizer].contains { report[$0].isMissing } ? .incomplete(location.directory) : .ready(location.directory)
    }

    static func facts(terms: TermsAgreement, requirements: MacRequirements = .current) -> SetupFacts {
        var facts = SetupFacts(requirementGaps: requirements.gaps,
                               requirementsAcknowledged: AppHome.defaults.bool(forKey: MacRequirements.acknowledgedKey),
                               termsAccepted: terms.isAccepted)
        if isPreviewEngine {
            facts.weightsReady = true
            facts.codecReady = true
            facts.tuned = true
            return facts
        }
        let location = location(customPath: customPath)
        let report = DramaSetup.check(location: location, codecDirectory: AppHome.codec, tuningDirectory: AppHome.tuning)
        facts.weightsReady = location != nil && ![.weights, .tokenizer].contains { report[$0].isMissing }
        facts.codecReady = report[.codecModels].state == .ready
        facts.tuned = report[.tuning].state == .ready
        return facts
    }

    /// Why the weights can't download now: a Settings folder that is missing files, or too little disk.
    static func weightsBlocker() -> String? {
        if !customPath.isEmpty {
            return "The folder chosen in Settings, \(customPath), is missing Drama 3 files. Choose another folder in Settings, or click Use Default Location there so Drama Studio downloads the model, then click Try Again."
        }
        let plan = ModelDownloadPlan.drama3
        let needed = plan.totalBytes - plan.bytesOnDisk(in: AppHome.models) + reservedBytes
        guard let free = freeBytes(near: AppHome.support), free < needed else { return nil }
        return "Drama Studio needs \(gigabytes(needed)) free to download and set up the model, and this disk has \(gigabytes(free)) free. Free up \(gigabytes(needed - free)), then click Try Again."
    }

    static func freeBytes(near url: URL) -> Int64? {
        var folder = url
        while !FileManager.default.fileExists(atPath: folder.path), folder.pathComponents.count > 1 { folder.deleteLastPathComponent() }
        return (try? folder.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey]))?.volumeAvailableCapacityForImportantUsage
    }

    static func gigabytes(_ bytes: Int64) -> String { String(format: "%.1f GB", Double(bytes) / 1e9) }

    /// The Terminal commands that build the codec models, for builds with no codec bundle to download.
    static func codecConversionCommand() -> String {
        let python = DramaSetup.suggestedPythonEnvironment.appending(path: "bin/python")
        let location = location(customPath: customPath) ?? ModelLocation(directory: AppHome.models)
        return DramaSetup.pythonSetupCommand + "\n" + DramaSetup.codecConversionCommand(python: python, location: location, codecDirectory: AppHome.codec)
    }
}

extension SetupServices {
    /// The real downloads, into ``AppHome``.
    @MainActor
    static func live(session: SessionModel, server: APIServerController) -> SetupServices {
        let plan = ModelDownloadPlan.drama3
        let downloader = FileDownloader()
        let bundle = CodecBundle(info: Bundle.main.infoDictionary)
        let terms = TermsAgreement(defaults: AppHome.defaults)
        let weights = PlanDownload(plan: plan, directory: AppHome.models, downloader: downloader)
        let attach: @MainActor () -> any SpeechSynthesizer = {
            guard !ModelSetup.isPreviewEngine, let location = ModelSetup.location(customPath: ModelSetup.customPath) else { return session.synthesizer }
            if let engine = session.synthesizer as? DramaSynthesizer, engine.location == location { return engine }
            let engine = DramaSynthesizer(location: location, codecDirectory: AppHome.codec, tuningDirectory: AppHome.tuning)
            session.replaceSynthesizer(engine)
            server.restart()
            return engine
        }
        return SetupServices(
            facts: {
                var facts = ModelSetup.facts(terms: terms)
                facts.usageChoiceMade = UsageMonitoring.shared.choiceMade
                return facts
            },
            acknowledgeRequirements: { AppHome.defaults.set(true, forKey: MacRequirements.acknowledgedKey) },
            acceptTerms: { terms.accept() },
            weightsBlocker: { ModelSetup.weightsBlocker() },
            weightsProgress: { weights.progress },
            downloadWeights: { report in try await weights.run(progress: report) },
            codecConversionCommand: { bundle == nil ? ModelSetup.codecConversionCommand() : nil },
            downloadCodec: { report in
                guard let bundle else { throw CodecBundleError.missingModels(CodecBundle.packageNames) }
                let throttle = ProgressThrottle()
                return try await bundle.download(into: AppHome.downloads, with: downloader) { bytes, total in
                    guard throttle.allows() || bytes == total else { return }
                    report(DownloadProgress(totalBytes: total ?? 0, doneBytes: bytes, file: bundle.url.lastPathComponent,
                                            fileBytes: bytes, fileSize: total ?? 0, filesDone: 0, fileCount: 1))
                }
            },
            installCodec: { archive in
                try await Task.detached {
                    try CodecBundle.install(archive: archive, into: AppHome.codec)
                    try? FileManager.default.removeItem(at: archive)
                }.value
            },
            tune: { detail in try await attach().load(onPhase: detail) },
            attachEngine: { _ = attach() },
            prepareFirstLine: { session.prepareFirstLine() },
            chooseUsage: { UsageMonitoring.shared.choose(enabled: $0) },
            recordUsage: { UsageMonitoring.shared.record($0) }
        )
    }
}

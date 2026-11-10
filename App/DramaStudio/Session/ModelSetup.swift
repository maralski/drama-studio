// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import DramaEngine
import Foundation

enum ModelSetup {
    static let customDirectoryKey = "modelDirectory"

    /// Set DRAMA_PREVIEW_ENGINE=1 to run the app on the mock instead of the Metal engine.
    static let isPreviewEngine = ProcessInfo.processInfo.environment["DRAMA_PREVIEW_ENGINE"] == "1"

    /// Disk an engine's setup takes beside its weights: for one with a codec,
    /// the codec zip, the unpacked and compiled codec models and the packed-weight cache.
    static func reservedBytes(for engine: SpeechEngineID) -> Int64 {
        engine.install.components.contains(.codecModels) ? 7_000_000_000 : 0
    }

    static var customPath: String { AppHome.defaults.string(forKey: customDirectoryKey) ?? "" }

    /// The engine new takes use: the `speechEngine` default, or `DRAMA_ENGINE=ming` for test runs.
    static let engineKey = "speechEngine"
    static var selectedEngine: SpeechEngineID {
        let raw = ProcessInfo.processInfo.environment["DRAMA_ENGINE"] ?? AppHome.defaults.string(forKey: engineKey) ?? ""
        return SpeechEngineID(rawValue: raw) ?? .drama3
    }

    /// The folder chosen for Ming in Settings; empty for the default search.
    static let mingDirectoryKey = "mingDirectory"
    static var mingCustomPath: String { AppHome.defaults.string(forKey: mingDirectoryKey) ?? "" }

    /// Where Ming is, complete or not, so Settings can say what it lacks:
    /// `MING_WEIGHTS`; else the folder chosen in Settings; else the app's own
    /// download once it is complete; else the newest Hugging Face cache
    /// snapshot; else a download under way. Nil when there is none of these.
    static func mingCandidate(customPath: String) -> MingLocation? {
        if ProcessInfo.processInfo.environment["MING_WEIGHTS"] != nil { return MingLocation.find() }
        if !customPath.isEmpty { return MingLocation(directory: URL(filePath: customPath, directoryHint: .isDirectory)) }
        let downloaded = MingLocation(directory: AppHome.models(for: .ming), source: .appDownload)
        if downloaded.isComplete { return downloaded }
        if AppHome.usesHuggingFaceCache, let cached = MingLocation.find() { return cached }
        return FileManager.default.fileExists(atPath: downloaded.directory.path) ? downloaded : nil
    }

    /// The Ming checkpoint the app runs; nil unless complete.
    static func mingLocation() -> MingLocation? {
        mingCandidate(customPath: mingCustomPath).flatMap { $0.isComplete ? $0 : nil }
    }

    /// How to make `engine` from the files on this Mac. Nil while its files
    /// are incomplete, and nil until the terms are agreed to for an engine
    /// whose license asks for that, so such an engine cannot be selected or
    /// run before then. The preview engine stands in for the files only.
    static func factory(for engine: SpeechEngineID, terms: TermsAgreement = TermsAgreement(defaults: AppHome.defaults)) -> SpeechEngines.Factory? {
        guard terms.allows(engine) else { return nil }
        if isPreviewEngine { return .mock(engine) }
        switch engine {
        case .drama3:
            let onDisk = files(of: .drama3)
            guard onDisk.weights, onDisk.codec, let location = location(customPath: customPath) else { return nil }
            return .drama3(location: location, codecDirectory: AppHome.codec, tuningDirectory: AppHome.tuning)
        case .ming:
            return mingLocation().map { .ming(location: $0) }
        }
    }

    /// Every installed engine behind one synthesizer.
    static func makeEngines(library: VoiceLibrary) -> SpeechEngines {
        let pending = SetupEngineChoice.stored(in: AppHome.defaults) ?? MacRequirements.current.recommendedEngine
        let factories = SpeechEngineID.allCases.reduce(into: [:]) { $0[$1] = factory(for: $1) }
        return registry(of: factories, preferred: selectedEngine, pending: pending, voices: library)
    }

    /// The registry for `factories`, one per engine that can run. `preferred`
    /// is selected when it can run, else the first engine that can. With no
    /// engine yet, a placeholder for `pending`, the engine setup is about to
    /// install, holds the selection; it is not offered and never speaks.
    static func registry(of factories: [SpeechEngineID: SpeechEngines.Factory], preferred: SpeechEngineID, pending: SpeechEngineID,
                         voices: (any VoiceEncodingStore)?) -> SpeechEngines {
        var factories = factories
        let selected = factories[preferred] != nil ? preferred : SpeechEngineID.allCases.first { factories[$0] != nil } ?? pending
        if factories.isEmpty { factories[pending] = .placeholder(pending) }
        return try! SpeechEngines(selected: selected, factories: factories, voices: voices)
    }

    /// The folder chosen in Settings when there is one; then the setup
    /// window's download once every file is in place; then the newest
    /// Hugging Face cache snapshot, so `hf download` users skip the download.
    static func location(customPath: String) -> ModelLocation? {
        if !customPath.isEmpty { return ModelLocation(directory: URL(filePath: customPath, directoryHint: .isDirectory)) }
        let downloaded = AppHome.models(for: .drama3)
        if ModelDownloadPlan.drama3.missingFiles(in: downloaded).isEmpty { return ModelLocation(directory: downloaded) }
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

    /// What of an engine's setup is in place on this Mac.
    struct Files: Equatable {
        var weights = false
        var codec = false
        var tuned = false
    }

    /// An engine without a codec or tuning reports them in place.
    static func files(of engine: SpeechEngineID) -> Files {
        if isPreviewEngine { return Files(weights: true, codec: true, tuned: true) }
        switch engine {
        case .drama3:
            let location = location(customPath: customPath)
            let report = DramaSetup.check(location: location, codecDirectory: AppHome.codec, tuningDirectory: AppHome.tuning)
            return Files(weights: location != nil && ![.weights, .tokenizer].contains { report[$0].isMissing },
                         codec: CodecBundle(info: Bundle.main.infoDictionary)?.isInstalled(in: AppHome.codec) ?? (report[.codecModels].state == .ready),
                         tuned: report[.tuning].state == .ready)
        case .ming:
            return Files(weights: mingLocation() != nil, codec: true, tuned: true)
        }
    }

    static func facts(terms: TermsAgreement, engine: SpeechEngineID?, requirements: MacRequirements = .current) -> SetupFacts {
        let onDisk = Dictionary(uniqueKeysWithValues: SpeechEngineID.allCases.map { ($0, files(of: $0)) })
        let downloaded = isPreviewEngine ? [] : Set(onDisk.filter { $0.value.weights && $0.value.codec }.keys)
        var facts = SetupFacts(engine: engine, offer: requirements.offer(downloaded: downloaded), termsAccepted: terms.isAccepted)
        guard let engine = facts.engine, let mine = onDisk[engine] else { return facts }
        facts.requirementsAcknowledged = AppHome.defaults.bool(forKey: MacRequirements.acknowledgedKey(for: engine))
        facts.weightsReady = mine.weights
        facts.codecReady = mine.codec
        facts.tuned = mine.tuned
        return facts
    }

    /// A folder someone named for `engine`, which the app must not download
    /// into, with the words that say who named it.
    private static func namedFolder(for engine: SpeechEngineID) -> (path: String, fix: String)? {
        let settings = "Choose another folder in Settings, or click Use Default Location there so Drama Studio downloads the model, then click Try Again."
        switch engine {
        case .drama3:
            return customPath.isEmpty ? nil : ("The folder chosen in Settings, \(customPath),", settings)
        case .ming:
            if let named = ProcessInfo.processInfo.environment["MING_WEIGHTS"] {
                return ("The folder MING_WEIGHTS names, \(named),", "Put the whole checkpoint there or unset MING_WEIGHTS, then open Drama Studio again.")
            }
            return mingCustomPath.isEmpty ? nil : ("The folder chosen in Settings, \(mingCustomPath),", settings)
        }
    }

    /// Why the weights can't download now: a named folder that is missing files, or too little disk.
    static func weightsBlocker(for engine: SpeechEngineID) -> String? {
        let name = engine.capabilities.name
        if let named = namedFolder(for: engine) { return "\(named.path) is missing \(name) files. \(named.fix)" }
        let plan = engine.install.plan
        let needed = plan.totalBytes - plan.bytesOnDisk(in: AppHome.models(for: engine)) + reservedBytes(for: engine)
        guard let free = freeBytes(near: AppHome.support), free < needed else { return nil }
        return "Drama Studio needs \(gigabytes(needed)) free to download and set up \(name), and this disk has \(gigabytes(free)) free. Free up \(gigabytes(needed - free)), then click Try Again."
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
        let location = location(customPath: customPath) ?? ModelLocation(directory: AppHome.models(for: .drama3))
        return DramaSetup.pythonSetupCommand + "\n" + DramaSetup.codecConversionCommand(python: python, location: location, codecDirectory: AppHome.codec)
    }
}

extension SetupServices {
    /// The real downloads, into ``AppHome``.
    @MainActor
    static func live(session: SessionModel, server: APIServerController) -> SetupServices {
        let downloader = FileDownloader()
        let bundle = CodecBundle(info: Bundle.main.infoDictionary)
        let terms = TermsAgreement(defaults: AppHome.defaults)
        let choice = SetupEngineChoice(defaults: AppHome.defaults)
        let weights: @Sendable (SpeechEngineID) -> PlanDownload = {
            PlanDownload(plan: $0.install.plan, directory: AppHome.models(for: $0), downloader: downloader)
        }
        /// Gives the registry the engine's factory for the files now in
        /// place, and selects the engine when the user chose it this launch.
        let attach: @MainActor (SpeechEngineID) async -> Void = { engine in
            guard let factory = ModelSetup.factory(for: engine, terms: terms) else { return }
            await session.install(engine, factory)
            if choice.madeThisLaunch, choice.engine == engine { session.engine = engine }
        }
        return SetupServices(
            facts: {
                var facts = ModelSetup.facts(terms: terms, engine: choice.engine)
                facts.usageChoiceMade = UsageMonitoring.shared.choiceMade
                return facts
            },
            chooseEngine: { choice.set($0) },
            acknowledgeRequirements: { AppHome.defaults.set(true, forKey: MacRequirements.acknowledgedKey(for: $0)) },
            acceptTerms: { terms.accept() },
            weightsBlocker: { ModelSetup.weightsBlocker(for: $0) },
            weightsProgress: { weights($0).progress },
            downloadWeights: { engine, report in try await weights(engine).run(progress: report) },
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
                guard let bundle else { throw CodecBundleError.missingModels(CodecBundle.packageNames) }
                try await Task.detached {
                    try bundle.install(archive: archive, into: AppHome.codec)
                    try? FileManager.default.removeItem(at: archive)
                }.value
            },
            tune: { engine, detail in
                await attach(engine)
                try await session.engines.load(engine, onPhase: detail)
            },
            attachEngine: { engine in Task { await attach(engine) } },
            prepareFirstLine: { session.prepareFirstLine() },
            chooseUsage: { UsageMonitoring.shared.choose(enabled: $0) },
            recordUsage: { UsageMonitoring.shared.record($0) }
        )
    }
}

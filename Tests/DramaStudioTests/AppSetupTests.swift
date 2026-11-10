// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import DramaEngine
import DramaServer
import XCTest
@testable import DramaStudio

/// A Mac whose disk and settings live in memory. Downloads finish when the
/// test says so, or block until cancelled.
@MainActor
final class FakeMac {
    /// The engine setup installs; tests of the Drama 3 path start with it chosen.
    var engine: SpeechEngineID? = .drama3
    /// What is on disk for Drama 3, and the facts no engine owns.
    var facts = SetupFacts()
    var mingOnDisk = false
    var usageChoiceMade = true
    var usageEnabled = false
    var blocker: String?
    var conversionCommand: String?
    var weightFailures: [Error] = []
    var holdWeights = false
    var tuningWritesSchedule = true
    private(set) var calls: [String] = []

    var services: SetupServices {
        SetupServices(
            facts: { [unowned self] in
                var result = facts
                result.engine = engine
                result.usageChoiceMade = usageChoiceMade
                if engine == .ming { result.weightsReady = mingOnDisk }
                return result
            },
            chooseEngine: { [unowned self] in calls.append("choose \($0?.rawValue ?? "again")"); engine = $0 },
            acknowledgeRequirements: { [unowned self] _ in calls.append("acknowledge"); facts.requirementsAcknowledged = true },
            acceptTerms: { [unowned self] in calls.append("accept"); facts.termsAccepted = true },
            weightsBlocker: { [unowned self] _ in blocker },
            weightsProgress: { _ in DownloadProgress(totalBytes: 100, doneBytes: 10, filesDone: 1, fileCount: 3) },
            downloadWeights: { [unowned self] engine, report in
                try await self.runWeights(engine, report)
            },
            codecConversionCommand: { [unowned self] in conversionCommand },
            downloadCodec: { [unowned self] report in
                report(DownloadProgress(totalBytes: 50, doneBytes: 50, filesDone: 0, fileCount: 1))
                await self.record("codec download")
                return URL(filePath: "/tmp/codec.zip")
            },
            installCodec: { [unowned self] _ in
                await self.record("codec install")
                await MainActor.run { self.facts.codecReady = true }
            },
            tune: { [unowned self] _, detail in
                calls.append("tune")
                detail("Tuning kernels")
                if tuningWritesSchedule { facts.tuned = true }
            },
            attachEngine: { [unowned self] engine in calls.append(engine == .drama3 ? "attach" : "attach \(engine.rawValue)") },
            prepareFirstLine: { [unowned self] in calls.append("first line") },
            chooseUsage: { [unowned self] in usageEnabled = $0; usageChoiceMade = true }
        )
    }

    func record(_ call: String) { calls.append(call) }

    private func runWeights(_ engine: SpeechEngineID, _ report: @escaping @Sendable (DownloadProgress) -> Void) async throws {
        calls.append(engine == .drama3 ? "weights" : "weights \(engine.rawValue)")
        if !weightFailures.isEmpty { throw weightFailures.removeFirst() }
        report(DownloadProgress(totalBytes: 100, doneBytes: 60, file: "model-00001-of-00008.safetensors", fileBytes: 50, fileSize: 90, filesDone: 1, fileCount: 3))
        while holdWeights { try await Task.sleep(for: .milliseconds(5)) }
        if engine == .ming { mingOnDisk = true } else { facts.weightsReady = true }
    }
}

@MainActor
final class AppSetupTests: XCTestCase {
    func settle(_ setup: SetupCoordinator, until done: (SetupStage) -> Bool, file: StaticString = #filePath, line: UInt = #line) async {
        for _ in 0..<400 {
            if done(setup.stage) { return }
            try? await Task.sleep(for: .milliseconds(5))
        }
        XCTFail("stage stayed \(setup.stage)", file: file, line: line)
    }

    func testAFreshMacWalksTermsWeightsCodecTuningThenOffersTheFirstLine() async {
        let mac = FakeMac()
        let setup = SetupCoordinator(services: mac.services)
        XCTAssertEqual(setup.stage, .terms)
        XCTAssertEqual(setup.generationRefusal, TermsAgreement.refusal)
        XCTAssertTrue(mac.calls.isEmpty, "nothing downloads before the terms are agreed to")

        mac.holdWeights = true
        setup.acceptTerms()
        guard case .weights(let start) = setup.stage else { return XCTFail("\(setup.stage)") }
        XCTAssertEqual(start.progress.doneBytes, 10, "a resumed download starts from the bytes on disk")
        await settle(setup) { if case .weights(let t) = $0 { t.progress.doneBytes == 60 } else { false } }
        XCTAssertNotNil(setup.generationRefusal)
        XCTAssertFalse(setup.generationRefusal == TermsAgreement.refusal, "after agreeing, the refusal says setup is still running")

        mac.holdWeights = false
        await settle(setup) { $0 == .firstLine }
        XCTAssertEqual(mac.calls, ["accept", "weights", "codec download", "codec install", "tune", "attach", "first line"])
        XCTAssertNil(setup.generationRefusal)
        setup.finish()
        XCTAssertEqual(setup.stage, .ready)
    }

    func testAMacThatHasEverythingGoesStraightToTheStudioAfterTheTerms() {
        let mac = FakeMac()
        mac.facts = SetupFacts(weightsReady: true, codecReady: true, tuned: true)
        let setup = SetupCoordinator(services: mac.services)
        XCTAssertEqual(setup.stage, .terms)
        setup.acceptTerms()
        XCTAssertEqual(setup.stage, .ready, "no first-line offer when nothing was installed")
        XCTAssertEqual(mac.calls, ["accept", "attach"])
    }

    func testAShortfallIsShownBeforeTheTermsAndCanBeAccepted() {
        let mac = FakeMac()
        mac.facts.offer.gaps[.drama3] = ["This Mac has 16 GB of memory."]
        let setup = SetupCoordinator(services: mac.services)
        XCTAssertEqual(setup.stage, .requirements(gaps: ["This Mac has 16 GB of memory."]))
        XCTAssertEqual(setup.generationRefusal, TermsAgreement.refusal)
        setup.acknowledgeRequirements()
        XCTAssertEqual(setup.stage, .terms)
    }

    func testAFailedDownloadLandsOnFailedAndTryAgainResumes() async {
        let mac = FakeMac()
        mac.facts.termsAccepted = true
        mac.weightFailures = [PlanDownloadError.fileFailed(path: "model-00003-of-00008.safetensors", attempts: 4, last: DownloadError.httpStatus(503, URL(string: "https://huggingface.co")!))]
        let setup = SetupCoordinator(services: mac.services)
        await settle(setup) { if case .failed = $0 { true } else { false } }
        guard case .failed(let step, let reason) = setup.stage else { return XCTFail() }
        XCTAssertEqual(step, .weights)
        XCTAssertTrue(reason.contains("model-00003-of-00008.safetensors failed 4 times"), reason)
        XCTAssertNotNil(setup.generationRefusal)

        setup.resume()
        await settle(setup) { $0 == .firstLine }
        XCTAssertEqual(mac.calls.filter { $0 == "weights" }.count, 2)
    }

    func testPauseKeepsTheTransferAndResumeStartsItAgain() async {
        let mac = FakeMac()
        mac.facts.termsAccepted = true
        mac.holdWeights = true
        let setup = SetupCoordinator(services: mac.services)
        await settle(setup) { if case .weights(let t) = $0 { t.progress.doneBytes == 60 } else { false } }
        setup.pause()
        guard case .weights(let paused) = setup.stage else { return XCTFail("\(setup.stage)") }
        XCTAssertTrue(paused.paused)
        XCTAssertEqual(paused.progress.doneBytes, 60)
        try? await Task.sleep(for: .milliseconds(50))
        XCTAssertEqual(setup.stage, .weights(paused), "the cancelled download does not turn into a failure or move on")

        mac.holdWeights = false
        setup.resume()
        await settle(setup) { $0 == .firstLine }
        XCTAssertEqual(mac.calls.filter { $0 == "weights" }.count, 2)
    }

    func testATuningThatKeepsTheDefaultScheduleStillFinishes() async {
        let mac = FakeMac()
        mac.facts = SetupFacts(termsAccepted: true, weightsReady: true, codecReady: true, tuned: false)
        mac.tuningWritesSchedule = false
        let setup = SetupCoordinator(services: mac.services)
        await settle(setup) { $0 == .firstLine }
        XCTAssertEqual(mac.calls.filter { $0 == "tune" }.count, 1)
    }

    func testADevelopmentBuildShowsTheConversionCommandAndChecksAgain() {
        let mac = FakeMac()
        mac.facts = SetupFacts(termsAccepted: true, weightsReady: true, tuned: true)
        mac.conversionCommand = "python convert_codec.py"
        let setup = SetupCoordinator(services: mac.services)
        XCTAssertEqual(setup.stage, .codec(.needsConversion(command: "python convert_codec.py")))
        mac.facts.codecReady = true
        setup.resume()
        XCTAssertEqual(setup.stage, .ready)
    }

    func testTooLittleDiskFailsBeforeAnyDownload() {
        let mac = FakeMac()
        mac.facts.termsAccepted = true
        mac.blocker = "Drama Studio needs 48.5 GB free."
        let setup = SetupCoordinator(services: mac.services)
        XCTAssertEqual(setup.stage, .failed(.weights, reason: "Drama Studio needs 48.5 GB free."))
        XCTAssertFalse(mac.calls.contains("weights"))
    }

    func testTermsAreAskedAgainWhenTheirVersionChanges() throws {
        let defaults = try XCTUnwrap(UserDefaults(suiteName: "AppSetupTests-\(UUID().uuidString)"))
        let terms = TermsAgreement(defaults: defaults)
        XCTAssertFalse(terms.isAccepted)
        defaults.set("fish-audio-community-license-0.9", forKey: TermsAgreement.versionKey)
        XCTAssertFalse(terms.isAccepted, "an older version does not count")
        let when = Date(timeIntervalSince1970: 1_800_000_000)
        terms.accept(at: when)
        XCTAssertTrue(terms.isAccepted)
        XCTAssertEqual(defaults.string(forKey: TermsAgreement.versionKey), TermsAgreement.version)
        XCTAssertEqual(terms.acceptedDate, when)
    }

    func testTheWindowAndTheAPIRefuseUntilTheTermsAreAgreedTo() async throws {
        let directory = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let mac = FakeMac()
        mac.facts = SetupFacts(weightsReady: true, codecReady: true, tuned: true)
        let setup = SetupCoordinator(services: mac.services)
        let session = SessionModel(synthesizer: MockSynthesizer(pace: .instant), library: VoiceLibrary(directory: directory))
        session.playsWhileGenerating = false
        session.generationRefusal = { [setup] in setup.generationRefusal }
        let service = APIService(synthesizer: session.synthesizer, voices: session.library, token: nil, version: "test",
                                 refusal: { [session] in await session.generationRefusal() })

        session.script = "Hello there."
        session.generate()
        XCTAssertEqual(session.state, .failed(TermsAgreement.refusal))
        XCTAssertTrue(session.audio.isEmpty)
        do {
            try await session.encodeVoice(clip: directory.appending(path: "clip.wav"), transcript: "Hi.", name: "Ann")
            XCTFail("cloning must be refused too")
        } catch let refused as GenerationRefused {
            XCTAssertEqual(refused.message, TermsAgreement.refusal)
        }
        let denied = await service.respond(to: HTTPRequest(method: "POST", path: "/v1/tts", body: Data("{\"text\":\"Hello.\"}".utf8)))
        XCTAssertEqual(denied.status, .forbidden)
        XCTAssertEqual(String(decoding: denied.bodyData ?? Data(), as: UTF8.self), TermsAgreement.refusal + "\n")

        setup.acceptTerms()
        let allowed = await service.respond(to: HTTPRequest(method: "POST", path: "/v1/tts", body: Data("{\"text\":\"Hello.\"}".utf8)))
        XCTAssertEqual(allowed.status, .ok)
        session.generate()
        await settle(setup) { _ in !session.state.isGenerating }
        XCTAssertFalse(session.audio.isEmpty)
    }

    func testEachEnginesStepsComeFromItsLicenseAndComponents() {
        XCTAssertEqual(SpeechEngineID.drama3.setupSteps, [.engine, .requirements, .terms, .usage, .weights, .codec, .tuning])
        XCTAssertEqual(SpeechEngineID.ming.setupSteps, [.engine, .requirements, .usage, .weights], "Ming has no terms to agree to, no codec and no tuning")
    }

    func testAMacBelowDrama3sMemoryIsOfferedMingAndACapableOneDrama3() {
        let small = MacRequirements(appleSilicon: true, memoryBytes: 16 << 30)
        XCTAssertEqual(small.recommendedEngine, .ming)
        XCTAssertEqual(small.gaps(for: .ming), [], "Ming has no known memory floor")
        XCTAssertEqual(small.gaps(for: .drama3).count, 1)
        XCTAssertTrue(small.gaps(for: .drama3)[0].hasPrefix("This Mac has 16 GB of memory. Drama 3 maps about 41 GB"), small.gaps(for: .drama3)[0])
        let offer = small.offer(downloaded: [.ming])
        XCTAssertEqual(offer.recommended, .ming)
        XCTAssertEqual(Set(offer.gaps.keys), [.drama3], "the offer explains why next to the engine that falls short")
        XCTAssertEqual(offer.downloaded, [.ming])

        XCTAssertEqual(MacRequirements(appleSilicon: true, memoryBytes: 63 << 30).recommendedEngine, .ming)
        let capable = MacRequirements(appleSilicon: true, memoryBytes: 64 << 30)
        XCTAssertEqual(capable.recommendedEngine, .drama3)
        XCTAssertEqual(capable.offer(downloaded: []).gaps, [:])
        XCTAssertEqual(MacRequirements(appleSilicon: false, memoryBytes: 128 << 30).gaps(for: .ming).count, 1, "Ming needs Apple silicon too")
    }

    func testAFreshMacChoosesItsEngineBeforeAnythingElse() {
        let mac = FakeMac()
        mac.engine = nil
        mac.facts.offer = MacRequirements(appleSilicon: true, memoryBytes: 16 << 30).offer(downloaded: [])
        let setup = SetupCoordinator(services: mac.services)
        XCTAssertEqual(setup.stage, .engine(mac.facts.offer))
        XCTAssertEqual(setup.steps, SpeechEngineID.ming.setupSteps, "the strip shows the recommended engine's steps")
        XCTAssertNotNil(setup.generationRefusal)
        XCTAssertNotEqual(setup.generationRefusal, TermsAgreement.refusal, "no terms are pending before an engine is chosen")
        XCTAssertTrue(mac.calls.isEmpty, "nothing downloads before the choice")
    }

    func testAMingOnlySetupFinishesWithoutTheTermsTheCodecOrDrama3() async {
        let mac = FakeMac()
        mac.engine = nil
        mac.usageChoiceMade = false
        let setup = SetupCoordinator(services: mac.services)
        setup.chooseEngine(.ming)
        XCTAssertEqual(setup.stage, .usage)
        setup.chooseUsage(enabled: false)
        await settle(setup) { $0 == .firstLine }
        XCTAssertEqual(mac.calls, ["choose ming", "weights ming", "attach ming", "first line"])
        XCTAssertFalse(mac.facts.termsAccepted, "the Ming path never records an agreement to the Fish Audio terms")
        XCTAssertNil(setup.generationRefusal)
        setup.finish()
        XCTAssertEqual(setup.stage, .ready)
        XCTAssertEqual(SetupCoordinator(services: mac.services).stage, .ready, "the next launch goes straight to the studio")
    }

    func testMingAlreadyOnThisMacNeedsNoDownload() {
        let mac = FakeMac()
        mac.engine = nil
        mac.mingOnDisk = true
        let setup = SetupCoordinator(services: mac.services)
        setup.chooseEngine(.ming)
        XCTAssertEqual(setup.stage, .ready)
        XCTAssertEqual(mac.calls, ["choose ming", "attach ming"])
    }

    func testAddingDrama3ToAMingSetupAsksForTheTermsBeforeItDownloadsOrSpeaks() async {
        let mac = FakeMac()
        mac.engine = .ming
        mac.mingOnDisk = true
        let setup = SetupCoordinator(services: mac.services)
        XCTAssertEqual(setup.stage, .ready)

        setup.chooseEngine(.drama3)
        XCTAssertEqual(setup.stage, .terms)
        XCTAssertEqual(setup.generationRefusal, TermsAgreement.refusal)
        try? await Task.sleep(for: .milliseconds(30))
        XCTAssertEqual(mac.calls, ["attach ming", "choose drama3"], "Drama 3 neither downloads nor attaches before the terms are agreed to")

        setup.chooseAnotherEngine()
        guard case .engine = setup.stage else { return XCTFail("\(setup.stage)") }
        setup.chooseEngine(.ming)
        XCTAssertEqual(setup.stage, .ready, "backing out returns to the engine already installed")
        XCTAssertFalse(mac.facts.termsAccepted)

        setup.chooseEngine(.drama3)
        setup.acceptTerms()
        await settle(setup) { $0 == .firstLine }
        XCTAssertEqual(Array(mac.calls.suffix(7)), ["accept", "weights", "codec download", "codec install", "tune", "attach", "first line"])
    }

    func testChoosingAnotherEngineStopsTheDownloadAndForgetsWhatThisLaunchFinished() async {
        let mac = FakeMac()
        mac.facts.termsAccepted = true
        mac.holdWeights = true
        let setup = SetupCoordinator(services: mac.services)
        await settle(setup) { if case .weights(let t) = $0 { t.progress.doneBytes == 60 } else { false } }
        setup.chooseAnotherEngine()
        guard case .engine = setup.stage else { return XCTFail("\(setup.stage)") }
        try? await Task.sleep(for: .milliseconds(50))
        guard case .engine = setup.stage else { return XCTFail("the cancelled download moved the stage to \(setup.stage)") }
        XCTAssertFalse(mac.facts.weightsReady)

        mac.holdWeights = false
        setup.chooseEngine(.ming)
        await settle(setup) { $0 == .firstLine }
        XCTAssertEqual(mac.calls.filter { $0.hasPrefix("weights") }, ["weights", "weights ming"], "Ming's own download runs; Drama 3's does not resume")
    }

    func testAnInstallFromBeforeTheChoiceCountsAsHavingChosenDrama3() throws {
        let defaults = try XCTUnwrap(UserDefaults(suiteName: "AppSetupTests-\(UUID().uuidString)"))
        XCTAssertNil(SetupEngineChoice(defaults: defaults).engine, "a fresh Mac has not chosen")
        defaults.set("fish-audio-community-license-0.9", forKey: TermsAgreement.versionKey)
        let choice = SetupEngineChoice(defaults: defaults)
        XCTAssertEqual(choice.engine, .drama3, "agreeing to any version of the terms was setting up Drama 3")
        XCTAssertFalse(choice.madeThisLaunch)
        choice.set(nil)
        XCTAssertNil(choice.engine)
        XCTAssertNil(SetupEngineChoice(defaults: defaults).engine, "choosing again is not answered by the old agreement, at the next launch either")
        choice.set(.ming)
        XCTAssertEqual(choice.engine, .ming)
        XCTAssertTrue(choice.madeThisLaunch)
        let relaunched = SetupEngineChoice(defaults: defaults)
        XCTAssertEqual(relaunched.engine, .ming)
        XCTAssertFalse(relaunched.madeThisLaunch, "a later launch leaves the engine picker's selection alone")
    }

    func testDrama3HasNoFactoryUntilItsTermsAreAgreedTo() throws {
        let defaults = try XCTUnwrap(UserDefaults(suiteName: "AppSetupTests-\(UUID().uuidString)"))
        let terms = TermsAgreement(defaults: defaults)
        XCTAssertFalse(terms.allows(.drama3))
        XCTAssertTrue(terms.allows(.ming), "Ming's license asks for no agreement")
        XCTAssertNil(ModelSetup.factory(for: .drama3, terms: terms), "whatever is on disk, Drama 3 can't be built before the agreement")
        terms.accept()
        XCTAssertTrue(terms.allows(.drama3))
    }

    func testAMingOnlyInstallRunsOnMingAndDoesNotOfferDrama3() async throws {
        let directory = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let library = VoiceLibrary(directory: directory)
        let engines = ModelSetup.registry(of: [.ming: .mock(.ming, pace: .instant)], preferred: .drama3, pending: .drama3, voices: library)
        XCTAssertEqual(engines.selected, .ming, "the preferred engine is not installed, so the installed one is used")
        let session = SessionModel(synthesizer: engines, library: library, pronunciationURL: directory.appending(path: "pronunciation.json"))
        session.playsWhileGenerating = false
        XCTAssertEqual(session.availableEngines, [.ming])
        session.engine = .drama3
        XCTAssertEqual(session.engine, .ming)

        session.script = "Hello there."
        session.generate()
        for _ in 0..<400 where session.state.isGenerating || session.audio.isEmpty { try? await Task.sleep(for: .milliseconds(5)) }
        XCTAssertFalse(session.audio.isEmpty)
        XCTAssertEqual(session.takes.first?.take?.context.engine, .ming)
    }

    func testBeforeAnyEngineIsInstalledAPlaceholderWaitsForTheFirstOne() async throws {
        let directory = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let library = VoiceLibrary(directory: directory)
        let engines = ModelSetup.registry(of: [:], preferred: .drama3, pending: .ming, voices: library)
        let session = SessionModel(synthesizer: engines, library: library, pronunciationURL: directory.appending(path: "pronunciation.json"))
        XCTAssertEqual(session.availableEngines, [], "no engine shows as installed during a first run")
        XCTAssertEqual(session.engine, .ming)

        await session.install(.drama3, .mock(.drama3, pace: .instant))
        XCTAssertEqual(session.availableEngines, [.drama3])
        XCTAssertEqual(session.engine, .drama3, "the first engine installed is the one new takes use")
    }

    func testAVoiceKeptOnlyAsDrama3CodesNeedsDrama3SetUpBeforeAnotherEngineCanUseIt() async throws {
        let directory = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        var codecAllowed = false
        let library = VoiceLibrary(directory: directory, codecDirectory: directory.appending(path: "NoCodec")) { codecAllowed }
        let old = ReferenceVoice(name: "Ann", transcript: "Hi.", encodings: [.drama3(codes: Array(repeating: [1, 2], count: 10))])
        let recorded = ReferenceVoice(name: "Ben", transcript: "Hi.", encodings: [.drama3(codes: Array(repeating: [1, 2], count: 10))])
        try library.add(old, clip: nil)
        try library.add(recorded, clip: [0.1, 0.2, 0.3])

        do {
            _ = try await library.clip(for: old.id)
            XCTFail("the Drama 3 codec must not run before Drama 3 is set up")
        } catch let error as ClipNeedsDrama3 {
            XCTAssertEqual(error.voiceName, "Ann")
            XCTAssertTrue(SessionModel.describe(error).contains("Set up Drama 3 in Settings"))
        }
        let clip = try await library.clip(for: recorded.id)
        XCTAssertEqual(clip?.count, 3, "a voice with its recording needs no codec")

        codecAllowed = true
        do {
            _ = try await library.clip(for: old.id)
            XCTFail("there is no codec in this folder")
        } catch {
            XCTAssertFalse(error is ClipNeedsDrama3, "once allowed, the codec itself is asked: \(error)")
        }
    }

    func testUsageChoiceIsSeparateFromTermsAndEitherAnswerContinues() {
        for enabled in [false, true] {
            let mac = FakeMac()
            mac.usageChoiceMade = false
            mac.facts = SetupFacts(weightsReady: true, codecReady: true, tuned: true)
            let setup = SetupCoordinator(services: mac.services)
            setup.acceptTerms()
            XCTAssertEqual(setup.stage, .usage)
            XCTAssertFalse(mac.usageEnabled)
            XCTAssertFalse(mac.calls.contains("attach"))
            setup.chooseUsage(enabled: enabled)
            XCTAssertEqual(setup.stage, .ready)
            XCTAssertEqual(mac.usageEnabled, enabled)
            setup.reconcile()
            XCTAssertEqual(setup.stage, .ready)
        }
    }

    func testExistingInstallIsAskedForUsageConsent() {
        let mac = FakeMac()
        mac.usageChoiceMade = false
        mac.facts = SetupFacts(termsAccepted: true, weightsReady: true, codecReady: true, tuned: true)
        XCTAssertEqual(SetupCoordinator(services: mac.services).stage, .usage)
    }

    func testRateEstimateNeedsAFewSecondsThenDividesRemainingByRate() {
        var rate = RateEstimate()
        rate.add(bytes: 0, at: 0)
        rate.add(bytes: 100_000_000, at: 2)
        XCTAssertNil(rate.secondsLeft(remaining: 1_000_000_000), "two seconds is too early")
        var slow = RateEstimate()
        slow.add(bytes: 0, at: 0)
        slow.add(bytes: 1_000_000, at: 10)
        XCTAssertNil(slow.secondsLeft(remaining: 1_000_000_000), "a megabyte of small files says nothing about the shards")
        var steady = RateEstimate()
        steady.add(bytes: 0, at: 0)
        steady.add(bytes: 500_000_000, at: 10)
        XCTAssertEqual(steady.secondsLeft(remaining: 1_000_000_000) ?? 0, 20, accuracy: 0.001)
    }
}

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
    var facts = SetupFacts()
    var blocker: String?
    var conversionCommand: String?
    var weightFailures: [Error] = []
    var holdWeights = false
    var tuningWritesSchedule = true
    private(set) var calls: [String] = []

    var services: SetupServices {
        SetupServices(
            facts: { [unowned self] in facts },
            acknowledgeRequirements: { [unowned self] in calls.append("acknowledge"); facts.requirementsAcknowledged = true },
            acceptTerms: { [unowned self] in calls.append("accept"); facts.termsAccepted = true },
            weightsBlocker: { [unowned self] in blocker },
            weightsProgress: { DownloadProgress(totalBytes: 100, doneBytes: 10, filesDone: 1, fileCount: 3) },
            downloadWeights: { [unowned self] report in
                try await self.runWeights(report)
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
            tune: { [unowned self] detail in
                calls.append("tune")
                detail("Tuning kernels")
                if tuningWritesSchedule { facts.tuned = true }
            },
            attachEngine: { [unowned self] in calls.append("attach") },
            prepareFirstLine: { [unowned self] in calls.append("first line") }
        )
    }

    func record(_ call: String) { calls.append(call) }

    private func runWeights(_ report: @escaping @Sendable (DownloadProgress) -> Void) async throws {
        calls.append("weights")
        if !weightFailures.isEmpty { throw weightFailures.removeFirst() }
        report(DownloadProgress(totalBytes: 100, doneBytes: 60, file: "model-00001-of-00008.safetensors", fileBytes: 50, fileSize: 90, filesDone: 1, fileCount: 3))
        while holdWeights { try await Task.sleep(for: .milliseconds(5)) }
        facts.weightsReady = true
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
        mac.facts.requirementGaps = ["This Mac has 16 GB of memory."]
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

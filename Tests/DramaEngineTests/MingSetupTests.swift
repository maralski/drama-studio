// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
@testable import DramaEngine
import XCTest

final class MingSetupTests: XCTestCase {
    var root: URL!
    private var saved: [String: String?] = [:]

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory.appending(path: "MingSetupTests-\(UUID().uuidString)", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        for name in ["MING_WEIGHTS", "HF_HUB_CACHE", "HF_HOME"] { saved[name] = ProcessInfo.processInfo.environment[name] }
    }

    override func tearDownWithError() throws {
        for (name, value) in saved {
            if let value { setenv(name, value, 1) } else { unsetenv(name) }
        }
        try? FileManager.default.removeItem(at: root)
    }

    private func checkpoint(at dir: URL, files: [String]) throws -> URL {
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        for name in files { try Data(count: 1000).write(to: dir.appending(path: name)) }
        return dir
    }

    func testEnginesAreNamedByIdModelIdOrNameInAnyCase() {
        XCTAssertEqual(SpeechEngineID.named("ming"), .ming)
        XCTAssertEqual(SpeechEngineID.named("inclusionai/ming-3.0-tts-0.5b"), .ming)
        XCTAssertEqual(SpeechEngineID.named("Drama 3"), .drama3)
        XCTAssertEqual(SpeechEngineID.named("fishaudio/drama-3"), .drama3)
        XCTAssertNil(SpeechEngineID.named("tts-1"))
    }

    func testTheMingPlanIsTheVerifiedRevisionAndHoldsEveryFileTheEngineLoads() {
        let plan = SpeechEngineID.ming.install.plan
        XCTAssertEqual(plan.repo, "inclusionAI/Ming-3.0-tts-0.5B")
        XCTAssertEqual(plan.revision, "7247f4c94a5b2e987404ed28a5cfec41b1a4ca61")
        XCTAssertTrue(Set(MingLocation.requiredFiles).isSubset(of: plan.files.map(\.path)))
        XCTAssertEqual(plan.files.first { $0.path == "model.safetensors" }?.size, 2_959_055_504)
    }

    func testMingIsFoundInTheHubCacheThatHFHubCacheNames() throws {
        unsetenv("MING_WEIGHTS")
        let hub = root.appending(path: "hub", directoryHint: .isDirectory)
        let snapshot = try checkpoint(at: hub.appending(path: "models--inclusionAI--Ming-3.0-tts-0.5B/snapshots/7247f4c"), files: MingLocation.requiredFiles)
        setenv("HF_HUB_CACHE", hub.path, 1)
        XCTAssertEqual(MingLocation.find()?.directory.standardizedFileURL, snapshot.standardizedFileURL)
        XCTAssertEqual(MingLocation.find()?.isComplete, true)
    }

    func testTheNewestCompleteSnapshotWinsOverNewerPartialOnesAndStrayFiles() throws {
        unsetenv("MING_WEIGHTS")
        let hub = root.appending(path: "hub", directoryHint: .isDirectory)
        let snapshots = hub.appending(path: "models--inclusionAI--Ming-3.0-tts-0.5B/snapshots")
        let complete = try checkpoint(at: snapshots.appending(path: "older"), files: MingLocation.requiredFiles)
        let partial = try checkpoint(at: snapshots.appending(path: "newer"), files: ["config.json"])
        try Data().write(to: snapshots.appending(path: ".DS_Store"))
        let now = Date()
        for (url, age) in [(complete, 300.0), (partial, 200), (snapshots.appending(path: ".DS_Store"), 100)] {
            try FileManager.default.setAttributes([.modificationDate: now.addingTimeInterval(-age)], ofItemAtPath: url.path)
        }
        setenv("HF_HUB_CACHE", hub.path, 1)
        XCTAssertEqual(MingLocation.find()?.directory.lastPathComponent, "older")

        try FileManager.default.removeItem(at: complete.appending(path: "campplus.onnx"))
        try FileManager.default.setAttributes([.modificationDate: now.addingTimeInterval(-300)], ofItemAtPath: complete.path)
        XCTAssertEqual(MingLocation.find()?.directory.lastPathComponent, "newer", "with none complete, the newest folder, so setup names its gaps")

        let drama = hub.appending(path: "models--fishaudio--drama-3/snapshots")
        let whole = try checkpoint(at: drama.appending(path: "older"), files: ["config.json", "tokenizer.json", "model.safetensors.index.json", "codec.pth"])
        let started = try checkpoint(at: drama.appending(path: "newer"), files: ["config.json"])
        try FileManager.default.setAttributes([.modificationDate: now.addingTimeInterval(-300)], ofItemAtPath: whole.path)
        try FileManager.default.setAttributes([.modificationDate: now.addingTimeInterval(-200)], ofItemAtPath: started.path)
        XCTAssertEqual(ModelLocation.huggingFaceCache()?.directory.lastPathComponent, "older")
    }

    func testMingWeightsWinsOverTheCacheEvenWhenIncomplete() throws {
        let hub = root.appending(path: "hub", directoryHint: .isDirectory)
        _ = try checkpoint(at: hub.appending(path: "models--inclusionAI--Ming-3.0-tts-0.5B/snapshots/7247f4c"), files: MingLocation.requiredFiles)
        setenv("HF_HUB_CACHE", hub.path, 1)
        let named = try checkpoint(at: root.appending(path: "named"), files: ["config.json"])
        setenv("MING_WEIGHTS", named.path, 1)
        let report = DramaSetup.check(ming: MingLocation.find())
        XCTAssertEqual(report.directory?.standardizedFileURL, named.standardizedFileURL)
        XCTAssertTrue(report[.weights].summary.contains("model.safetensors"), report[.weights].summary)
        XCTAssertFalse(report.isReady)
        let fix = try XCTUnwrap(report[.weights].fix)
        XCTAssertFalse(fix.contains("hf download"), "downloading into the cache doesn't fill the folder MING_WEIGHTS names: \(fix)")
        XCTAssertTrue(fix.contains("model.safetensors") && fix.contains(named.path), fix)
        XCTAssertTrue(report[.speakerEncoder].fix?.contains("campplus.onnx") == true, report[.speakerEncoder].fix ?? "")
    }

    func testEachSourceOfAnIncompleteCheckpointGetsItsOwnFix() throws {
        unsetenv("MING_WEIGHTS")
        let hub = root.appending(path: "hub", directoryHint: .isDirectory)
        _ = try checkpoint(at: hub.appending(path: "models--inclusionAI--Ming-3.0-tts-0.5B/snapshots/7247f4c"), files: ["config.json"])
        setenv("HF_HUB_CACHE", hub.path, 1)
        let cached = DramaSetup.check(ming: MingLocation.find())
        XCTAssertTrue(cached[.weights].fix?.contains("hf download inclusionAI/Ming-3.0-tts-0.5B") == true, cached[.weights].fix ?? "")

        let downloading = try checkpoint(at: root.appending(path: "Models/Ming-3.0-tts-0.5B-7247f4c"), files: ["config.json"])
        let app = DramaSetup.check(ming: MingLocation(directory: downloading, source: .appDownload))
        for component in [SetupItem.Component.weights, .tokenizer, .speakerEncoder] {
            XCTAssertEqual(app[component].fix, "Resume the Ming download in Settings.")
        }
    }

    func testHFHomeHoldsTheHubWhenHFHubCacheIsUnset() throws {
        unsetenv("HF_HUB_CACHE")
        setenv("HF_HOME", root.path, 1)
        XCTAssertEqual(HuggingFaceCache.directory.standardizedFileURL, root.appending(path: "hub").standardizedFileURL)
    }

    func testNoCheckpointReportsTheMingDownloadCommandForEveryFile() {
        let report = DramaSetup.check(ming: nil)
        XCTAssertEqual(report.engine, .ming)
        XCTAssertEqual(report.items.map(\.component), SpeechEngineID.ming.install.components)
        for component in [SetupItem.Component.weights, .tokenizer, .speakerEncoder] {
            XCTAssertTrue(report[component].fix?.contains("hf download inclusionAI/Ming-3.0-tts-0.5B") == true, "\(report[component])")
        }
    }

    func testAMissingSpeakerEncoderIsTheOnlyGapNamed() throws {
        let dir = try checkpoint(at: root.appending(path: "ming"), files: ["config.json", "model.safetensors", "tokenizer.json"])
        let report = DramaSetup.check(ming: MingLocation(directory: dir))
        XCTAssertEqual(report[.weights].state, .ready)
        XCTAssertEqual(report[.tokenizer].state, .ready)
        XCTAssertTrue(report[.speakerEncoder].isMissing)
        XCTAssertTrue(report[.speakerEncoder].summary.contains("campplus.onnx"))
        XCTAssertEqual(report.items.filter { $0.isMissing && $0.component != .wordRecognizer }.map(\.component), [.speakerEncoder])
    }
}

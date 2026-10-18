import DramaEngine
import XCTest

final class DramaSetupTests: XCTestCase {
    var root: URL!

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory.appending(path: "DramaSetupTests-\(UUID().uuidString)", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: root)
    }

    func snapshot(shards: [String], present: [String]) throws -> ModelLocation {
        let dir = root.appending(path: "snapshot", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let map = Dictionary(uniqueKeysWithValues: shards.enumerated().map { ("tensor.\($0.offset)", $0.element) })
        try JSONSerialization.data(withJSONObject: ["weight_map": map]).write(to: dir.appending(path: "model.safetensors.index.json"))
        for name in present { try Data(count: 1000).write(to: dir.appending(path: name)) }
        return ModelLocation(directory: dir)
    }

    var codecDirectory: URL { root.appending(path: "Codec", directoryHint: .isDirectory) }
    var tuningDirectory: URL { root.appending(path: "Tuning", directoryHint: .isDirectory) }

    func testNoSnapshotReportsTheDownloadCommand() {
        let report = DramaSetup.check(location: nil, codecDirectory: codecDirectory, tuningDirectory: tuningDirectory)
        XCTAssertEqual(report.items.map(\.component), SetupItem.Component.allCases)
        XCTAssertFalse(report.isReady)
        XCTAssertTrue(report[.weights].fix?.contains(DramaSetup.downloadCommand) == true)
    }

    func testMissingShardIsNamed() throws {
        let location = try snapshot(shards: ["a.safetensors", "b.safetensors"],
                                    present: ["config.json", "tokenizer.json", "codec.pth", "a.safetensors"])
        let report = DramaSetup.check(location: location, codecDirectory: codecDirectory, tuningDirectory: tuningDirectory)
        XCTAssertTrue(report[.weights].isMissing)
        XCTAssertTrue(report[.weights].summary.contains("b.safetensors"), report[.weights].summary)
        XCTAssertFalse(report[.weights].summary.contains("a.safetensors"), report[.weights].summary)
        XCTAssertEqual(report[.tokenizer].state, .ready)
    }

    func testCompleteSnapshotWithoutCodecModelsNeedsOnlyTheConversion() throws {
        let location = try snapshot(shards: ["a.safetensors"], present: ["config.json", "tokenizer.json", "codec.pth", "a.safetensors"])
        let report = DramaSetup.check(location: location, codecDirectory: codecDirectory, tuningDirectory: tuningDirectory)
        XCTAssertEqual(report[.weights].state, .ready)
        XCTAssertEqual(report[.codecCheckpoint].state, .ready)
        XCTAssertTrue(report[.codecModels].isMissing)
        XCTAssertTrue(report[.codecModels].fix?.contains("convert_codec.py") == true, report[.codecModels].fix ?? "")
        guard case .builtOnFirstLoad = report[.tuning].state else { return XCTFail("tuning: \(report[.tuning])") }
        XCTAssertEqual(report.items.filter(\.isMissing).map(\.component), [.codecModels])
    }

    func testCodecCheckpointIsNotNeededOnceTheModelsExist() throws {
        let location = try snapshot(shards: ["a.safetensors"], present: ["config.json", "tokenizer.json", "a.safetensors"])
        for name in ["DramaCodecEncoder.mlmodelc", "DramaCodecDecoder.mlmodelc"] {
            try FileManager.default.createDirectory(at: codecDirectory.appending(path: name), withIntermediateDirectories: true)
        }
        let report = DramaSetup.check(location: location, codecDirectory: codecDirectory, tuningDirectory: tuningDirectory)
        XCTAssertEqual(report[.codecCheckpoint].state, .ready)
        XCTAssertEqual(report[.codecModels].state, .ready)
        XCTAssertTrue(report.isReady)
        XCTAssertFalse(report.isPrepared)
    }

    func testPrepareWithoutPythonThrowsTheCommands() async throws {
        let location = try snapshot(shards: ["a.safetensors"], present: ["config.json", "tokenizer.json", "codec.pth", "a.safetensors"])
        do {
            try await DramaSetup.prepare(location: location, codecDirectory: codecDirectory, tuningDirectory: tuningDirectory)
            XCTFail("prepare succeeded without codec models or Python")
        } catch DramaSetupError.pythonRequired(let command) {
            XCTAssertTrue(command.contains("uv venv"), command)
            XCTAssertTrue(command.contains("--codec"), command)
            XCTAssertTrue(command.contains(location.directory.appending(path: "codec.pth").path), command)
        }
    }

    func testPrepareReportsAFailedConversion() async throws {
        let location = try snapshot(shards: ["a.safetensors"], present: ["config.json", "tokenizer.json", "codec.pth", "a.safetensors"])
        do {
            try await DramaSetup.prepare(location: location, codecDirectory: codecDirectory, python: URL(filePath: "/usr/bin/false"),
                                         tuningDirectory: tuningDirectory)
            XCTFail("prepare succeeded with a Python that fails")
        } catch DramaSetupError.conversionFailed(let command, let status, _) {
            XCTAssertEqual(status, 1)
            XCTAssertTrue(command.hasPrefix("/usr/bin/false -I -B "), command)
        }
    }

    func testPrepareStopsOnMissingWeights() async throws {
        let location = try snapshot(shards: ["a.safetensors"], present: ["tokenizer.json", "codec.pth"])
        do {
            try await DramaSetup.prepare(location: location, codecDirectory: codecDirectory, python: URL(filePath: "/usr/bin/false"),
                                         tuningDirectory: tuningDirectory)
            XCTFail("prepare succeeded without weights")
        } catch DramaSetupError.notReady(let report) {
            XCTAssertTrue(report[.weights].isMissing)
        }
    }

    func testBundledConversionScriptShips() throws {
        let script = try XCTUnwrap(DramaSetup.conversionScript)
        XCTAssertTrue(FileManager.default.fileExists(atPath: script.path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: script.deletingLastPathComponent().appending(path: "dac_reference/__init__.py").path))
    }

    func testRealInstallFindsItsFiles() throws {
        guard let location = ModelLocation.huggingFaceCache() else { throw XCTSkip("drama-3 not downloaded") }
        let report = DramaSetup.check(location: location)
        for item in report.items { print("\(item.component): \(item.summary)") }
        XCTAssertEqual(report[.weights].state, .ready)
        XCTAssertEqual(report[.tokenizer].state, .ready)
    }
}

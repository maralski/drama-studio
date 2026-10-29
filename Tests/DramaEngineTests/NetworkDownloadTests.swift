// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
@testable import DramaEngine
import XCTest

/// Real downloads, skipped unless asked for:
///
///     DRAMA_NETWORK_TESTS=1 swift test --filter NetworkDownloadTests
///
/// The shard test keeps its partial file in `DRAMA_NETWORK_TEST_DIR` when set,
/// so its bytes can be compared with `curl -r`. The codec test needs
/// `DRAMA_CODEC_ZIP_URL` and `DRAMA_CODEC_ZIP_SHA256`.
final class NetworkDownloadTests: XCTestCase {
    let environment = ProcessInfo.processInfo.environment
    var root: URL!

    override func setUpWithError() throws {
        try XCTSkipUnless(environment["DRAMA_NETWORK_TESTS"] == "1", "set DRAMA_NETWORK_TESTS=1 to download from Hugging Face")
        root = environment["DRAMA_NETWORK_TEST_DIR"].map { URL(filePath: $0, directoryHint: .isDirectory) }
            ?? FileManager.default.temporaryDirectory.appending(path: "NetworkDownloadTests-\(UUID().uuidString)", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    func testThePinnedSmallFilesDownloadAndVerify() async throws {
        let full = ModelDownloadPlan.drama3
        let small = ModelDownloadPlan(repo: full.repo, revision: full.revision, files: full.files.filter { $0.size < 100_000_000 })
        let directory = root.appending(path: "small", directoryHint: .isDirectory)
        try? FileManager.default.removeItem(at: directory)
        let start = Date()
        try await PlanDownload(plan: small, directory: directory, downloader: FileDownloader()).run { _ in }
        XCTAssertTrue(small.missingFiles(in: directory).isEmpty)
        print("small files: \(small.files.count) files, \(small.totalBytes) bytes verified in \(String(format: "%.1f", Date().timeIntervalSince(start))) s into \(directory.path)")
    }

    func testAShardResumesAfterACancelAndKeepsItsBytes() async throws {
        let plan = ModelDownloadPlan.drama3
        let shard = try XCTUnwrap(plan.files.first { $0.path == "model-00001-of-00008.safetensors" })
        let destination = root.appending(path: "shard/\(shard.path)")
        let partial = FileDownloader.partialURL(for: destination)
        try? FileManager.default.removeItem(at: partial)
        let target = DownloadTarget(url: plan.url(for: shard), destination: destination, size: shard.size, sha256: shard.sha256)
        let downloader = FileDownloader()

        let first = try await fetch(target, with: downloader, until: 300_000_000)
        let afterCancel = Int64(DramaSetup.fileSize(partial))
        XCTAssertGreaterThanOrEqual(afterCancel, 300_000_000)
        XCTAssertEqual(first.min(), first.first)

        let second = try await fetch(target, with: downloader, until: afterCancel + 100_000_000)
        let afterResume = Int64(DramaSetup.fileSize(partial))
        XCTAssertGreaterThanOrEqual(second.min() ?? 0, afterCancel, "the resumed transfer never restarted from zero, so the server honored the Range")
        XCTAssertGreaterThan(afterResume, afterCancel)
        XCTAssertFalse(FileManager.default.fileExists(atPath: destination.path), "an unfinished shard is never renamed into place")
        print("shard resume: cancelled at \(afterCancel) bytes, resumed and cancelled again at \(afterResume) bytes; partial \(partial.path)")
    }

    /// Downloads until the partial file reaches `bytes`, then cancels; returns every progress value seen.
    private func fetch(_ target: DownloadTarget, with downloader: FileDownloader, until bytes: Int64) async throws -> [Int64] {
        let seen = Recorder<Int64>()
        let partial = FileDownloader.partialURL(for: target.destination)
        let task = Task { try await downloader.download(target) { done, _ in Task { await seen.add(done) } } }
        while Int64(DramaSetup.fileSize(partial)) < bytes { try await Task.sleep(for: .milliseconds(50)) }
        task.cancel()
        do {
            try await task.value
            XCTFail("expected the cancel to stop the download")
        } catch is CancellationError {}
        try await Task.sleep(for: .milliseconds(200))
        return await seen.values
    }

    func testTheCodecZipDownloadsInstallsAndLoads() async throws {
        let url = try XCTUnwrap(environment["DRAMA_CODEC_ZIP_URL"].flatMap(URL.init(string:)), "set DRAMA_CODEC_ZIP_URL")
        let sha = try XCTUnwrap(environment["DRAMA_CODEC_ZIP_SHA256"], "set DRAMA_CODEC_ZIP_SHA256")
        let bundle = CodecBundle(url: url, sha256: sha)
        let downloads = root.appending(path: "codec-downloads", directoryHint: .isDirectory)
        let codec = root.appending(path: "codec-home/Codec", directoryHint: .isDirectory)
        try? FileManager.default.removeItem(at: downloads)
        try? FileManager.default.removeItem(at: codec)
        let archive = try await bundle.download(into: downloads, with: FileDownloader()) { _, _ in }
        try CodecBundle.install(archive: archive, into: codec)
        XCTAssertTrue(DramaCodec.modelsExist(in: codec))
        let loaded = try await DramaCodec(directory: codec)
        let frames = await loaded.decoderFrames
        XCTAssertEqual(frames, [DramaCodec.probeFrames, DramaCodec.windowFrames])
        print("codec zip: installed and loaded from \(codec.path), decoders take \(frames) frames")
    }
}

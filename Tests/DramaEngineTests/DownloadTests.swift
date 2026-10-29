// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import CryptoKit
@testable import DramaEngine
import XCTest

/// Serves `StubServer.files` over a fake HTTP stack, honoring Range requests
/// unless told not to, and records every request it sees.
final class StubServer: URLProtocol, @unchecked Sendable {
    struct Behavior {
        var body: Data
        var honorsRange = true
        var redirect: URL?
        /// Answer this status instead of the body this many times first.
        var failures: [Int] = []
        /// Pause between 4 KB chunks, so a test can cancel mid-file.
        var chunkDelay: TimeInterval = 0
    }

    nonisolated(unsafe) static var files: [URL: Behavior] = [:]
    nonisolated(unsafe) static var requests: [URLRequest] = []
    static let lock = NSLock()
    private var stopped = false

    static func reset() { lock.withLock { files = [:]; requests = [] } }
    static var seen: [URLRequest] { lock.withLock { requests } }
    static func configuration() -> URLSessionConfiguration {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [StubServer.self]
        return configuration
    }

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        let url = request.url!
        let behavior: Behavior? = Self.lock.withLock {
            Self.requests.append(request)
            guard var found = Self.files[url] else { return nil }
            if !found.failures.isEmpty {
                let status = found.failures.removeFirst()
                Self.files[url] = found
                found.body = Data()
                found.failures = [status]
            }
            return found
        }
        guard let behavior else { return answer(404, [:], Data()) }
        if let status = behavior.failures.first { return answer(status, [:], Data()) }
        if let redirect = behavior.redirect {
            let response = HTTPURLResponse(url: url, statusCode: 302, httpVersion: "HTTP/1.1", headerFields: ["Location": redirect.absoluteString])!
            client?.urlProtocol(self, wasRedirectedTo: URLRequest(url: redirect), redirectResponse: response)
            return
        }
        let body = behavior.body
        var start = 0
        if behavior.honorsRange, let range = request.value(forHTTPHeaderField: "Range"),
           let from = Int(range.dropFirst("bytes=".count).dropLast()) {
            start = from
        }
        let headers = start > 0
            ? ["Content-Range": "bytes \(start)-\(body.count - 1)/\(body.count)", "Content-Length": "\(body.count - start)"]
            : ["Content-Length": "\(body.count)"]
        let response = HTTPURLResponse(url: url, statusCode: start > 0 ? 206 : 200, httpVersion: "HTTP/1.1", headerFields: headers)!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        let rest = body.subdata(in: start..<body.count)
        guard behavior.chunkDelay > 0 else {
            client?.urlProtocol(self, didLoad: rest)
            client?.urlProtocolDidFinishLoading(self)
            return
        }
        DispatchQueue.global().async { [self] in
            var offset = 0
            while offset < rest.count {
                if Self.lock.withLock({ stopped }) { return }
                let end = min(rest.count, offset + 4096)
                client?.urlProtocol(self, didLoad: rest.subdata(in: offset..<end))
                offset = end
                Thread.sleep(forTimeInterval: behavior.chunkDelay)
            }
            client?.urlProtocolDidFinishLoading(self)
        }
    }

    override func stopLoading() { Self.lock.withLock { stopped = true } }

    private func answer(_ status: Int, _ headers: [String: String], _ body: Data) {
        client?.urlProtocol(self, didReceive: HTTPURLResponse(url: request.url!, statusCode: status, httpVersion: "HTTP/1.1", headerFields: headers)!,
                            cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: body)
        client?.urlProtocolDidFinishLoading(self)
    }
}

final class DownloadTests: XCTestCase {
    var root: URL!
    let downloader = FileDownloader(configuration: StubServer.configuration)

    override func setUpWithError() throws {
        StubServer.reset()
        root = FileManager.default.temporaryDirectory.appending(path: "DownloadTests-\(UUID().uuidString)", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: root)
    }

    static func payload(_ count: Int, seed: UInt8 = 7) -> Data {
        Data((0..<count).map { UInt8(truncatingIfNeeded: $0 &* 31 &+ Int(seed)) })
    }

    static func sha(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }

    func serve(_ data: Data, at path: String = "file.bin", _ configure: (inout StubServer.Behavior) -> Void = { _ in }) -> DownloadTarget {
        let url = URL(string: "https://stub.test/\(path)")!
        var behavior = StubServer.Behavior(body: data)
        configure(&behavior)
        StubServer.lock.withLock { StubServer.files[url] = behavior }
        return DownloadTarget(url: url, destination: root.appending(path: path), size: Int64(data.count), sha256: Self.sha(data))
    }

    func testPinnedPlanNamesRevisionE0a9218AndItsTotals() {
        let plan = ModelDownloadPlan.drama3
        XCTAssertEqual(plan.files.count, 17)
        XCTAssertEqual(plan.totalBytes, 41_502_592_187)
        XCTAssertEqual(plan.files.filter { $0.path.hasSuffix(".safetensors") }.count, 8)
        XCTAssertEqual(plan.folderName, "drama-3-e0a9218")
        for name in ["config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                     "special_tokens_map.json", "LICENSE", "README.md", "codec.pth"] {
            XCTAssertTrue(plan.files.contains { $0.path == name }, name)
        }
        for file in plan.files {
            XCTAssertEqual(plan.url(for: file).absoluteString,
                           "https://huggingface.co/fishaudio/drama-3/resolve/e0a9218aa48888270f06ef1ed9ea67622c0e17d2/\(file.path)")
            XCTAssertEqual(file.sha256.count, 64)
            XCTAssertTrue(file.sha256.allSatisfy(\.isHexDigit))
        }
        XCTAssertEqual(plan.files.first { $0.path == "tokenizer.json" }?.sha256, "0e9b9db76b2f0389b04b6ef057a2ac3b89100b7e96f2dd7034d19580a09654d6")
    }

    func testDownloadVerifiesThenRenamesIntoPlace() async throws {
        let data = Self.payload(50_000)
        let target = serve(data)
        try await downloader.download(target)
        XCTAssertEqual(try Data(contentsOf: target.destination), data)
        XCTAssertFalse(FileManager.default.fileExists(atPath: FileDownloader.partialURL(for: target.destination).path))
    }

    func testResumesAPartialFileWithARangeRequest() async throws {
        let data = Self.payload(50_000)
        let target = serve(data)
        try data.prefix(12_345).write(to: FileDownloader.partialURL(for: target.destination))
        try await downloader.download(target)
        XCTAssertEqual(StubServer.seen.map { $0.value(forHTTPHeaderField: "Range") }, ["bytes=12345-"])
        XCTAssertEqual(try Data(contentsOf: target.destination), data)
    }

    func testAServerThatIgnoresRangeStartsTheFileOver() async throws {
        let data = Self.payload(30_000)
        let target = serve(data) { $0.honorsRange = false }
        try data.prefix(10_000).write(to: FileDownloader.partialURL(for: target.destination))
        try await downloader.download(target)
        XCTAssertEqual(try Data(contentsOf: target.destination), data)
    }

    func testTheRangeSurvivesARedirect() async throws {
        let data = Self.payload(20_000)
        let cdn = serve(data, at: "cdn.bin")
        var target = serve(Data(), at: "resolve.bin") { $0.redirect = cdn.url }
        target.size = cdn.size
        target.sha256 = cdn.sha256
        try data.prefix(5_000).write(to: FileDownloader.partialURL(for: target.destination))
        try await downloader.download(target)
        XCTAssertEqual(StubServer.seen.last?.url, cdn.url)
        XCTAssertEqual(StubServer.seen.last?.value(forHTTPHeaderField: "Range"), "bytes=5000-")
        XCTAssertEqual(try Data(contentsOf: target.destination), data)
    }

    func testAChecksumMismatchIsRejectedAndThePartialDeleted() async throws {
        var target = serve(Self.payload(40_000))
        target.sha256 = String(repeating: "0", count: 64)
        do {
            try await downloader.download(target)
            XCTFail("a wrong hash must not be accepted")
        } catch let error as DownloadError {
            guard case .checksumMismatch = error else { return XCTFail("\(error)") }
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: target.destination.path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: FileDownloader.partialURL(for: target.destination).path))
    }

    func testCancellationLeavesAPartialThatTheNextDownloadResumes() async throws {
        let data = Self.payload(400_000)
        let target = serve(data) { $0.chunkDelay = 0.002 }
        let partial = FileDownloader.partialURL(for: target.destination)
        let task = Task { [downloader] in try await downloader.download(target) }
        while DramaSetup.fileSize(partial) < 40_000 { try await Task.sleep(for: .milliseconds(5)) }
        task.cancel()
        do {
            try await task.value
            XCTFail("the download should have been cancelled")
        } catch is CancellationError {}
        let kept = DramaSetup.fileSize(partial)
        XCTAssertGreaterThan(kept, 0)
        XCTAssertLessThan(kept, UInt64(data.count))
        XCTAssertEqual(try Data(contentsOf: partial), data.prefix(Int(kept)))
        XCTAssertFalse(FileManager.default.fileExists(atPath: target.destination.path))

        StubServer.lock.withLock { StubServer.files[target.url]?.chunkDelay = 0 }
        try await downloader.download(target)
        XCTAssertEqual(StubServer.seen.last?.value(forHTTPHeaderField: "Range"), "bytes=\(kept)-")
        XCTAssertEqual(try Data(contentsOf: target.destination), data)
    }

    func testAnHTTPErrorKeepsTheFileOutOfPlace() async throws {
        let target = serve(Self.payload(1000)) { $0.failures = [503] }
        do {
            try await downloader.download(target)
            XCTFail("503 must fail")
        } catch let error as DownloadError {
            XCTAssertEqual(error, .httpStatus(503, target.url))
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: target.destination.path))
    }

    func plan(_ files: [(String, Data)]) -> ModelDownloadPlan {
        ModelDownloadPlan(repo: "test/model", revision: "abcdef0123456789", files: files.map {
            ModelDownloadPlan.File(path: $0.0, size: Int64($0.1.count), sha256: Self.sha($0.1))
        })
    }

    func stubPlan(_ plan: ModelDownloadPlan, bodies: [String: Data], failures: [String: [Int]] = [:]) {
        for file in plan.files {
            StubServer.lock.withLock {
                StubServer.files[plan.url(for: file)] = StubServer.Behavior(body: bodies[file.path]!, failures: failures[file.path] ?? [])
            }
        }
    }

    func testAPlanRetriesAFailedFileWithBackoffAndSkipsFinishedFiles() async throws {
        let a = Self.payload(3000, seed: 1), b = Self.payload(5000, seed: 2)
        let plan = plan([("a.json", a), ("b.bin", b)])
        stubPlan(plan, bodies: ["a.json": a, "b.bin": b], failures: ["b.bin": [500, 502]])
        let directory = root.appending(path: "model", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try a.write(to: directory.appending(path: "a.json"))

        let waits = Recorder<Duration>()
        var download = PlanDownload(plan: plan, directory: directory, downloader: downloader)
        download.sleep = { await waits.add($0) }
        try await download.run { _ in }

        let waited = await waits.values
        XCTAssertEqual(waited, [.seconds(2), .seconds(8)])
        XCTAssertEqual(try Data(contentsOf: directory.appending(path: "b.bin")), b)
        XCTAssertFalse(StubServer.seen.contains { $0.url?.lastPathComponent == "a.json" }, "a finished file is not fetched again")
        XCTAssertEqual(download.progress, DownloadProgress(totalBytes: 8000, doneBytes: 8000, filesDone: 2, fileCount: 2))
        XCTAssertTrue(plan.missingFiles(in: directory).isEmpty)
    }

    func testAPlanGivesUpAfterThreeRetries() async throws {
        let a = Self.payload(1000)
        let plan = plan([("a.bin", a)])
        stubPlan(plan, bodies: ["a.bin": a], failures: ["a.bin": [500, 500, 500, 500]])
        var download = PlanDownload(plan: plan, directory: root, downloader: downloader)
        download.sleep = { _ in }
        do {
            try await download.run { _ in }
            XCTFail("four failures must fail the plan")
        } catch PlanDownloadError.fileFailed(let path, let attempts, _) {
            XCTAssertEqual(path, "a.bin")
            XCTAssertEqual(attempts, 4)
        }
        XCTAssertEqual(StubServer.seen.count, 4)
    }

    func testCodecBundleReadsBothInfoPlistKeysOrNothing() {
        let sha = String(repeating: "a", count: 64)
        XCTAssertEqual(CodecBundle(info: ["DSCodecBundleURL": "https://github.com/x/y/releases/download/v1/c.zip", "DSCodecBundleSHA256": sha]),
                       CodecBundle(url: URL(string: "https://github.com/x/y/releases/download/v1/c.zip")!, sha256: sha))
        XCTAssertNil(CodecBundle(info: ["DSCodecBundleURL": "https://example.com/c.zip"]))
        XCTAssertNil(CodecBundle(info: ["DSCodecBundleURL": "http://example.com/c.zip", "DSCodecBundleSHA256": sha]))
        XCTAssertNil(CodecBundle(info: ["DSCodecBundleURL": "https://example.com/c.zip", "DSCodecBundleSHA256": "abc"]))
        XCTAssertNil(CodecBundle(info: nil))
    }

    func zip(packages: [String]) throws -> URL {
        let source = root.appending(path: "zip-source", directoryHint: .isDirectory)
        for name in packages {
            let package = source.appending(path: name, directoryHint: .isDirectory)
            try FileManager.default.createDirectory(at: package, withIntermediateDirectories: true)
            try Data("{}".utf8).write(to: package.appending(path: "Manifest.json"))
        }
        try Data("license".utf8).write(to: source.appending(path: "LICENSE-FISH-AUDIO"))
        let archive = root.appending(path: "codec.zip")
        let ditto = try Process.run(URL(filePath: "/usr/bin/ditto"), arguments: ["-c", "-k", source.path, archive.path])
        ditto.waitUntilExit()
        XCTAssertEqual(ditto.terminationStatus, 0)
        return archive
    }

    func testCodecInstallMovesTheThreePackagesInAndDropsStaleCompiledModels() throws {
        let archive = try zip(packages: CodecBundle.packageNames)
        let codec = root.appending(path: "Support/Codec", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: codec.appending(path: "DramaCodecDecoder.mlmodelc"), withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: codec.deletingLastPathComponent().appending(path: ".codec-staging-old"), withIntermediateDirectories: true)
        try CodecBundle.install(archive: archive, into: codec)
        XCTAssertTrue(DramaCodec.modelsExist(in: codec))
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: codec.path).sorted(), CodecBundle.packageNames.sorted())
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: codec.deletingLastPathComponent().path), ["Codec"])
    }

    func testCodecInstallRefusesAZipWithoutAllThreeModels() throws {
        let archive = try zip(packages: Array(CodecBundle.packageNames.prefix(2)))
        let codec = root.appending(path: "Support/Codec", directoryHint: .isDirectory)
        XCTAssertThrowsError(try CodecBundle.install(archive: archive, into: codec)) { error in
            XCTAssertEqual(error as? CodecBundleError, .missingModels(["DramaCodecProbe.mlpackage"]))
        }
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: codec.path), [])
    }
}

actor Recorder<Value: Sendable> {
    private(set) var values: [Value] = []
    func add(_ value: Value) { values.append(value) }
}

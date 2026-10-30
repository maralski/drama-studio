// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import CryptoKit
import XCTest
@testable import DramaStudio

final class AppUpdateTests: XCTestCase {
    private func release(_ version: String = "1.10.0", prerelease: Bool = false, host: String = "github.com") -> AppRelease {
        AppRelease(tag_name: "v\(version)", draft: false, prerelease: prerelease, assets: [
            .init(name: "DramaStudio-\(version).dmg", browser_download_url: URL(string: "https://\(host)/maralski/drama-studio/releases/download/v\(version)/DramaStudio-\(version).dmg")!),
            .init(name: "SHA256SUMS", browser_download_url: URL(string: "https://\(host)/maralski/drama-studio/releases/download/v\(version)/SHA256SUMS")!)
        ])
    }

    func testSelectsOnlyNewerStableRelease() throws {
        XCTAssertEqual(try release().update(after: "1.9.0")?.version, "1.10.0")
        XCTAssertNil(try release().update(after: "1.10.0"))
        XCTAssertNil(try release().update(after: "2.0.0"))
        XCTAssertNil(try release(prerelease: true).update(after: "1.0.0"))
        XCTAssertNil(try release("2.0.0-beta.1").update(after: "1.0.0"))
        XCTAssertThrowsError(try release(host: "example.com").update(after: "1.0.0"))
        XCTAssertThrowsError(try release().update(after: "development"))
        XCTAssertNil(AppVersion("1..0"))
        XCTAssertNil(AppVersion("1.0.-1"))
    }

    func testRequiresExactAppChecksumAndRejectsDuplicates() throws {
        let hash = String(repeating: "a", count: 64)
        let line = "\(hash)  DramaStudio-1.10.0.dmg\n"
        XCTAssertEqual(try AppRelease.checksum(in: line + "bad  DramaStudio-Codec-1.10.0.zip\n", version: "1.10.0"), hash)
        XCTAssertThrowsError(try AppRelease.checksum(in: line + line, version: "1.10.0"))
        XCTAssertThrowsError(try AppRelease.checksum(in: line, version: "1.0.0"))
        XCTAssertThrowsError(try AppRelease.checksum(in: "bad  DramaStudio-1.10.0.dmg", version: "1.10.0"))
    }

    @MainActor
    func testAutomaticUpdatesDefaultOnAndOptOutPersists() async throws {
        let name = "AppUpdateTests.\(UUID())"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: name))
        defer { defaults.removePersistentDomain(forName: name) }
        let updater = AppUpdater(defaults: defaults)
        XCTAssertTrue(updater.automatic)
        updater.automatic = false
        let reopened = AppUpdater(defaults: defaults)
        XCTAssertFalse(reopened.automatic)
        await reopened.start()
        XCTAssertEqual(reopened.status, "Updates are checked when Drama Studio launches.")
        XCTAssertFalse(reopened.busy)
        XCTAssertNil(reopened.readyVersion)
    }

    func testChecksumMismatchNeverMountsOrStagesApp() throws {
        let root = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let image = root.appending(path: "update.dmg")
        try Data("corrupt download".utf8).write(to: image)
        XCTAssertThrowsError(try UpdateInstaller.prepare(image: image, checksum: String(repeating: "0", count: 64), version: "1.1.0", currentApp: root, directory: root)) { error in
            XCTAssertEqual(error.localizedDescription, "The downloaded update failed checksum verification.")
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appending(path: "mount").path))
    }

    func testInstallerReplacesAppAndRollsBackOnRelaunchFailure() throws {
        for succeeds in [true, false] {
            let root = FileManager.default.temporaryDirectory.appending(path: "Update test ' \(UUID())")
            let work = root.appending(path: "work")
            let target = root.appending(path: "DramaStudio.app")
            let staged = work.appending(path: "DramaStudio.app")
            try FileManager.default.createDirectory(at: staged, withIntermediateDirectories: true)
            try FileManager.default.createDirectory(at: target, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: root) }
            try Data("old".utf8).write(to: target.appending(path: "version"))
            try Data("new".utf8).write(to: staged.appending(path: "version"))
            let helper = root.appending(path: "install.sh")
            // Exercise real filesystem moves without launching an app in the test runner.
            let script = UpdateInstaller.script.replacingOccurrences(of: "/usr/bin/open", with: succeeds ? "/usr/bin/true" : "/usr/bin/false")
            try script.write(to: helper, atomically: true, encoding: .utf8)
            let process = Process()
            process.executableURL = URL(filePath: "/bin/sh")
            process.arguments = [helper.path, "2147483647", target.path, staged.path, work.path]
            try process.run()
            process.waitUntilExit()
            XCTAssertEqual(process.terminationStatus, succeeds ? 0 : 1)
            XCTAssertEqual(try String(contentsOf: target.appending(path: "version"), encoding: .utf8), succeeds ? "new" : "old")
        }
    }
}

private final class UpdateResponseProtocol: URLProtocol {
    static var requests = 0
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {
        Self.requests += 1
        let response = HTTPURLResponse(url: request.url!, statusCode: 200, httpVersion: nil, headerFields: nil)!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: Data(#"{"tag_name":"v1.0.0","draft":false,"prerelease":false,"assets":[]}"#.utf8))
        client?.urlProtocolDidFinishLoading(self)
    }
    override func stopLoading() { }
}

extension AppUpdateTests {
    @MainActor
    func testLaunchChecksOnceAndManualCheckWorksWithAutomaticUpdatesOff() async throws {
        let root = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString + ".app")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let name = "AppUpdateTests.\(UUID())"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: name))
        defer { defaults.removePersistentDomain(forName: name) }
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [UpdateResponseProtocol.self]
        let session = URLSession(configuration: configuration)
        defer { session.invalidateAndCancel() }
        UpdateResponseProtocol.requests = 0
        let updater = AppUpdater(defaults: defaults, app: root, version: "1.0.0", session: session)
        await updater.start()
        await updater.start()
        XCTAssertEqual(UpdateResponseProtocol.requests, 1)
        XCTAssertEqual(updater.status, "Drama Studio 1.0.0 is up to date.")
        updater.automatic = false
        await updater.check()
        XCTAssertEqual(UpdateResponseProtocol.requests, 2)
        XCTAssertNil(updater.readyVersion)
    }

    func testPreparesVerifiedAppFromRealDiskImage() throws {
        let root = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        let source = root.appending(path: "source")
        let app = source.appending(path: "DramaStudio.app")
        let contents = app.appending(path: "Contents")
        let work = root.appending(path: "work")
        try FileManager.default.createDirectory(at: contents.appending(path: "MacOS"), withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: work, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let plist: [String: Any] = ["CFBundleIdentifier": "io.github.maralski.DramaStudio", "CFBundleShortVersionString": "1.2.0", "CFBundleExecutable": "DramaStudio", "CFBundlePackageType": "APPL"]
        try PropertyListSerialization.data(fromPropertyList: plist, format: .xml, options: 0).write(to: contents.appending(path: "Info.plist"))
        try FileManager.default.copyItem(at: URL(filePath: "/usr/bin/true"), to: contents.appending(path: "MacOS/DramaStudio"))
        try UpdateInstaller.run("/usr/bin/codesign", ["--force", "--sign", "-", app.path])
        let image = root.appending(path: "update.dmg")
        try UpdateInstaller.run("/usr/bin/hdiutil", ["create", "-quiet", "-format", "UDZO", "-srcfolder", source.path, image.path])
        let hash = SHA256.hash(data: try Data(contentsOf: image)).map { String(format: "%02x", $0) }.joined()
        let prepared = try UpdateInstaller.prepare(image: image, checksum: hash, version: "1.2.0", currentApp: app, directory: work)
        XCTAssertEqual(Bundle(url: prepared)?.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String, "1.2.0")
        XCTAssertTrue(FileManager.default.isExecutableFile(atPath: prepared.appending(path: "Contents/MacOS/DramaStudio").path))
        XCTAssertTrue(try FileManager.default.contentsOfDirectory(atPath: work.appending(path: "mount").path).isEmpty, "The disk image is detached after preparation")
    }
}

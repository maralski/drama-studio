// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import AppKit
import Foundation
import Observation

@MainActor @Observable
final class AppUpdater {
    static let automaticKey = "automaticallyUpdatesApp"
    private let defaults: UserDefaults
    private let app: URL
    private let currentVersion: String
    private let session: URLSession
    private var started = false
    private var installing = false
    private var pendingInstall = false
    @ObservationIgnored var requestTermination: () -> Void = { NSApp.terminate(nil) }
    @ObservationIgnored var launchInstaller: (URL, URL, URL) throws -> Void = {
        try UpdateInstaller.launch(staged: $0, target: $1, directory: $2)
    }
    private var directory: URL?
    private var staged: URL?
    private(set) var busy = false
    private(set) var status = "Updates are checked when Drama Studio launches."
    private(set) var readyVersion: String?
    var showsReady = false

    var automatic: Bool {
        didSet { defaults.set(automatic, forKey: Self.automaticKey) }
    }

    init(defaults: UserDefaults = AppHome.defaults, app: URL = Bundle.main.bundleURL,
         version: String = Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "",
         session: URLSession = .shared) {
        self.defaults = defaults
        self.app = app
        self.currentVersion = version
        self.session = session
        automatic = defaults.object(forKey: Self.automaticKey) as? Bool ?? true
    }

    func start() async {
        guard !started else { return }
        started = true
        if automatic { await check() }
    }

    func check() async {
        guard !busy, readyVersion == nil else { return }
        guard app.pathExtension == "app", AppVersion(currentVersion) != nil,
              !app.path.hasPrefix("/Volumes/"), !app.path.contains("/AppTranslocation/"),
              FileManager.default.isWritableFile(atPath: app.deletingLastPathComponent().path),
              FileManager.default.isWritableFile(atPath: app.path) else {
            status = "Install Drama Studio in a writable Applications folder to enable updates."
            return
        }
        busy = true
        status = "Checking for updates…"
        defer { busy = false }
        var work: URL?
        do {
            var request = URLRequest(url: URL(string: "https://api.github.com/repos/maralski/drama-studio/releases/latest")!)
            request.setValue("application/vnd.github+json", forHTTPHeaderField: "Accept")
            request.cachePolicy = .reloadIgnoringLocalCacheData
            request.timeoutInterval = 30
            let (data, response) = try await session.data(for: request)
            if (response as? HTTPURLResponse)?.statusCode == 404 {
                status = "No published release is available yet."
                return
            }
            try Self.validate(response)
            let release = try JSONDecoder().decode(AppRelease.self, from: data)
            guard let update = try release.update(after: currentVersion) else {
                status = "Drama Studio \(currentVersion) is up to date."
                return
            }
            status = "Downloading Drama Studio \(update.version) in the background…"
            // A sibling keeps both replacement renames on the same filesystem.
            let folder = app.deletingLastPathComponent().appending(path: ".DramaStudio-update-\(UUID().uuidString)")
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: false, attributes: [.posixPermissions: 0o700])
            work = folder
            let (sums, sumsResponse) = try await session.data(from: update.checksums)
            try Self.validate(sumsResponse)
            let checksum = try AppRelease.checksum(in: String(decoding: sums, as: UTF8.self), version: update.version)
            let (download, downloadResponse) = try await session.download(from: update.image)
            defer { try? FileManager.default.removeItem(at: download) }
            try Self.validate(downloadResponse)
            let image = folder.appending(path: "update.dmg")
            try FileManager.default.moveItem(at: download, to: image)
            status = "Verifying Drama Studio \(update.version)…"
            let currentApp = app
            let candidate = try await Task.detached {
                try UpdateInstaller.prepare(image: image, checksum: checksum, version: update.version, currentApp: currentApp, directory: folder)
            }.value
            offerPreparedUpdate(candidate, directory: folder, version: update.version)
        } catch {
            if let work { try? FileManager.default.removeItem(at: work) }
            status = "Could not update: \(error.localizedDescription)"
        }
    }

    func offerPreparedUpdate(_ candidate: URL, directory: URL, version: String) {
        self.directory = directory
        staged = candidate
        readyVersion = version
        status = "Drama Studio \(version) is ready. Save your work, then install and restart."
        showsReady = true
    }

    func installAndRestart() {
        guard staged != nil, directory != nil, readyVersion != nil, !busy, !pendingInstall else { return }
        pendingInstall = true
        requestTermination()
    }

    func cancelTermination() { pendingInstall = false }

    func completeTermination() -> Bool {
        guard pendingInstall else { return true }
        pendingInstall = false
        guard let staged, let directory else { return false }
        do {
            try launchInstaller(staged, app, directory)
            installing = true
            busy = true
            status = "Restarting to install the update…"
            return true
        } catch {
            status = "Could not install: \(error.localizedDescription)"
            return false
        }
    }

    func discardOnQuit() {
        if !installing, let directory { try? FileManager.default.removeItem(at: directory) }
    }

    private static func validate(_ response: URLResponse) throws {
        guard let http = response as? HTTPURLResponse, http.statusCode == 200,
              http.url?.scheme == "https" else { throw UpdateError.message("The release server could not complete the request. Try again later.") }
    }
}

// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation
import Observation

/// Closed vocabulary: callers cannot attach text, errors, URLs or identifiers.
enum UsageMetric: String, CaseIterable {
    case launch = "app.launch"
    case generationSucceeded = "generation.succeeded"
    case generationFailed = "generation.failed"
    case generationCancelled = "generation.cancelled"
    case setupFailed = "setup.failed"
}

@MainActor @Observable
final class UsageMonitoring {
    static let choiceKey = "usageSharing.v1"
    static let explanation = "Share anonymous usage information to help improve Drama Studio. This includes counts of app launches, completed or cancelled generations, and failures. Your scripts, audio, voices, file names and personal identifiers are never included. Sharing is optional and you can turn it off at any time in Settings."
    static let shared = UsageMonitoring(defaults: AppHome.defaults,
        endpoint: endpoint(from: Bundle.main.infoDictionary?["DSUsageEndpoint"] as? String))

    private(set) var enabled: Bool
    private(set) var choiceMade: Bool
    var available: Bool { endpoint != nil }
    @ObservationIgnored private let defaults: UserDefaults
    @ObservationIgnored private let endpoint: URL?
    @ObservationIgnored private let session: URLSession
    @ObservationIgnored private var counts: [String: Int] = [:]
    @ObservationIgnored private var timer: Task<Void, Never>?
    @ObservationIgnored private var upload: URLSessionDataTask?
    @ObservationIgnored private var uploadID: UUID?

    init(defaults: UserDefaults, endpoint: URL?, session: URLSession? = nil) {
        self.defaults = defaults
        self.endpoint = endpoint
        let choice = defaults.string(forKey: Self.choiceKey)
        enabled = choice == "enabled"
        choiceMade = choice == "enabled" || choice == "disabled"
        let configuration = URLSessionConfiguration.ephemeral
        configuration.httpCookieStorage = nil
        configuration.httpShouldSetCookies = false
        configuration.urlCache = nil
        configuration.urlCredentialStorage = nil
        configuration.timeoutIntervalForRequest = 10
        configuration.timeoutIntervalForResource = 15
        self.session = session ?? URLSession(configuration: configuration, delegate: NoUsageRedirects(), delegateQueue: nil)
        if enabled { startTimer() }
    }

    static func endpoint(from value: String?) -> URL? {
        guard let value, let url = URL(string: value), url.scheme == "https",
              url.host != nil, url.user == nil, url.password == nil,
              url.query == nil, url.fragment == nil else { return nil }
        return url
    }

    func choose(enabled: Bool) {
        self.enabled = enabled
        choiceMade = true
        defaults.set(enabled ? "enabled" : "disabled", forKey: Self.choiceKey)
        timer?.cancel()
        timer = nil
        upload?.cancel()
        upload = nil
        uploadID = nil
        counts.removeAll()
        if enabled { startTimer() }
    }

    func record(_ metric: UsageMetric) {
        guard enabled, available else { return }
        counts[metric.rawValue] = min(10_000, (counts[metric.rawValue] ?? 0) + 1)
    }

    /// Memory only, no retries or offline queue. Withdrawal discards unsent counts.
    func flush() {
        guard enabled, let endpoint, upload == nil, !counts.isEmpty else { return }
        var request = URLRequest(url: endpoint)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("DramaStudio", forHTTPHeaderField: "User-Agent")
        request.httpBody = try? JSONEncoder().encode(counts)
        counts.removeAll()
        let id = UUID()
        uploadID = id
        upload = session.dataTask(with: request) { [weak self] _, _, _ in
            Task { @MainActor in
                guard let self, self.uploadID == id else { return }
                self.upload = nil
                self.uploadID = nil
            }
        }
        upload?.resume()
    }

    private func startTimer() {
        guard available else { return }
        timer = Task { [weak self] in
            while !Task.isCancelled {
                do { try await Task.sleep(for: .seconds(60)) } catch { return }
                guard let self else { return }
                self.flush()
            }
        }
    }
}

private final class NoUsageRedirects: NSObject, URLSessionTaskDelegate {
    func urlSession(_ session: URLSession, task: URLSessionTask,
                    willPerformHTTPRedirection response: HTTPURLResponse, newRequest request: URLRequest,
                    completionHandler: @escaping (URLRequest?) -> Void) {
        completionHandler(nil)
    }
}

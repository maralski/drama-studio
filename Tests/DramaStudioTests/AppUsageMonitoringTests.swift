// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation
import XCTest
@testable import DramaStudio

private final class UsageRequests: @unchecked Sendable {
    private let lock = NSLock()
    private var received: [URLRequest] = []
    private var stops = 0
    var requests: [URLRequest] { lock.withLock { received } }
    var cancellations: Int { lock.withLock { stops } }
    func add(_ request: URLRequest) { lock.withLock { received.append(request) } }
    func stop() { lock.withLock { stops += 1 } }
}

private final class UsageProtocol: URLProtocol {
    static var recorder = UsageRequests()
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() { Self.recorder.add(request) }
    override func stopLoading() { Self.recorder.stop() }
}

@MainActor
final class AppUsageMonitoringTests: XCTestCase {
    private var defaults: UserDefaults!
    private var suite: String!
    private var session: URLSession!
    private let endpoint = URL(string: "https://usage.example.test/v1/usage")!

    override func setUp() async throws {
        suite = "AppUsageMonitoringTests-\(UUID().uuidString)"
        defaults = UserDefaults(suiteName: suite)!
        UsageProtocol.recorder = UsageRequests()
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [UsageProtocol.self]
        session = URLSession(configuration: configuration)
    }

    override func tearDown() async throws {
        session.invalidateAndCancel()
        defaults.removePersistentDomain(forName: suite)
    }

    private func waitFor(_ condition: () -> Bool) async {
        for _ in 0..<100 {
            if condition() { return }
            try? await Task.sleep(for: .milliseconds(10))
        }
        XCTFail("Expected request state was not reached")
    }

    func testNoRequestBeforeConsentOrAfterDecliningAcrossRelaunch() {
        let monitoring = UsageMonitoring(defaults: defaults, endpoint: endpoint, session: session)
        XCTAssertFalse(monitoring.enabled)
        XCTAssertFalse(monitoring.choiceMade)
        monitoring.record(.launch)
        monitoring.flush()
        monitoring.choose(enabled: false)
        let relaunched = UsageMonitoring(defaults: defaults, endpoint: endpoint, session: session)
        XCTAssertTrue(relaunched.choiceMade)
        XCTAssertFalse(relaunched.enabled)
        relaunched.record(.generationFailed)
        relaunched.flush()
        XCTAssertTrue(UsageProtocol.recorder.requests.isEmpty)
    }

    func testOptInSendsOnlyAllowlistedCountsAndRevocationCancelsAndDropsPendingData() async throws {
        let monitoring = UsageMonitoring(defaults: defaults, endpoint: endpoint, session: session)
        monitoring.record(.generationFailed) // Must not be backfilled after consent.
        monitoring.choose(enabled: true)
        monitoring.record(.launch)
        monitoring.record(.generationSucceeded)
        monitoring.record(.generationSucceeded)
        monitoring.flush()
        await waitFor { UsageProtocol.recorder.requests.count == 1 }
        let request = try XCTUnwrap(UsageProtocol.recorder.requests.first)
        XCTAssertEqual(request.url, endpoint)
        XCTAssertEqual(request.httpMethod, "POST")
        XCTAssertNil(request.value(forHTTPHeaderField: "Authorization"))
        XCTAssertNil(request.value(forHTTPHeaderField: "Cookie"))
        let body: Data
        if let data = request.httpBody {
            body = data
        } else {
            let stream = try XCTUnwrap(request.httpBodyStream)
            stream.open()
            defer { stream.close() }
            var data = Data()
            var buffer = [UInt8](repeating: 0, count: 1024)
            while stream.hasBytesAvailable {
                let count = stream.read(&buffer, maxLength: buffer.count)
                if count <= 0 { break }
                data.append(buffer, count: count)
            }
            body = data
        }
        XCTAssertEqual(try JSONDecoder().decode([String: Int].self, from: body),
                       ["app.launch": 1, "generation.succeeded": 2])
        monitoring.record(.setupFailed)
        monitoring.choose(enabled: false)
        await waitFor { UsageProtocol.recorder.cancellations == 1 }
        monitoring.record(.generationFailed)
        monitoring.flush()
        monitoring.choose(enabled: true)
        monitoring.flush()
        XCTAssertEqual(UsageProtocol.recorder.requests.count, 1)
        monitoring.choose(enabled: false)
    }

    func testMissingConfigurationDoesNotSendEvenWithConsent() {
        let monitoring = UsageMonitoring(defaults: defaults, endpoint: nil, session: session)
        monitoring.choose(enabled: true)
        monitoring.record(.launch)
        monitoring.flush()
        XCTAssertFalse(monitoring.available)
        XCTAssertTrue(UsageProtocol.recorder.requests.isEmpty)
        monitoring.choose(enabled: false)
    }

    func testUnknownConsentValueDoesNotEnableSharing() {
        defaults.set("old-version", forKey: UsageMonitoring.choiceKey)
        let monitoring = UsageMonitoring(defaults: defaults, endpoint: endpoint, session: session)
        XCTAssertFalse(monitoring.enabled)
        XCTAssertFalse(monitoring.choiceMade)
    }

    func testEndpointRejectsUnencryptedOrIdentifyingConfiguration() {
        for value in ["http://example.test/usage", "https://user:pass@example.test/usage",
                      "https://example.test/usage?user=123", "https://example.test/usage#user", ""] {
            XCTAssertNil(UsageMonitoring.endpoint(from: value))
        }
        XCTAssertEqual(UsageMonitoring.endpoint(from: endpoint.absoluteString), endpoint)
    }
}

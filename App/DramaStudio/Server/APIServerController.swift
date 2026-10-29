// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import AppKit
import DramaEngine
import DramaServer
import Foundation

@MainActor @Observable
final class APIServerController {
    static let startsWithAppKey = "apiServerStartsWithApp"
    static let portKey = "apiServerPort"

    private(set) var state: ServerState = .stopped
    private(set) var job: SpeechJob?
    private(set) var token: String?
    let session: SessionModel
    @ObservationIgnored private var server: APIServer?
    @ObservationIgnored private lazy var jobs = SpeechJobSlot { job in
        DispatchQueue.main.async { MainActor.assumeIsolated { self.jobChanged(job) } }
    }

    init(session: SessionModel) {
        self.session = session
        AppHome.defaults.register(defaults: [Self.startsWithAppKey: true, Self.portKey: Int(APIService.defaultPort)])
    }

    var port: UInt16 { UInt16(exactly: AppHome.defaults.integer(forKey: Self.portKey)) ?? APIService.defaultPort }
    var url: String { "http://127.0.0.1:\(state.port ?? port)" }

    var isRunning: Bool {
        switch state {
        case .starting, .listening: true
        case .stopped, .failed: false
        }
    }

    var statusLine: String {
        switch state {
        case .stopped: "API: off"
        case .starting: "API: starting on port \(port)"
        case .listening: "API: \(url)"
        case .failed(let message): "API: \(message)"
        }
    }

    func startIfEnabled() {
        if AppHome.defaults.bool(forKey: Self.startsWithAppKey) { start() }
    }

    func start() {
        guard server == nil else { return }
        do {
            token = try APIToken.loadOrCreate()
        } catch {
            state = .failed("Couldn't create the API token: \(error.localizedDescription)")
            return
        }
        let service = APIService(
            synthesizer: session.synthesizer,
            voices: session.library,
            token: token,
            version: Self.version,
            jobs: jobs,
            engineBusyElsewhere: { [session] in await session.state.isGenerating },
            refusal: { [session] in await session.generationRefusal() }
        )
        let server = APIServer(service: service, port: port) { state in
            DispatchQueue.main.async { MainActor.assumeIsolated { self.serverChanged(state, from: service) } }
        }
        self.server = server
        server.start()
    }

    func stop(then next: (@MainActor () -> Void)? = nil) {
        guard let server else {
            state = .stopped
            next?()
            return
        }
        self.server = nil
        state = .stopped
        server.stop { DispatchQueue.main.async { MainActor.assumeIsolated { next?() } } }
    }

    func toggle() {
        isRunning ? stop() : start()
    }

    /// After a port change: also retries a server that failed to bind.
    func restart() {
        guard isRunning || state != .stopped else { return }
        stop { self.start() }
    }

    func copyURL() { Self.copy(url) }
    func copyToken() { token.map(Self.copy) }

    private func serverChanged(_ next: ServerState, from service: APIService) {
        guard server?.service === service else { return }
        state = next
        if case .failed = next { server = nil }
    }

    private func jobChanged(_ next: SpeechJob?) {
        job = next
        session.refreshEngineState()
    }

    private static func copy(_ string: String) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(string, forType: .string)
    }

    static var version: String {
        let info = Bundle.main.infoDictionary ?? [:]
        guard let short = info["CFBundleShortVersionString"] as? String else { return "dev" }
        return "\(short) (\(info["CFBundleVersion"] as? String ?? "0"))"
    }
}

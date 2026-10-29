// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import DramaEngine
import DramaServer
import SwiftUI

struct MenuBarContent: View {
    let model: SessionModel
    let server: APIServerController
    let presence: AppPresence
    let setup: SetupCoordinator
    @Environment(\.openWindow) private var openWindow

    var body: some View {
        Text(MenuBarStatus.model(preview: model.synthesizer is MockSynthesizer, loaded: model.modelLoaded, session: model.state, job: server.job))
        Divider()
        Text(server.statusLine)
        Button("Copy API URL") { server.copyURL() }
            .disabled(server.state.port == nil)
        Button("Copy API Token") { server.copyToken() }
            .disabled(server.token == nil)
        Divider()
        Text(MenuBarStatus.job(session: model.state, job: server.job))
        Divider()
        Button(presence.hasWindow ? "Show Drama Studio" : "Open Drama Studio") {
            presence.showMainWindow(openWindow, id: setup.stage == .ready ? DramaStudioApp.mainWindowID : SetupView.windowID)
        }
        Button(server.isRunning ? "Stop API Server" : "Start API Server") { server.toggle() }
        Divider()
        Button("Quit Drama Studio") { NSApp.terminate(nil) }
            .keyboardShortcut("q")
    }
}

enum MenuBarStatus {
    static func model(preview: Bool, loaded: Bool, session: PlaybackState, job: SpeechJob?) -> String {
        let engine = preview ? "Preview engine" : "Drama 3 on Metal"
        if case .loading(_, let detail, _) = session { return "\(engine): loading, \(detail)" }
        if case .loadingModel(let detail)? = job?.progress { return "\(engine): loading, \(detail)" }
        return "\(engine): \(loaded ? "loaded" : "not loaded yet")"
    }

    static func job(session: PlaybackState, job: SpeechJob?) -> String {
        if let job { return "API request: \(job.summary)" }
        switch session {
        case .loading(_, let detail, _): return "Loading model: \(detail)"
        case .generating(let chunk, let count, _): return "Generating chunk \(chunk) of \(count)"
        default: return "No generation running"
        }
    }
}

struct APIChip: View {
    let server: APIServerController

    var body: some View {
        if let port = server.state.port {
            Label("API", systemImage: server.job == nil ? "circle.fill" : "waveform")
                .labelStyle(APIChipLabelStyle())
                .help(server.job.map { "API request: \($0.summary)" } ?? "API server listening on http://127.0.0.1:\(port)")
        }
    }
}

private struct APIChipLabelStyle: LabelStyle {
    func makeBody(configuration: Configuration) -> some View {
        HStack(spacing: 4) {
            configuration.icon
                .font(.system(size: 6))
                .foregroundStyle(.green)
            configuration.title
                .font(.caption2.weight(.semibold))
        }
        .padding(.horizontal, 7)
        .padding(.vertical, 2)
        .background(.quaternary, in: Capsule())
    }
}

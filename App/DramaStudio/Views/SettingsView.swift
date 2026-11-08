// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import AppKit
import DramaEngine
import DramaServer
import SwiftUI

enum SettingsTab: Hashable { case general, updates, pronunciation }

struct SettingsView: View {
    let server: APIServerController
    let setup: SetupCoordinator
    let ming: MingInstall
    @Binding var selection: SettingsTab
    @State private var modelStatus: ModelSetup.Status?

    var body: some View {
        TabView(selection: $selection) {
            Tab("General", systemImage: "gearshape", value: SettingsTab.general) {
                GeneralSettingsView(server: server, setup: setup, ming: ming, status: $modelStatus)
            }
            Tab("Updates", systemImage: "arrow.down.circle", value: SettingsTab.updates) {
                UpdateSettingsView()
            }
            Tab("Pronunciation", systemImage: "character.book.closed", value: SettingsTab.pronunciation) {
                PronunciationSettingsView()
            }
        }
        .frame(width: 560)
        .frame(minHeight: 460, idealHeight: 600)
    }
}

private struct GeneralSettingsView: View {
    let server: APIServerController
    let setup: SetupCoordinator
    let ming: MingInstall
    @AppStorage(ModelSetup.customDirectoryKey) private var modelPath = ""
    @AppStorage(APIServerController.startsWithAppKey) private var startsWithApp = true
    @AppStorage(APIServerController.portKey) private var port = Int(APIService.defaultPort)
    @AppStorage(SessionModel.streamsAudioKey) private var streamsAudio = true
    @Binding var status: ModelSetup.Status?
    @State private var codecReady = true

    var body: some View {
        Form {
            Section(SpeechEngineID.drama3.capabilities.name) {
                LabeledContent {
                    Text(location).textSelection(.enabled).lineLimit(2).truncationMode(.middle)
                } label: {
                    HelpLabel("Location", help: .modelLocation)
                }
                LabeledContent("Status") { statusLabel }
                HStack {
                    Button("Choose Folder\u{2026}") { chooseFolder() }
                    Button("Use Default Location") { modelPath = "" }
                        .disabled(modelPath.isEmpty)
                }
                if let problem = status?.problem {
                    Text(problem).font(.caption).foregroundStyle(.secondary)
                }
                if !codecReady, let command = codecCommand {
                    Text("The Core ML codec models are missing, and this build has no codec download. Build them once in Terminal:")
                        .font(.caption).foregroundStyle(.secondary)
                    HStack(alignment: .top) {
                        Text(command).font(.caption.monospaced()).textSelection(.enabled)
                        Spacer()
                        Button("Copy Command") {
                            NSPasteboard.general.clearContents()
                            NSPasteboard.general.setString(command, forType: .string)
                        }
                    }
                    .padding(8)
                    .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 6))
                }
            }
            MingSection(install: ming)
            Section("Playback") {
                Toggle(isOn: $streamsAudio) { HelpLabel("Stream audio while generating", help: .streaming) }
            }
            APIServerSection(server: server, startsWithApp: $startsWithApp, port: $port)
            Section("Privacy") {
                Toggle("Share anonymous usage information", isOn: Binding(
                    get: { UsageMonitoring.shared.enabled },
                    set: { UsageMonitoring.shared.choose(enabled: $0) }
                ))
                Text(UsageMonitoring.explanation).font(.caption).foregroundStyle(.secondary)
            }
            Section("About") {
                Text("Drama Studio by Peter Marelas").font(.headline)
                Text("Built with Fish Audio. Drama Studio runs Fish Audio Drama 3 locally on Apple silicon.").foregroundStyle(.secondary)
                Link("Fish Audio Community License", destination: TermsAgreement.licenseURL)
                if let date = TermsAgreement(defaults: AppHome.defaults).acceptedDate {
                    Text("You agreed to its terms on \(date.formatted(date: .long, time: .shortened)).").foregroundStyle(.secondary)
                }
            }
        }
        .formStyle(.grouped)
        .task(id: modelPath) {
            let path = modelPath
            let result = await Task.detached {
                (ModelSetup.status(customPath: path), DramaCodec.modelsExist(in: AppHome.codec))
            }.value
            guard !Task.isCancelled else { return }
            status = result.0
            codecReady = result.1
        }
        .onChange(of: modelPath) { status = nil; setup.resume() }
    }

    private var codecCommand: String? {
        CodecBundle(info: Bundle.main.infoDictionary) == nil ? ModelSetup.codecConversionCommand() : nil
    }

    private var location: String {
        switch status {
        case .ready(let url), .incomplete(let url): url.path
        case nil: modelPath.isEmpty ? AppHome.models.path : modelPath
        case .missing: modelPath.isEmpty ? "\(AppHome.models.path) (not downloaded)" : modelPath
        }
    }

    @ViewBuilder
    private var statusLabel: some View {
        switch status {
        case nil: Label("Checking…", systemImage: "clock").foregroundStyle(.secondary)
        case .ready: Label("Ready", systemImage: "checkmark.circle.fill").foregroundStyle(.green)
        case .incomplete: Label("Incomplete", systemImage: "exclamationmark.triangle.fill").foregroundStyle(.orange)
        case .missing: Label("Not found", systemImage: "xmark.circle.fill").foregroundStyle(.red)
        }
    }

    private func chooseFolder() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.message = "Choose the folder that holds config.json, tokenizer.json, the safetensors shards, and codec.pth."
        if panel.runModal() == .OK, let url = panel.url { modelPath = url.path }
    }
}

/// Ming is optional, so it lives here rather than in the setup window: where
/// it was found, what it lacks, and a download of the pinned revision.
private struct MingSection: View {
    let install: MingInstall
    @AppStorage(ModelSetup.mingDirectoryKey) private var path = ""

    var body: some View {
        Section("Ming") {
            LabeledContent("Location") {
                Text(location).textSelection(.enabled).lineLimit(2).truncationMode(.middle)
            }
            LabeledContent("Status") { status }
            HStack {
                Button("Choose Folder\u{2026}") { chooseFolder() }
                Button("Use Default Location") { path = "" }
                    .disabled(path.isEmpty)
                Spacer()
                download
            }
            if let gap = install.report?.items.first(where: { $0.isMissing && $0.component != .wordRecognizer }) {
                Text(gap.summary + " " + (gap.fix ?? "")).font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
            }
            if case .failed(let message) = install.phase {
                Text(message).font(.caption).foregroundStyle(.red)
            }
        }
        .task(id: path) { await install.refresh() }
    }

    private var location: String {
        install.report?.directory?.path ?? "\(AppHome.mingModels.path) (not downloaded)"
    }

    @ViewBuilder
    private var status: some View {
        switch (install.report, install.phase) {
        case (_, .downloading(let progress)):
            ProgressView(value: progress.fraction) {
                Text("Downloading \(ModelSetup.gigabytes(progress.doneBytes)) of \(ModelSetup.gigabytes(progress.totalBytes))")
            }
        case (nil, _): Label("Checking\u{2026}", systemImage: "clock").foregroundStyle(.secondary)
        case _ where install.isReady: Label("Ready", systemImage: "checkmark.circle.fill").foregroundStyle(.green)
        case (let report?, _) where report.directory != nil: Label("Incomplete", systemImage: "exclamationmark.triangle.fill").foregroundStyle(.orange)
        default: Label("Not installed", systemImage: "xmark.circle.fill").foregroundStyle(.secondary)
        }
    }

    @ViewBuilder
    private var download: some View {
        if case .downloading = install.phase {
            Button("Stop Download") { install.cancelDownload() }
        } else if !install.isReady, path.isEmpty, install.report != nil, ProcessInfo.processInfo.environment["MING_WEIGHTS"] == nil {
            Button("Download (\(ModelSetup.gigabytes(SpeechEngineID.ming.install.plan.totalBytes)))") { install.startDownload() }
        }
    }

    private func chooseFolder() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.message = "Choose the folder that holds Ming's model.safetensors, config.json, tokenizer.json and campplus.onnx."
        if panel.runModal() == .OK, let url = panel.url { path = url.path }
    }
}

private struct APIServerSection: View {
    let server: APIServerController
    @Binding var startsWithApp: Bool
    @Binding var port: Int

    private var mcpPath: String? {
        let url = Bundle.main.bundleURL.appending(path: "Contents/MacOS/drama-mcp")
        return FileManager.default.isExecutableFile(atPath: url.path) ? url.path : nil
    }

    var body: some View {
        Section("API Server") {
            Toggle("Start the API server with Drama Studio", isOn: $startsWithApp)
            LabeledContent("Port") {
                TextField("Port", value: $port, format: .number.grouping(.never))
                    .frame(width: 80)
                    .labelsHidden()
                    .multilineTextAlignment(.trailing)
                    .onSubmit { server.restart() }
            }
            LabeledContent("Status") {
                HStack {
                    Text(server.statusLine).textSelection(.enabled)
                    Button(server.isRunning ? "Stop" : "Start") { server.toggle() }
                }
            }
            LabeledContent("Bearer token") {
                if let token = server.token {
                    HStack {
                        Text(token).font(.caption.monospaced()).textSelection(.enabled).lineLimit(1).truncationMode(.middle)
                        Button("Copy") { server.copyToken() }
                    }
                } else {
                    Text("Created when the server first starts").foregroundStyle(.secondary)
                }
            }
            if let mcpPath {
                LabeledContent("MCP server") {
                    HStack {
                        Text(mcpPath).font(.caption.monospaced()).textSelection(.enabled).lineLimit(1).truncationMode(.head)
                        Button("Copy") {
                            NSPasteboard.general.clearContents()
                            NSPasteboard.general.setString(mcpPath, forType: .string)
                        }
                    }
                }
            }
        }
    }
}

import AppKit
import DramaServer
import SwiftUI

struct SettingsView: View {
    let server: APIServerController
    @AppStorage(ModelSetup.customDirectoryKey) private var modelPath = ""
    @AppStorage(APIServerController.startsWithAppKey) private var startsWithApp = true
    @AppStorage(APIServerController.portKey) private var port = Int(APIService.defaultPort)
    @State private var status: ModelSetup.Status = .missing

    var body: some View {
        Form {
            Section("Model") {
                LabeledContent {
                    Text(location).textSelection(.enabled).lineLimit(2).truncationMode(.middle)
                } label: {
                    HelpLabel("Location", help: .modelLocation)
                }
                LabeledContent("Status") { statusLabel }
                HStack {
                    Button("Choose Folder\u{2026}") { chooseFolder() }
                    Button("Use Hugging Face Cache") { modelPath = "" }
                        .disabled(modelPath.isEmpty)
                }
                if let problem = status.problem {
                    Text(problem).font(.caption).foregroundStyle(.secondary)
                    HStack {
                        Text(ModelSetup.downloadCommand).font(.body.monospaced()).textSelection(.enabled)
                        Spacer()
                        Button("Copy Command") {
                            NSPasteboard.general.clearContents()
                            NSPasteboard.general.setString(ModelSetup.downloadCommand, forType: .string)
                        }
                    }
                    .padding(8)
                    .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 6))
                }
            }
            APIServerSection(server: server, startsWithApp: $startsWithApp, port: $port)
            Section("About") {
                Text("Built with Fish Audio").font(.headline)
                Text("Drama Studio runs Fish Audio Drama 3 locally on Apple silicon.").foregroundStyle(.secondary)
                Link("Fish Audio Community License", destination: ModelSetup.licenseURL)
            }
        }
        .formStyle(.grouped)
        .frame(width: 520)
        .task(id: modelPath) { status = ModelSetup.status(customPath: modelPath) }
    }

    private var location: String {
        switch status {
        case .ready(let url), .incomplete(let url): url.path
        case .missing: modelPath.isEmpty ? "Hugging Face cache (not downloaded)" : modelPath
        }
    }

    @ViewBuilder
    private var statusLabel: some View {
        switch status {
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

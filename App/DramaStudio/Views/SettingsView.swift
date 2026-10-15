import AppKit
import SwiftUI

struct SettingsView: View {
    @AppStorage(ModelSetup.customDirectoryKey) private var modelPath = ""
    @State private var status: ModelSetup.Status = .missing

    var body: some View {
        Form {
            Section("Model") {
                LabeledContent("Location") {
                    Text(location).textSelection(.enabled).lineLimit(2).truncationMode(.middle)
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

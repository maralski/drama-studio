import SwiftUI

struct ScriptEditorView: View {
    @Bindable var model: SessionModel
    @State private var isDropTarget = false
    @AppStorage(ModelSetup.customDirectoryKey) private var modelPath = ""
    @State private var modelStatus: ModelSetup.Status?

    var body: some View {
        VStack(spacing: 0) {
            if let notice = model.notice {
                InlineBanner(message: notice, systemImage: "exclamationmark.triangle.fill", tint: .orange) { model.notice = nil }
            }
            if let problem = modelStatus?.problem {
                InlineBanner(message: problem, systemImage: "arrow.down.circle", tint: .accentColor, onDismiss: nil)
            }
            editor
            Divider()
            ScriptFooter(model: model)
        }
        .navigationTitle("Drama Studio")
        .task(id: modelPath) { modelStatus = ModelSetup.status(customPath: modelPath) }
    }

    private var editor: some View {
        TextEditor(text: $model.script, selection: $model.selection)
            .font(.system(size: 15))
            .lineSpacing(3)
            .contentMargins(.horizontal, 14, for: .scrollContent)
            .contentMargins(.vertical, 12, for: .scrollContent)
            .scrollContentBackground(.hidden)
            .background(Color(nsColor: .textBackgroundColor))
            .overlay(alignment: .topLeading) {
                if model.script.isEmpty {
                    Text("Type or paste a script, or drop a text, Word, or subtitle file here.\nUse \u{201C}Name: line\u{201D} dialogue with Screenplay Mode for several voices.")
                        .font(.system(size: 15))
                        .foregroundStyle(.tertiary)
                        .padding(.horizontal, 19)
                        .padding(.vertical, 12)
                        .allowsHitTesting(false)
                }
            }
            .overlay {
                if isDropTarget {
                    RoundedRectangle(cornerRadius: 8).strokeBorder(Color.accentColor, lineWidth: 3).padding(4)
                }
            }
            .dropDestination(for: URL.self) { urls, _ in
                guard let url = urls.first else { return false }
                model.importScript(from: url)
                return true
            } isTargeted: { isDropTarget = $0 }
    }
}

private struct ScriptFooter: View {
    let model: SessionModel

    var body: some View {
        HStack(spacing: 12) {
            Text("\(model.script.count.formatted()) characters")
            switch model.prepared {
            case .success(let script):
                Text(script.speakers.count == 1 ? "1 voice" : "\(script.speakers.count) speakers")
            case .failure(.empty):
                EmptyView()
            case .failure(let error):
                Label(error.localizedDescription, systemImage: "exclamationmark.triangle.fill")
                    .foregroundStyle(.red)
            }
            Spacer()
            if model.screenplayMode {
                Text("Screenplay Mode").foregroundStyle(.secondary)
            }
        }
        .font(.caption)
        .foregroundStyle(.secondary)
        .padding(.horizontal, 14)
        .padding(.vertical, 6)
    }
}

struct InlineBanner: View {
    let message: String
    let systemImage: String
    let tint: Color
    var onDismiss: (() -> Void)?

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            Image(systemName: systemImage).foregroundStyle(tint)
            Text(message).textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading)
            if let onDismiss {
                Button("Dismiss", systemImage: "xmark", action: onDismiss)
                    .labelStyle(.iconOnly)
                    .buttonStyle(.borderless)
            }
        }
        .font(.callout)
        .padding(.horizontal, 14)
        .padding(.vertical, 8)
        .background(tint.opacity(0.12))
    }
}

// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import AppKit
import SwiftUI

struct ScriptEditorView: View {
    @Bindable var model: SessionModel
    @State private var isDropTarget = false
    @AppStorage(ModelSetup.customDirectoryKey) private var modelPath = ""
    @State private var modelStatus: ModelSetup.Status?
    @State private var locator = HighlightLocator()

    var body: some View {
        VStack(spacing: 0) {
            if let url = model.exportedURL {
                HStack {
                    Label("Exported \(url.lastPathComponent)", systemImage: "checkmark.circle.fill")
                    Spacer()
                    Button("Show in Finder") { NSWorkspace.shared.activateFileViewerSelecting([url]) }
                    Button("Dismiss", systemImage: "xmark") { model.exportedURL = nil }.labelStyle(.iconOnly)
                }
                .font(.callout).padding(10).background(Color.green.opacity(0.1))
            }
            if let notice = model.notice {
                InlineBanner(message: notice, systemImage: "exclamationmark.triangle.fill", tint: .orange) { model.notice = nil }
            }
            if model.canExport, let warning = model.exportWarning {
                InlineBanner(message: warning, systemImage: "clock.badge.exclamationmark", tint: .orange, onDismiss: nil)
            }
            if !ModelSetup.isPreviewEngine, let problem = modelStatus?.problem {
                InlineBanner(message: problem, systemImage: "arrow.down.circle", tint: .accentColor, onDismiss: nil)
            }
            TimelineView(.animation(minimumInterval: 0.1, paused: !model.state.isPlaying)) { _ in
                let highlight = model.state.isPlaying ? model.highlight(at: model.position) : nil
                editor(highlight: model.document == nil || highlight?.segmentID == model.selectedSegmentID ? highlight : nil)
            }
            Divider()
            ScriptFooter(model: model)
        }
        .navigationTitle(model.projectName)
        .background(ProjectWindowState(model: model).frame(width: 0, height: 0))
        .task(id: modelPath) { modelStatus = ModelSetup.status(customPath: modelPath) }
        .overlay {
            if isDropTarget {
                RoundedRectangle(cornerRadius: 8).strokeBorder(Color.accentColor, lineWidth: 3).padding(4)
            }
        }
        .dropDestination(for: URL.self) { urls, _ in
            guard let url = urls.first else { return false }
            model.open(url)
            return true
        } isTargeted: { isDropTarget = $0 }
    }

    private func editor(highlight: SpokenHighlight?) -> some View {
        ScriptTextView(text: $model.editorText, selection: $model.selection, highlight: locator.range(of: highlight, in: model.editorText))
            .background(Color(nsColor: .textBackgroundColor))
            .overlay(alignment: .topLeading) {
                if model.editorText.isEmpty && model.document == nil {
                    VStack(alignment: .leading, spacing: 14) {
                        Text("Type or paste a script, or drop a text, Word, subtitle, PowerPoint, PDF, or EPUB file here.\nUse \u{201C}Name: line\u{201D} dialogue with Screenplay Mode for several voices.")
                            .font(.system(size: 15))
                            .foregroundStyle(.secondary)
                            .allowsHitTesting(false)
                        HStack {
                            Button("Try an Example") { model.insert(Examples.firstLine) }
                            Button("Import Document…") { FilePanels.importScript(into: model) }
                            Button("Open Project…") { FilePanels.openProject(into: model) }
                        }
                    }
                    .padding(.horizontal, 19)
                    .padding(.vertical, 12)
                }
            }
    }
}

/// Finds the highlighted sentence's range in the editor text, remembering
/// the last text and sentence list so a 10 Hz tick costs one array lookup.
final class HighlightLocator {
    private var texts: [String] = []
    private var text = ""
    private var ranges: [NSRange?] = []

    func range(of highlight: SpokenHighlight?, in text: String) -> NSRange? {
        guard let highlight else { return nil }
        if highlight.texts != texts || text != self.text {
            texts = highlight.texts
            self.text = text
            ranges = SentenceLocator.ranges(of: texts, in: text)
        }
        return ranges.indices.contains(highlight.index) ? ranges[highlight.index] : nil
    }
}

private struct ScriptFooter: View {
    let model: SessionModel

    var body: some View {
        HStack(spacing: 12) {
            if let document = model.document {
                Text("\(document.segments.count) sections, \(SectionRequests.queue(document).count) to narrate, \(model.narrationTakes.count) narrated")
                Text("\(model.editorText.count.formatted()) characters in this section")
            } else {
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
            }
            Spacer()
            if model.screenplayMode && model.document == nil {
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

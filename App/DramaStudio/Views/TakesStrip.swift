// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import DramaEngine
import SwiftUI

struct TakesStrip: View {
    let model: SessionModel
    @State private var note = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            ScrollView(.horizontal) {
                HStack(spacing: 6) {
                    if let section = model.selectedSegment, model.document != nil {
                        Text(section.label).font(.caption).foregroundStyle(.secondary).lineLimit(1)
                    }
                    ForEach(model.visibleTakes) { take in
                        TakeChip(take: take, isSelected: take.id == model.visibleSelectedTakeID, isKept: take.id == model.visibleExportTakeID, isOutdated: model.isOutdated(take, in: model.selectedSegment)) {
                            model.selectTake(take.id)
                        }
                    }
                }
                .padding(2)
            }
            .scrollIndicators(.never)
            .fixedSize(horizontal: false, vertical: true)
            .defaultScrollAnchor(.leading, for: .alignment)
            .defaultScrollAnchor(.trailing, for: .sizeChanges)
            HStack {
                Text(model.exportSummary).font(.caption).foregroundStyle(.secondary)
                if model.previewingTakeID != nil {
                    Button("Stop Audition", systemImage: "stop.fill") { model.stopPreview() }
                }
                Spacer()
                Button(isSelectedKept ? "Used for Export" : "Use for Export", systemImage: "checkmark.circle") {
                    model.useSelectedTakeForExport()
                }
                .disabled(model.visibleSelectedTakeID == nil || model.state.isGenerating || isSelectedKept)
                .help("Use the selected take in exported audio and captions")
            }
            if model.document == nil {
            HStack(spacing: 8) {
                TextField("Give a note", text: $note, prompt: Text("Give a note, such as softer and warmer"))
                    .textFieldStyle(.roundedBorder)
                    .frame(maxWidth: 320)
                    .onSubmit(retake)
                HelpButton(.directorsNote)
                Button("Retake", systemImage: "arrow.clockwise", action: retake)
                    .disabled(!model.canRetake || DirectorsNote.clean(note) == nil)
                    .help("Read the line again with this note, after the takes so far")
                Spacer()
            }
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 140), alignment: .leading)], alignment: .leading, spacing: 6) {
                    ForEach(Examples.notes, id: \.self) { suggestion in
                        Button(suggestion) { note = suggestion }
                            .buttonStyle(.bordered)
                            .buttonBorderShape(.capsule)
                            .controlSize(.small)
                            .fixedSize(horizontal: false, vertical: true)
                    }
            }
            .fixedSize(horizontal: false, vertical: true)
            if let blocker = model.retakeBlocker {
                Text(blocker).font(.caption).foregroundStyle(.secondary)
            }
            }
        }
        .padding(.horizontal, 16)
        .padding(.top, 12)
        .padding(.bottom, 16)
        .background(.bar)
        .overlay(alignment: .top) { Divider() }
    }

    private var isSelectedKept: Bool { model.visibleSelectedTakeID != nil && model.visibleSelectedTakeID == model.visibleExportTakeID }

    private func retake() {
        guard model.canRetake else { return }
        model.retake(note: note)
    }
}

private struct TakeChip: View {
    let take: SessionTake
    let isSelected: Bool
    let isKept: Bool
    let isOutdated: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 4) {
                if isKept { Label("Export", systemImage: "checkmark.circle.fill").font(.caption) }
                if isOutdated { Label("Outdated", systemImage: "clock.badge.exclamationmark").font(.caption) }
                Text(take.label).lineLimit(1)
            }
        }
        .buttonStyle(.bordered)
        .buttonBorderShape(.capsule)
        .overlay { Capsule().strokeBorder(Color.accentColor, lineWidth: isSelected ? 2 : 0) }
        .accessibilityLabel("\(take.label)\(isKept ? ", used for export" : "")\(isOutdated ? ", outdated" : "")")
        .accessibilityAddTraits(isSelected ? .isSelected : [])
        .help("Click to audition. Use for Export changes the exported take.")
    }
}

extension SessionTake {
    var label: String {
        (["Take \(number)", TransportBar.seconds(duration)] + [note].compactMap { $0 }).joined(separator: " \u{00B7} ")
    }
}

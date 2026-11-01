// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import DramaEngine
import SwiftUI

struct TakesStrip: View {
    let model: SessionModel
    var resizable = false
    @State private var note = ""
    @State private var takeToDelete: SessionTake?

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .firstTextBaseline, spacing: 8) {
                Text("Takes (\(model.visibleTakes.count))").font(.subheadline.weight(.semibold))
                    .fixedSize()
                if let section = model.selectedSegment, model.document != nil {
                    Text(section.label).font(.caption).foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            ScrollViewReader { scroll in
                ScrollView(.vertical) {
                    LazyVGrid(columns: [GridItem(.adaptive(minimum: 220), alignment: .top)], spacing: 8) {
                        ForEach(model.visibleTakes) { take in
                            TakeCard(take: take, isSelected: take.id == model.visibleSelectedTakeID, isKept: take.id == model.visibleExportTakeID, isOutdated: model.isOutdated(take, in: model.selectedSegment), delete: {
                                takeToDelete = take
                            }) {
                                model.selectTake(take.id)
                            }
                            .id(take.id)
                            .disabled(model.state.isGenerating)
                        }
                    }
                    .padding(2)
                    .padding(.trailing, 10)
                }
                .scrollIndicators(.visible)
                .frame(height: resizable ? nil : (model.visibleTakes.count == 1 ? 76 : 152))
                .frame(minHeight: resizable ? 76 : nil, maxHeight: resizable ? .infinity : nil)
                .onChange(of: model.visibleSelectedTakeID, initial: true) { _, id in
                    if let id { scroll.scrollTo(id, anchor: .center) }
                }
            }
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
        .confirmationDialog("Delete this take?", isPresented: Binding(
            get: { takeToDelete != nil },
            set: { if !$0 { takeToDelete = nil } }
        ), titleVisibility: .visible, presenting: takeToDelete) { take in
            Button("Delete Take \(take.number)", role: .destructive) { model.deleteTake(take.id) }
            Button("Cancel", role: .cancel) {}
        } message: { take in
            Text(take.id == model.visibleExportTakeID
                 ? "This take is used for export. The newest remaining take will replace it. If none remain, this audio will be removed."
                 : "This removes the take and its audio from the project. This cannot be undone.")
        }
    }

    private var isSelectedKept: Bool { model.visibleSelectedTakeID != nil && model.visibleSelectedTakeID == model.visibleExportTakeID }

    private func retake() {
        guard model.canRetake else { return }
        model.retake(note: note)
    }
}

private struct TakeCard: View {
    let take: SessionTake
    let isSelected: Bool
    let isKept: Bool
    let isOutdated: Bool
    let delete: () -> Void
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    Label("Take \(take.number)", systemImage: "play.fill")
                        .font(.callout.weight(.medium))
                    Spacer(minLength: 8)
                    Text(TransportBar.seconds(take.duration))
                        .font(.callout.monospacedDigit()).foregroundStyle(.secondary)
                }
                .padding(.trailing, 26)
                if isKept || isOutdated || !take.hasCurrentWordCheck {
                    HStack(spacing: 8) {
                        if isKept { Label("Export", systemImage: "checkmark.circle.fill") }
                        if isOutdated { Label("Outdated", systemImage: "clock.badge.exclamationmark") }
                        if !take.hasCurrentWordCheck {
                            Text(take.isIncomplete ? "Incomplete" : (take.wordsChecked ? "Earlier word check" : "Unchecked"))
                                .help(take.isIncomplete ? "Generation ended before the whole script was completed." : "This take has not passed the current spoken-word checks. Generate with Quick Preview off for checked audio.")
                        }
                    }
                    .font(.caption).foregroundStyle(.secondary)
                }
                if let note = take.note {
                    Text(note).font(.caption).foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            .padding(10)
            .frame(maxWidth: .infinity, minHeight: 64, alignment: .leading)
            .background(isSelected ? Color.accentColor.opacity(0.1) : Color.primary.opacity(0.04), in: RoundedRectangle(cornerRadius: 8))
            .contentShape(RoundedRectangle(cornerRadius: 8))
        }
        .buttonStyle(.plain)
        .overlay { RoundedRectangle(cornerRadius: 8).strokeBorder(Color.accentColor, lineWidth: isSelected ? 2 : 0) }
        .accessibilityLabel("\(take.label)\(isKept ? ", used for export" : "")\(isOutdated ? ", outdated" : "")\(take.isIncomplete ? ", incomplete" : (take.hasCurrentWordCheck ? "" : (take.wordsChecked ? ", earlier word check" : ", unchecked")))")
        .accessibilityAddTraits(isSelected ? .isSelected : [])
        .help("Click to audition. Use for Export changes the exported take.")
        .overlay(alignment: .topTrailing) {
            Button(role: .destructive, action: delete) { Image(systemName: "trash") }
                .buttonStyle(.borderless)
                .accessibilityLabel("Delete Take \(take.number)")
                .help("Delete Take \(take.number)")
                .padding(10)
        }
    }
}

extension SessionTake {
    var label: String {
        (["Take \(number)", TransportBar.seconds(duration)] + [note].compactMap { $0 }).joined(separator: " \u{00B7} ")
    }
}

// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import DramaEngine
import SwiftUI

struct TakesStrip: View {
    let model: SessionModel
    @State private var note = ""
    @State private var takeToDelete: SessionTake?
    @State private var takeToSave: SessionTake?
    /// What whole rows of tiles need, as last laid out. One plain row until the first layout.
    @State private var wholeRows: CGFloat = 68
    /// The grid height the user dragged to; nil shows whole rows.
    @State private var chosenHeight: CGFloat?
    @State private var gridHeight: CGFloat = 0
    @State private var dragStart: CGFloat?

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
                    TakeGrid {
                        ForEach(model.visibleTakes) { take in
                            TakeCard(take: take, engine: model.engineLabel(for: take), isSelected: take.id == model.visibleSelectedTakeID, isKept: take.id == model.visibleExportTakeID, isOutdated: model.isOutdated(take, in: model.selectedSegment), delete: {
                                takeToDelete = take
                            }) {
                                model.selectTake(take.id)
                            }
                            .id(take.id)
                            .disabled(model.state.isGenerating)
                            .anchorPreference(key: TakeRows.self, value: .bounds) { [$0] }
                        }
                    }
                    .padding(Self.gridInset)
                    .padding(.trailing, 10)
                    .onGeometryChange(for: CGSize.self, of: \.size) { _ in
                        // A new column count moves every row, so bring the selected take back.
                        if let id = model.visibleSelectedTakeID { scroll.scrollTo(id) }
                    }
                    .backgroundPreferenceValue(TakeRows.self) { tiles in
                        GeometryReader { grid in
                            let height = TakeRows.wholeRows(tiles.map { grid[$0] }, limit: Self.wholeRowsLimit) + 2 * Self.gridInset
                            Color.clear.onChange(of: height, initial: true) { wholeRows = height }
                        }
                    }
                }
                .scrollIndicators(.visible)
                .frame(idealHeight: max(wholeRows, chosenHeight ?? 0), maxHeight: max(wholeRows, chosenHeight ?? 0))
                .onGeometryChange(for: CGFloat.self, of: \.size.height) { gridHeight = $0 }
                .onChange(of: model.visibleSelectedTakeID, initial: true) { _, id in
                    if let id { scroll.scrollTo(id) }
                }
            }
            HStack {
                Text(model.exportSummary).font(.caption).foregroundStyle(.secondary)
                if model.previewingTakeID != nil {
                    Button("Stop Audition", systemImage: "stop.fill") { model.stopPreview() }
                }
                Spacer()
                Button("Save Voice to Library", systemImage: "square.and.arrow.down") {
                    takeToSave = model.visibleTakes.first { $0.id == model.visibleSelectedTakeID }
                }
                .disabled(model.voiceSaveBlocker(forTake: model.visibleSelectedTakeID) != nil)
                .help(model.voiceSaveBlocker(forTake: model.visibleSelectedTakeID) ?? "Save the selected take's voice and use it for future generation")
                Button(isSelectedKept ? "Used for Export" : "Use for Export", systemImage: "checkmark.circle") {
                    model.useSelectedTakeForExport()
                }
                .disabled(model.visibleSelectedTakeID == nil || model.state.isGenerating || isSelectedKept)
                .help("Use the selected take in exported audio and captions")
            }
            if model.visibleSelectedTakeID != nil, let blocker = model.voiceSaveBlocker(forTake: model.visibleSelectedTakeID) {
                Text(blocker).font(.caption).foregroundStyle(.secondary)
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
                    ForEach(Examples.notes(for: model.retakeEngine), id: \.self) { suggestion in
                        Button(suggestion) { note = suggestion }
                            .buttonStyle(.bordered)
                            .buttonBorderShape(.capsule)
                            .controlSize(.small)
                            .fixedSize(horizontal: false, vertical: true)
                    }
            }
            .fixedSize(horizontal: false, vertical: true)
            if let emotions = model.retakeEngine.capabilities.noteEmotions {
                Text("\(model.retakeEngine.capabilities.name) takes one emotion from a note: \(emotions.dropLast().joined(separator: ", ")) or \(emotions.last ?? "").")
                    .font(.caption).foregroundStyle(.secondary)
            }
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
        .overlay(alignment: .top) { resizeHandle }
        .sheet(item: $takeToSave) { take in
            SaveTakeVoiceSheet(model: model, take: take)
        }
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

    private static let gridInset: CGFloat = 2
    /// Room for two rows of tiles that each carry a status line and a one-line
    /// note. Taller rows show one at a time, so long notes leave the editor its space.
    private static let wholeRowsLimit: CGFloat = 160

    /// The strip's top edge, dragged to show more rows, or fewer down to the whole rows the grid starts with.
    private var resizeHandle: some View {
        Color.clear
            .frame(height: 8)
            .contentShape(Rectangle())
            .pointerStyle(.rowResize)
            .gesture(DragGesture(minimumDistance: 1, coordinateSpace: .global)
                .onChanged { drag in
                    let start = dragStart ?? gridHeight
                    dragStart = start
                    choose(start - drag.translation.height)
                }
                .onEnded { _ in
                    dragStart = nil
                    choose(gridHeight)
                })
            .accessibilityElement()
            .accessibilityLabel("Takes height")
            .accessibilityValue("\(Int(gridHeight)) points")
            .accessibilityAdjustableAction { direction in
                let step = wholeRows / 2
                choose(gridHeight + (direction == .increment ? step : -step))
            }
            .help("Drag to show more or fewer takes")
    }

    /// Back at the whole rows the grid starts with, the grid follows its tiles again.
    private func choose(_ height: CGFloat) {
        chosenHeight = height > wholeRows ? height : nil
    }

    private var isSelectedKept: Bool { model.visibleSelectedTakeID != nil && model.visibleSelectedTakeID == model.visibleExportTakeID }

    private func retake() {
        guard model.canRetake else { return }
        model.retake(note: note)
    }
}

/// Tiles in as many equal columns as fit, each row as tall as its tallest
/// tile. Every tile is laid out, in view or not, so `TakeRows` knows every
/// row's height before the grid has scrolled to it.
private struct TakeGrid: Layout {
    private static let minimumWidth: CGFloat = 220
    private static let spacing: CGFloat = 8

    private struct Row {
        var top: CGFloat
        var height: CGFloat
        var tiles: Range<Int>
    }

    private func arrange(_ subviews: Subviews, width: CGFloat) -> (column: CGFloat, rows: [Row]) {
        let columns = max(1, Int((width + Self.spacing) / (Self.minimumWidth + Self.spacing)))
        let column = (width - CGFloat(columns - 1) * Self.spacing) / CGFloat(columns)
        var rows: [Row] = []
        for start in stride(from: 0, to: subviews.count, by: columns) {
            let tiles = start..<min(start + columns, subviews.count)
            let height = tiles.map { subviews[$0].sizeThatFits(ProposedViewSize(width: column, height: nil)).height }.max() ?? 0
            rows.append(Row(top: rows.last.map { $0.top + $0.height + Self.spacing } ?? 0, height: height, tiles: tiles))
        }
        return (column, rows)
    }

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        // SwiftUI proposes an infinite width to ask how wide the grid could be.
        let width = proposal.width.flatMap { $0.isFinite ? $0 : nil } ?? 2 * Self.minimumWidth + Self.spacing
        let rows = arrange(subviews, width: width).rows
        return CGSize(width: width, height: rows.last.map { $0.top + $0.height } ?? 0)
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        let (column, rows) = arrange(subviews, width: bounds.width)
        for row in rows {
            for (position, tile) in row.tiles.enumerated() {
                subviews[tile].place(at: CGPoint(x: bounds.minX + CGFloat(position) * (column + Self.spacing), y: bounds.minY + row.top),
                                     proposal: ProposedViewSize(width: column, height: nil))
            }
        }
    }
}

/// Where the grid put its tiles, for sizing the grid by whole rows.
struct TakeRows: PreferenceKey {
    static let defaultValue: [Anchor<CGRect>] = []
    static func reduce(value: inout [Anchor<CGRect>], nextValue: () -> [Anchor<CGRect>]) { value += nextValue() }

    /// The height that shows any two neighbouring rows of `tiles` whole when
    /// that is within `limit`, and otherwise any one row, up to `limit`. A
    /// grid this tall, scrolled to rest on a row's edge, cuts no tile.
    static func wholeRows(_ tiles: [CGRect], limit: CGFloat) -> CGFloat {
        let rows = Dictionary(grouping: tiles) { $0.minY.rounded() }
            .map { (top: $0.key, bottom: $0.value.map(\.maxY).max() ?? $0.key) }
            .sorted { $0.top < $1.top }
        let one = rows.map { $0.bottom - $0.top }.max() ?? 0
        let two = zip(rows, rows.dropFirst()).map { $1.bottom - $0.top }.max() ?? one
        return (two <= limit ? two : min(one, limit)).rounded(.up)
    }
}

private struct SaveTakeVoiceSheet: View {
    let model: SessionModel
    let take: SessionTake
    @Environment(\.dismiss) private var dismiss
    @State private var name: String
    @State private var error: String?

    init(model: SessionModel, take: SessionTake) {
        self.model = model
        self.take = take
        _name = State(initialValue: "Take \(take.number) voice")
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Save Voice from Take \(take.number)").font(.headline)
            TextField("Voice name", text: $name)
                .textFieldStyle(.roundedBorder)
            Text("The saved voice will be selected for this speaker. You can edit the script and generate again using this voice.")
                .font(.callout).foregroundStyle(.secondary)
            if let error { Text(error).font(.callout).foregroundStyle(.red) }
            HStack {
                Spacer()
                Button("Cancel", role: .cancel) { dismiss() }
                    .keyboardShortcut(.cancelAction)
                Button("Save Voice") {
                    if model.saveVoice(fromTake: take.id, name: name) { dismiss() }
                    else { error = model.notice }
                }
                .keyboardShortcut(.defaultAction)
                .disabled(name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
        }
        .padding(24)
        .frame(width: 440)
    }
}

private struct TakeCard: View {
    let take: SessionTake
    /// The engine that made the take, when naming it tells takes apart.
    let engine: String?
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
                    if let engine { Text(engine).font(.caption).foregroundStyle(.secondary) }
                    Text(TransportBar.seconds(take.duration))
                        .font(.callout.monospacedDigit()).foregroundStyle(.secondary)
                }
                .padding(.trailing, 26)
                if isKept || isOutdated || !take.hasCurrentWordCheck || !take.notices.isEmpty {
                    HStack(spacing: 8) {
                        if isKept { Label("Export", systemImage: "checkmark.circle.fill") }
                        if isOutdated { Label("Outdated", systemImage: "clock.badge.exclamationmark") }
                        if !take.notices.isEmpty {
                            Label("\(take.notices.count) \(take.notices.count == 1 ? "notice" : "notices")", systemImage: "exclamationmark.triangle")
                                .help(take.notices.joined(separator: "\n"))
                        }
                        if !take.hasCurrentWordCheck {
                            Text(take.isIncomplete ? "Incomplete" : (take.wordsChecked ? "Earlier word check" : "Unchecked"))
                                .help(take.isIncomplete ? "Generation ended before the whole script was completed." : "This take has not passed the current spoken-word checks. Generate a new take for checked audio.")
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
        .accessibilityLabel("\(take.label)\(engine.map { ", \($0)" } ?? "")\(isKept ? ", used for export" : "")\(isOutdated ? ", outdated" : "")\(take.isIncomplete ? ", incomplete" : (take.hasCurrentWordCheck ? "" : (take.wordsChecked ? ", earlier word check" : ", unchecked")))\(take.notices.isEmpty ? "" : ". " + take.notices.joined(separator: " "))")
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

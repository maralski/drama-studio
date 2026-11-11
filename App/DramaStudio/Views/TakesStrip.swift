// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import DramaEngine
import SwiftUI

struct TakesStrip: View {
    let model: SessionModel
    @State private var note = ""
    @State private var takeToDelete: SessionTake?
    @State private var takeToSave: SessionTake?
    /// The most the strip may be tall, which leaves the editor above it its floor. Nil is no limit.
    var room: CGFloat?
    @State private var rows = TakeRows()
    /// The row the grid rests on: its top edge is the top of the grid.
    @State private var topRow = 0
    /// The grid height the user dragged to; nil shows up to `wholeRowsLimit` of whole rows.
    @State private var chosenHeight: CGFloat?
    /// The grid height under the pointer while the top edge is dragged.
    @State private var draggedHeight: CGFloat?
    @State private var dragStart: CGFloat?
    @State private var gridHeight: CGFloat = 0
    /// The strip's height apart from the grid.
    @State private var chrome: CGFloat = 0
    @State private var position = ScrollPosition()
    @State private var scroll = TakeScroll()

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
            ScrollView(.vertical) {
                TakeTiles(model: model, takeToDelete: $takeToDelete)
                .padding(Self.gridInset)
                .padding(.trailing, 10)
                .backgroundPreferenceValue(TakeTilePlaces.self) { tiles in
                    GeometryReader { grid in
                        let placed = TakeRows(tiles: tiles.map { grid[$0] }, inset: Self.gridInset)
                        Color.clear.onChange(of: placed, initial: true) { rows = placed }
                    }
                }
                // Room under the last rows, so the grid can scroll them to its top edge
                // before it shrinks to fit them.
                .padding(.bottom, max(0, rows.lastRest(within: restBudget) + height - rows.contentHeight))
            }
            .scrollIndicators(.visible)
            .scrollPosition($position)
            .scrollTargetBehavior(TakeRowRests(rows: rows, budget: restBudget))
            .onScrollGeometryChange(for: CGFloat.self, of: \.contentOffset.y) { _, offset in
                scroll.offset = offset
                if !scroll.isMoving { scroll.settle(after: .milliseconds(200)) { rest(nearest: offset, animated: true) } }
            }
            .onScrollPhaseChange { _, phase, context in
                scroll.isMoving = phase != .idle
                if phase == .idle { rest(nearest: context.geometry.contentOffset.y, animated: true) }
            }
            .frame(idealHeight: height, maxHeight: height)
            .onGeometryChange(for: [CGFloat].self, of: { grid in
                [grid.size.height, (grid.bounds(of: .named(Self.strip))?.height ?? grid.size.height) - grid.size.height]
            }) {
                gridHeight = $0[0]
                chrome = $0[1]
            }
            .onChange(of: rows) { old, new in arrange(reflowed: old.columns != new.columns) }
            .onChange(of: model.visibleSelectedTakeID, initial: true) { arrange() }
            .onChange(of: budget) { arrange() }
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
        .coordinateSpace(.named(Self.strip))
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
    private static let strip = "TakesStrip"
    /// Room for two rows of tiles that each carry a status line and a one-line
    /// note. Taller rows show one at a time, so long notes leave the editor its space.
    private static let wholeRowsLimit: CGFloat = 160

    /// The most the grid may be tall.
    private var gridRoom: CGFloat { room.map { max($0 - chrome, 0).rounded(.down) } ?? .infinity }
    private var automaticBudget: CGFloat { min(Self.wholeRowsLimit, gridRoom) }
    /// The height the grid fills with whole rows.
    private var budget: CGFloat { min(chosenHeight ?? Self.wholeRowsLimit, gridRoom) }
    /// The height that decides where scrolling rests: under the pointer during a drag, otherwise the budget.
    private var restBudget: CGFloat { draggedHeight ?? budget }
    /// The whole rows from the resting row that the budget holds. One plain row until the first layout.
    private var height: CGFloat { draggedHeight ?? (rows.isEmpty ? 68 : rows.height(from: topRow, within: budget)) }

    /// Rests the grid on a row from which the selected take shows whole when the
    /// selection is new or the columns changed, and otherwise on the row it was on.
    private func arrange(reflowed: Bool = false) {
        guard !rows.isEmpty, draggedHeight == nil else { return }
        var top = min(topRow, rows.lastStop(within: budget))
        let selected = model.visibleSelectedTakeID
        if reflowed || selected != scroll.shownTake,
           let row = model.visibleTakes.firstIndex(where: { $0.id == selected }).flatMap(rows.row(ofTile:)) {
            top = rows.top(showing: row, from: top, within: budget)
            scroll.shownTake = selected
        }
        if topRow != top { topRow = top }
        scroll(to: rows.offset(of: top, near: scroll.offset, within: budget), animated: false)
    }

    /// Rests the grid on the row nearest `offset`, where free scrolling left it.
    private func rest(nearest offset: CGFloat, animated: Bool) {
        guard !rows.isEmpty, draggedHeight == nil else { return }
        let rest = rows.rest(for: offset, from: offset, within: budget)
        let row = rows.row(restingAt: rest)
        if topRow != row { topRow = row }
        scroll(to: rest, animated: animated)
    }

    private func scroll(to offset: CGFloat, animated: Bool) {
        scroll.cancelSettle()
        guard abs(offset - scroll.offset) > 0.5 else { return }
        // In a task, because a scroll asked for inside the update that changes the grid's rows or height is dropped.
        Task { withAnimation(animated ? .snappy(duration: 0.2) : nil) { position.scrollTo(y: offset) } }
    }

    /// The strip's top edge, dragged to show more rows, or fewer down to the whole rows the grid starts with.
    private var resizeHandle: some View {
        Color.clear
            .frame(height: 8)
            .contentShape(Rectangle())
            .pointerStyle(.rowResize)
            .gesture(DragGesture(minimumDistance: 1, coordinateSpace: .global)
                .onChanged { drag in
                    let start = dragStart ?? height
                    dragStart = start
                    let least = rows.height(from: topRow, within: automaticBudget)
                    draggedHeight = min(max(start - drag.translation.height, least), max(gridRoom, least))
                }
                .onEnded { _ in
                    dragStart = nil
                    if let draggedHeight { choose(draggedHeight) }
                })
            .accessibilityElement()
            .accessibilityLabel("Takes height")
            .accessibilityValue("\(Int(gridHeight)) points")
            .accessibilityAdjustableAction { direction in
                let heights = rows.wholeHeights(around: topRow)
                let next = direction == .increment ? heights.first { $0 > height + 0.5 } : heights.last { $0 < height - 0.5 }
                if let next { choose(next) }
            }
            .help("Drag to show more or fewer takes")
    }

    /// Settles a height chosen by hand on the nearest whole rows around the
    /// resting row. Back at the rows the grid starts with, it follows its tiles again.
    private func choose(_ wanted: CGFloat) {
        draggedHeight = nil
        guard !rows.isEmpty else { return }
        let top = rows.row(restingAt: rows.rest(for: scroll.offset, from: scroll.offset, within: wanted))
        let automatic = rows.height(from: top, within: automaticBudget)
        let settled = rows.wholeHeights(around: top).filter { $0 <= gridRoom + 0.5 }
            .min { abs($0 - wanted) < abs($1 - wanted) } ?? automatic
        chosenHeight = settled > automatic + 0.5 ? settled : nil
        if topRow != top { topRow = top }
        arrange()
    }

    private var isSelectedKept: Bool { model.visibleSelectedTakeID != nil && model.visibleSelectedTakeID == model.visibleExportTakeID }

    private func retake() {
        guard model.canRetake else { return }
        model.retake(note: note)
    }
}

/// The takes as tiles in their grid. A view of its own, so that a change in
/// the strip around it, such as the grid's height, lays no tile out again.
private struct TakeTiles: View {
    let model: SessionModel
    @Binding var takeToDelete: SessionTake?

    var body: some View {
        TakeGrid {
            ForEach(model.visibleTakes) { take in
                TakeCard(take: take, engine: model.engineLabel(for: take), isSelected: take.id == model.visibleSelectedTakeID, isKept: take.id == model.visibleExportTakeID, isOutdated: model.isOutdated(take, in: model.selectedSegment), delete: {
                    takeToDelete = take
                }) {
                    model.selectTake(take.id)
                }
                .disabled(model.state.isGenerating)
                .anchorPreference(key: TakeTilePlaces.self, value: .bounds) { [$0] }
            }
        }
    }
}

/// Tiles in as many equal columns as fit, each row as tall as its tallest
/// tile. Every tile is laid out, in view or not, so `TakeRows` knows every
/// row's height before the grid has scrolled to it.
struct TakeGrid: Layout {
    private static let minimumWidth: CGFloat = 220
    private static let spacing: CGFloat = 8

    struct Row {
        var top: CGFloat
        var height: CGFloat
        var tiles: Range<Int>
    }

    struct Arrangement {
        var column: CGFloat
        var rows: [Row]
    }

    /// Arrangements by grid width. SwiftUI asks for the grid's size and then
    /// places it in one pass, often at one width, and every tile would be
    /// measured for each. A new pass starts with an empty cache, because the
    /// tiles or their contents may have changed since the last.
    typealias Cache = [CGFloat: Arrangement]

    static func columns(fitting width: CGFloat) -> Int {
        max(1, Int((width + spacing) / (minimumWidth + spacing)))
    }

    func makeCache(subviews: Subviews) -> Cache { [:] }

    func updateCache(_ cache: inout Cache, subviews: Subviews) { cache = [:] }

    private func arrange(_ subviews: Subviews, width: CGFloat, cache: inout Cache) -> Arrangement {
        if let arranged = cache[width] { return arranged }
        let columns = Self.columns(fitting: width)
        let column = (width - CGFloat(columns - 1) * Self.spacing) / CGFloat(columns)
        var rows: [Row] = []
        for start in stride(from: 0, to: subviews.count, by: columns) {
            let tiles = start..<min(start + columns, subviews.count)
            let height = tiles.map { subviews[$0].sizeThatFits(ProposedViewSize(width: column, height: nil)).height }.max() ?? 0
            rows.append(Row(top: rows.last.map { $0.top + $0.height + Self.spacing } ?? 0, height: height, tiles: tiles))
        }
        let arranged = Arrangement(column: column, rows: rows)
        cache[width] = arranged
        return arranged
    }

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout Cache) -> CGSize {
        // SwiftUI proposes an infinite width to ask how wide the grid could be.
        let width = proposal.width.flatMap { $0.isFinite ? $0 : nil } ?? 2 * Self.minimumWidth + Self.spacing
        let rows = arrange(subviews, width: width, cache: &cache).rows
        return CGSize(width: width, height: rows.last.map { $0.top + $0.height } ?? 0)
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout Cache) {
        let arranged = arrange(subviews, width: bounds.width, cache: &cache)
        for row in arranged.rows {
            for (position, tile) in row.tiles.enumerated() {
                subviews[tile].place(at: CGPoint(x: bounds.minX + CGFloat(position) * (arranged.column + Self.spacing), y: bounds.minY + row.top),
                                     proposal: ProposedViewSize(width: arranged.column, height: nil))
            }
        }
    }
}

/// Where the grid put its tiles.
struct TakeTilePlaces: PreferenceKey {
    static let defaultValue: [Anchor<CGRect>] = []
    static func reduce(value: inout [Anchor<CGRect>], nextValue: () -> [Anchor<CGRect>]) { value += nextValue() }
}

/// The grid's rows as it placed them, in the scroll content's coordinates,
/// each reaching to the grid's inset above and below its tiles. The grid's
/// height, where scrolling rests and where a selected take shows all read
/// rows from here.
struct TakeRows: Equatable {
    struct Row: Equatable {
        var top: CGFloat
        var bottom: CGFloat
        var tiles: Range<Int>
        var height: CGFloat { bottom - top }
    }

    private(set) var rows: [Row] = []
    /// Slack for heights that differ by rounding.
    private static let slack: CGFloat = 0.5

    init() {}

    /// Rows from tile frames in the order the grid holds them.
    init(tiles: [CGRect], inset: CGFloat) {
        for (index, tile) in tiles.enumerated() {
            if let last = rows.indices.last, abs(rows[last].top + inset - tile.minY) < Self.slack {
                rows[last].bottom = max(rows[last].bottom, tile.maxY + inset)
                rows[last].tiles = rows[last].tiles.lowerBound..<index + 1
            } else {
                rows.append(Row(top: tile.minY - inset, bottom: tile.maxY + inset, tiles: index..<index + 1))
            }
        }
    }

    var isEmpty: Bool { rows.isEmpty }
    var columns: Int { rows.first?.tiles.count ?? 0 }
    var contentHeight: CGFloat { rows.last?.bottom ?? 0 }

    func row(ofTile index: Int) -> Int? { rows.firstIndex { $0.tiles.contains(index) } }

    /// The height of the whole rows from `top` down that fit `budget`: one row
    /// at least, and `budget` itself for a row taller than it, which scrolls.
    func height(from top: Int, within budget: CGFloat) -> CGFloat {
        guard rows.indices.contains(top) else { return 0 }
        let last = rows[top...].lastIndex { $0.bottom - rows[top].top <= budget + Self.slack }
        return last.map { rows[$0].bottom - rows[top].top } ?? min(rows[top].height, budget)
    }

    /// The first row from which every row to the end fits `budget`. The grid
    /// rests no lower than this, so its last rows show whole with nothing under them.
    func lastStop(within budget: CGFloat) -> Int {
        rows.firstIndex { contentHeight - $0.top <= budget + Self.slack } ?? max(rows.count - 1, 0)
    }

    /// The scroll offsets the grid may rest at: each row's top edge down to the
    /// last stop, and anywhere that keeps a row taller than `budget` filling the grid.
    private func rests(within budget: CGFloat) -> [ClosedRange<CGFloat>] {
        guard !rows.isEmpty else { return [0...0] }
        return rows[...lastStop(within: budget)].map { $0.top...max($0.top, $0.bottom - budget) }
    }

    /// The lowest offset the grid may rest at.
    func lastRest(within budget: CGFloat) -> CGFloat { rests(within: budget).last?.upperBound ?? 0 }

    /// Where a scroll from `origin` that would stop at `target` rests instead:
    /// the rest nearest `target`, and at least one rest on from `origin`, so
    /// that a single slow turn of a wheel moves a row.
    func rest(for target: CGFloat, from origin: CGFloat, within budget: CGFloat) -> CGFloat {
        let rests = rests(within: budget)
        func into(_ rest: ClosedRange<CGFloat>) -> CGFloat { min(max(target, rest.lowerBound), rest.upperBound) }
        let nearest = rests.map(into).min { abs($0 - target) < abs($1 - target) } ?? 0
        if target > origin + Self.slack, let next = rests.first(where: { $0.upperBound > origin + Self.slack }) {
            return max(nearest, into(next))
        }
        if target < origin - Self.slack, let next = rests.last(where: { $0.lowerBound < origin - Self.slack }) {
            return min(nearest, into(next))
        }
        return nearest
    }

    /// The row whose top edge is at or last above `offset`.
    func row(restingAt offset: CGFloat) -> Int {
        max((rows.firstIndex { $0.top > offset + Self.slack } ?? rows.count) - 1, 0)
    }

    /// The offset that rests the grid on `row`: its top edge, or `current` when
    /// that already rests inside a row taller than `budget`.
    func offset(of row: Int, near current: CGFloat, within budget: CGFloat) -> CGFloat {
        guard rows.indices.contains(row) else { return 0 }
        return min(max(current, rows[row].top), max(rows[row].top, rows[row].bottom - budget))
    }

    /// The row to rest on so that `row` shows whole, moving from `top` no further than that takes.
    func top(showing row: Int, from top: Int, within budget: CGFloat) -> Int {
        guard rows.indices.contains(row), row >= top else { return max(min(row, rows.count - 1), 0) }
        let first = rows[top...row].firstIndex { rows[row].bottom - $0.top <= budget + Self.slack } ?? row
        return min(first, lastStop(within: budget))
    }

    /// Every height that shows whole rows with `top` among them, shortest
    /// first: more rows below `top`, then, past the last row, more above it.
    func wholeHeights(around top: Int) -> [CGFloat] {
        guard rows.indices.contains(top) else { return [] }
        return rows[top...].map { $0.bottom - rows[top].top } + rows[..<top].reversed().map { contentHeight - $0.top }
    }
}

/// Rests free scrolling on a row's top edge.
struct TakeRowRests: ScrollTargetBehavior {
    var rows: TakeRows
    var budget: CGFloat

    func updateTarget(_ target: inout ScrollTarget, context: TargetContext) {
        target.rect.origin.y = rows.rest(for: target.rect.minY, from: context.originalTarget.rect.minY, within: budget)
    }
}

/// Where the grid is scrolled, kept out of view state so that scrolling
/// redraws nothing.
@MainActor
final class TakeScroll {
    var offset: CGFloat = 0
    /// The take the grid last scrolled to show.
    var shownTake: SessionTake.ID?
    /// A wheel, trackpad or animation is moving the grid. A dragged scroller is not reported.
    var isMoving = false
    private var settling: Task<Void, Never>?

    /// Runs `action` once the grid has stayed still for `delay`.
    func settle(after delay: Duration, _ action: @escaping @MainActor () -> Void) {
        settling?.cancel()
        settling = Task {
            try? await Task.sleep(for: delay)
            if !Task.isCancelled { action() }
        }
    }

    func cancelSettle() { settling?.cancel() }
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

import DramaEngine
import SwiftUI

struct TakesStrip: View {
    let model: SessionModel
    @State private var note = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            ScrollView(.horizontal) {
                HStack(spacing: 6) {
                    ForEach(model.takes) { take in
                        TakeChip(take: take, isSelected: take.id == model.selectedTakeID, isKept: take.id == model.keptTakeID) {
                            model.selectTake(take.id)
                        }
                    }
                }
                .padding(.vertical, 1)
            }
            .scrollIndicators(.never)
            .defaultScrollAnchor(.trailing)
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
                Button(isSelectedKept ? "Kept" : "Keep", systemImage: isSelectedKept ? "star.fill" : "star") { model.toggleKeepSelectedTake() }
                    .disabled(model.selectedTakeID == nil || model.state.isGenerating)
                    .help("Mark the selected take as the one Export writes")
                HelpButton(.keepTake)
            }
            ScrollView(.horizontal) {
                HStack(spacing: 6) {
                    ForEach(Examples.notes, id: \.self) { suggestion in
                        Button(suggestion) { note = suggestion }
                            .buttonStyle(.bordered)
                            .buttonBorderShape(.capsule)
                            .controlSize(.small)
                    }
                }
            }
            .scrollIndicators(.never)
            if let blocker = model.retakeBlocker {
                Text(blocker).font(.caption).foregroundStyle(.secondary)
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 8)
        .background(.bar)
        .overlay(alignment: .top) { Divider() }
    }

    private var isSelectedKept: Bool { model.selectedTakeID != nil && model.selectedTakeID == model.keptTakeID }

    private func retake() {
        guard model.canRetake else { return }
        model.retake(note: note)
    }
}

private struct TakeChip: View {
    let take: SessionTake
    let isSelected: Bool
    let isKept: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 4) {
                if isKept { Image(systemName: "star.fill").foregroundStyle(.yellow) }
                Text(take.label).lineLimit(1)
            }
        }
        .buttonStyle(.bordered)
        .buttonBorderShape(.capsule)
        .tint(isSelected ? .accentColor : nil)
        .help(isKept ? "The kept take, which Export writes. Click to play." : "Click to play this take")
    }
}

extension SessionTake {
    var label: String {
        (["Take \(number)", TransportBar.seconds(duration)] + [note].compactMap { $0 }).joined(separator: " \u{00B7} ")
    }
}

import DramaEngine
import SwiftUI

/// A document's sections: pick one to edit it, skip it, give it a speaker,
/// or narrate it again.
struct SectionsView: View {
    let model: SessionModel
    let document: NarrationDocument
    let playingID: NarrationDocument.Segment.ID?

    var body: some View {
        VStack(spacing: 0) {
            List(selection: Binding(get: { model.selectedSegmentID }, set: { model.selectSegment($0) })) {
                ForEach(document.segments) { segment in
                    SectionRow(model: model, segment: segment, isPlaying: segment.id == playingID)
                        .tag(segment.id)
                }
            }
            .listStyle(.inset)
            Divider()
            HStack {
                Button("Narrate This Section", systemImage: "waveform.badge.plus") { model.narrateSelectedSection() }
                    .disabled(!model.canNarrateSection)
                    .help("Generate the selected section again; the other sections keep their takes")
                HelpButton(.sections)
                Spacer()
                Button("Flatten to Script", systemImage: "text.alignleft") { model.flattenToScript() }
                    .disabled(model.state.isGenerating)
                    .help("Replace the sections with one plain script of their text")
            }
            .controlSize(.small)
            .padding(.horizontal, 10)
            .padding(.vertical, 8)
        }
    }
}

private struct SectionRow: View {
    let model: SessionModel
    let segment: NarrationDocument.Segment
    let isPlaying: Bool

    var body: some View {
        HStack(alignment: .top, spacing: 8) {
            Toggle("Narrate", isOn: Binding(get: { !segment.skip }, set: { model.setSkip(!$0, for: segment.id) }))
                .labelsHidden()
                .toggleStyle(.checkbox)
                .help(segment.skip ? "Skipped. Check to narrate this section." : "Narrated. Uncheck to skip this section.")
            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 6) {
                    if isPlaying { Image(systemName: "speaker.wave.2.fill").foregroundStyle(Color.accentColor) }
                    Text(segment.label).lineLimit(1).fontWeight(isPlaying ? .semibold : .regular)
                }
                Text(preview).lineLimit(1).font(.caption).foregroundStyle(.secondary)
                HStack(spacing: 8) {
                    if let take = model.chosenTake(for: segment.id) {
                        Text(TransportBar.seconds(take.duration)).font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                    } else {
                        Text("Not narrated").font(.caption).foregroundStyle(.tertiary)
                    }
                    Picker("Speaker", selection: Binding(get: { segment.speakerTag ?? 0 }, set: { model.setSpeaker($0, for: segment.id) })) {
                        ForEach(0...ModelConstants.maxSpeakerID, id: \.self) { Text(SectionRequests.speakerName($0)).tag($0) }
                    }
                    .labelsHidden()
                    .controlSize(.mini)
                    .fixedSize()
                }
            }
        }
        .padding(.vertical, 2)
        .opacity(segment.skip ? 0.55 : 1)
        .disabled(model.state.isGenerating)
    }

    private var preview: String {
        let text = SectionRequests.spokenText(of: segment)
        return text.isEmpty ? "No text" : text.replacingOccurrences(of: #"\s+"#, with: " ", options: .regularExpression)
    }
}

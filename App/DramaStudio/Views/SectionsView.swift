// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
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
            VStack(alignment: .leading, spacing: 8) {
                Button("Generate Changes", systemImage: "arrow.clockwise") { model.generateChangedSections() }
                    .disabled(!model.canGenerateChangedSections)
                    .help("Generate changed sections, keeping the other sections' takes")
                Button("Narrate This Section", systemImage: "waveform.badge.plus") { model.narrateSelectedSection() }
                    .disabled(!model.canNarrateSection)
                    .help("Generate the selected section again; the other sections keep their takes")
                HStack {
                    Button("Flatten to Script", systemImage: "text.alignleft") { model.flattenToScript() }
                        .disabled(model.state.isGenerating)
                        .help("Replace the sections with one plain script of their text")
                    HelpButton(.sections)
                }
            }
            .controlSize(.small)
            .padding(.horizontal, 16)
            .padding(.top, 12)
            .padding(.bottom, 16)
            .fixedSize(horizontal: false, vertical: true)
            .layoutPriority(1)
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
                .help(segment.skip ? "Skipped. Check to include this section." : "Included. Uncheck to skip this section.")
            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 6) {
                    if isPlaying { Image(systemName: "speaker.wave.2.fill").foregroundStyle(Color.accentColor) }
                    Text(segment.label).lineLimit(1).fontWeight(isPlaying ? .semibold : .regular)
                }
                Text(preview).lineLimit(1).font(.caption).foregroundStyle(.secondary)
                ViewThatFits(in: .horizontal) {
                    HStack(spacing: 8) {
                        status.fixedSize()
                        speakerPicker
                    }
                    VStack(alignment: .leading, spacing: 4) {
                        status
                        speakerPicker
                    }
                }
            }
        }
        .padding(.vertical, 2)
        .opacity(segment.skip ? 0.55 : 1)
        .disabled(model.state.isGenerating)
    }

    private var status: some View {
        HStack(spacing: 6) {
            Text(model.sectionStatus(segment))
                .foregroundStyle(model.sectionStatus(segment) == "Needs regeneration" ? Color.orange : Color.secondary)
            if let take = model.chosenTake(for: segment.id) {
                Text(TransportBar.seconds(take.duration)).monospacedDigit().foregroundStyle(.secondary)
            }
        }
        .font(.caption)
    }

    private var speakerPicker: some View {
        Picker("Speaker", selection: Binding(get: { segment.speakerTag ?? 0 }, set: { model.setSpeaker($0, for: segment.id) })) {
            ForEach(0...ModelConstants.maxSpeakerID, id: \.self) { Text(SectionRequests.speakerName($0)).tag($0) }
        }
        .labelsHidden()
        .controlSize(.mini)
        .fixedSize()
    }

    private var preview: String {
        let text = SectionRequests.spokenText(of: segment)
        return text.isEmpty ? "No text" : text.replacingOccurrences(of: #"\s+"#, with: " ", options: .regularExpression)
    }
}

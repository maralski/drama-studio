// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import DramaEngine
import SwiftUI

struct VoiceLibraryView: View {
    let model: SessionModel
    @Binding var newVoice: NewVoiceRequest?
    @State private var selection: ReferenceVoice.ID?
    @State private var pendingRemoval: ReferenceVoice?
    @State private var isDropTarget = false

    var body: some View {
        List(selection: $selection) {
            Section("Voices") {
                ForEach(model.library.voices) { voice in
                    VoiceRow(voice: voice, isPreviewing: model.previewingVoice == voice.id) { model.togglePreview(voice) }
                        .tag(voice.id)
                        .contextMenu {
                            Button("Preview") { model.togglePreview(voice) }
                            Button("Remove\u{2026}", role: .destructive) { pendingRemoval = voice }
                        }
                }
            }
            if !model.library.unreadableFiles.isEmpty {
                Label("Couldn't read \(model.library.unreadableFiles.joined(separator: ", "))", systemImage: "exclamationmark.triangle.fill")
                    .font(.caption)
                    .foregroundStyle(.orange)
            }
        }
        .overlay {
            if model.library.voices.isEmpty {
                ContentUnavailableView("No Voices", systemImage: "person.wave.2", description: Text("Add or drop a reference clip to clone a voice."))
            }
        }
        .overlay {
            if isDropTarget { RoundedRectangle(cornerRadius: 8).strokeBorder(Color.accentColor, lineWidth: 3).padding(4) }
        }
        .dropDestination(for: URL.self) { urls, _ in
            newVoice = NewVoiceRequest(clip: urls.first)
            return urls.first != nil
        } isTargeted: { isDropTarget = $0 }
        .safeAreaInset(edge: .bottom) { footer }
        .confirmationDialog("Remove \u{201C}\(pendingRemoval?.name ?? "")\u{201D}?", isPresented: Binding(get: { pendingRemoval != nil }, set: { if !$0 { pendingRemoval = nil } })) {
            Button("Remove Voice", role: .destructive) {
                if let pendingRemoval { model.removeVoice(pendingRemoval.id) }
            }
        } message: {
            Text("The voice's codes are deleted from your library. The original clip is not touched.")
        }
    }

    private var footer: some View {
        HStack(spacing: 4) {
            Button("Add Voice", systemImage: "plus") { newVoice = NewVoiceRequest() }
                .help("Add a reference voice")
            Button("Remove Voice", systemImage: "minus") { pendingRemoval = model.library.voice(selection) }
                .disabled(selection == nil)
                .help("Remove the selected voice")
            Spacer()
        }
        .labelStyle(.iconOnly)
        .buttonStyle(.borderless)
        .padding(8)
    }
}

private struct VoiceRow: View {
    let voice: ReferenceVoice
    let isPreviewing: Bool
    let preview: () -> Void

    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                Text(voice.name).font(.body.weight(.medium))
                Text("\(voice.durationSeconds, format: .number.precision(.fractionLength(1))) s \u{00B7} \(voice.frameCount) frames")
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
                Text(voice.transcript)
                    .font(.caption)
                    .foregroundStyle(.tertiary)
                    .lineLimit(1)
            }
            Spacer()
            Button(isPreviewing ? "Stop Preview" : "Preview", systemImage: isPreviewing ? "stop.circle" : "play.circle", action: preview)
                .labelStyle(.iconOnly)
                .buttonStyle(.borderless)
                .help("Preview the decoded reference")
        }
        .padding(.vertical, 2)
    }
}

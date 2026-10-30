// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import AVFoundation
import SwiftUI

struct AddVoiceSheet: View {
    let model: SessionModel
    @State var clip: URL?
    @Environment(\.dismiss) private var dismiss
    @State private var clipSeconds: Double?
    @State private var name = ""
    @State private var transcript = ""
    @State private var hasConsent = false
    @State private var isEncoding = false
    @State private var error: String?
    @State private var choosingClip = false
    @State private var isShowingRecorder = false
    @State private var recorder = VoiceRecorder()
    @State private var assignedSpeaker = -1
    @State private var encodingTask: Task<Void, Never>?

    private var canAdd: Bool {
        clipSeconds != nil && hasConsent && !isEncoding
            && !name.trimmingCharacters(in: .whitespaces).isEmpty
            && !transcript.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Form {
                Section {
                    clipRow
                    if let clipSeconds, let warning = RecordingLength.warning(for: clipSeconds) {
                        Label(warning, systemImage: "exclamationmark.triangle.fill").font(.caption).foregroundStyle(.orange)
                    }
                    if isShowingRecorder {
                        RecorderPanel(recorder: recorder) { url in
                            clip = url
                            isShowingRecorder = false
                        }
                    }
                    TextField("Name", text: $name, prompt: Text("Narrator, Alice, \u{2026}"))
                    VStack(alignment: .leading, spacing: 8) {
                        HelpLabel("Transcript", help: .voiceTranscript)
                        TextEditor(text: $transcript)
                            .font(.body)
                            .frame(minHeight: 88)
                            .accessibilityLabel("Transcript")
                    }
                } footer: {
                    Text("Type exactly what is said in the clip. 5 to 20 seconds of clean speech works best.")
                }
                if case .success(let speakers) = model.castSpeakers {
                    Section {
                        Picker("Assign to", selection: $assignedSpeaker) {
                            Text("Library only").tag(-1)
                            ForEach(speakers) { Text($0.name).tag($0.index) }
                        }
                    }
                }
                Section {
                    Toggle(isOn: $hasConsent) {
                        HelpLabel("I own this voice or have the speaker's consent to clone it", help: .voiceConsent)
                    }
                    .accessibilityLabel("I own this voice or have consent to clone it")
                    Text("The Fish Audio Community License forbids generating speech that imitates an identifiable person without their consent.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                if let error {
                    Label(error, systemImage: "exclamationmark.triangle.fill").foregroundStyle(.red)
                }
            }
            .formStyle(.grouped)
            .disabled(isEncoding)
            if !canAdd && !isEncoding {
                Text(missingRequirement).font(.caption).foregroundStyle(.secondary).padding(.horizontal, 20).padding(.bottom, 8)
            }
            buttons
        }
        .frame(width: 480, height: 560)
        .fileImporter(isPresented: $choosingClip, allowedContentTypes: [.audio]) { result in
            if case .success(let url) = result { clip = url }
        }
        .dropDestination(for: URL.self) { urls, _ in
            clip = urls.first
            return clip != nil
        }
        .task(id: clip) { inspectClip() }
        .onAppear {
            if case .success(let speakers) = model.castSpeakers, speakers.count == 1 { assignedSpeaker = speakers[0].index }
        }
        .onDisappear { encodingTask?.cancel(); recorder.discard() }
    }

    private var missingRequirement: String {
        if clipSeconds == nil { return "Choose or record an audio clip to continue." }
        if name.trimmingCharacters(in: .whitespaces).isEmpty { return "Enter a name for this voice." }
        if transcript.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { return "Enter the words spoken in the clip." }
        return "Confirm that you own the voice or have permission to clone it."
    }

    private var clipRow: some View {
        LabeledContent {
            HStack {
                if let clip {
                    VStack(alignment: .trailing) {
                        Text(clip.lastPathComponent).lineLimit(1).truncationMode(.middle)
                        if let clipSeconds {
                            Text("\(clipSeconds, format: .number.precision(.fractionLength(1))) s").font(.caption).foregroundStyle(.secondary)
                        }
                    }
                } else {
                    Text("Drop an audio file, choose one, or record").foregroundStyle(.secondary)
                }
                Button("Choose\u{2026}") { choosingClip = true }
                Button("Record\u{2026}", systemImage: "mic") { isShowingRecorder.toggle() }
                    .disabled(recorder.isRecording)
            }
        } label: {
            HelpLabel("Reference clip", help: .referenceClip)
        }
    }

    private var buttons: some View {
        HStack {
            if isEncoding {
                ProgressView().controlSize(.small)
                Text("Encoding reference\u{2026}").foregroundStyle(.secondary)
            }
            Spacer()
            Button("Cancel", role: .cancel) { encodingTask?.cancel(); dismiss() }
                .keyboardShortcut(.cancelAction)
            Button("Add Voice") { add() }
                .keyboardShortcut(.defaultAction)
                .disabled(!canAdd)
        }
        .dialogActions()
    }

    private func inspectClip() {
        clipSeconds = nil
        error = nil
        guard let clip else { return }
        do {
            let file = try AVAudioFile(forReading: clip)
            clipSeconds = Double(file.length) / file.processingFormat.sampleRate
            if name.isEmpty { name = clip.deletingPathExtension().lastPathComponent }
        } catch {
            self.error = "\(clip.lastPathComponent) isn't an audio file this Mac can read."
        }
    }

    private func add() {
        guard let clip else { return }
        isEncoding = true
        error = nil
        encodingTask = Task {
            defer { isEncoding = false }
            do {
                let voice = try await model.encodeVoice(clip: clip, transcript: transcript.trimmingCharacters(in: .whitespacesAndNewlines), name: name.trimmingCharacters(in: .whitespaces))
                if assignedSpeaker >= 0 { model.setVoice(.library(voice.id), forSpeaker: assignedSpeaker) }
                dismiss()
            } catch is CancellationError {
                return
            } catch {
                self.error = "Couldn't add the voice: \(SessionModel.describe(error))"
            }
        }
    }
}

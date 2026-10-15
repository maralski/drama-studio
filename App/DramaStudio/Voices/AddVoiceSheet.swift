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
                    LabeledContent {
                        TextEditor(text: $transcript)
                            .font(.body)
                            .frame(minHeight: 72)
                    } label: {
                        HelpLabel("Transcript", help: .voiceTranscript)
                    }
                } footer: {
                    Text("Type exactly what is said in the clip. 5 to 20 seconds of clean speech works best.")
                }
                Section {
                    Toggle(isOn: $hasConsent) {
                        HelpLabel("I own this voice or have the speaker's consent to clone it", help: .voiceConsent)
                    }
                    Text("The Fish Audio Community License forbids generating speech that imitates an identifiable person without their consent.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                if let error {
                    Label(error, systemImage: "exclamationmark.triangle.fill").foregroundStyle(.red)
                }
            }
            .formStyle(.grouped)
            buttons
        }
        .frame(width: 480)
        .fileImporter(isPresented: $choosingClip, allowedContentTypes: [.audio]) { result in
            if case .success(let url) = result { clip = url }
        }
        .dropDestination(for: URL.self) { urls, _ in
            clip = urls.first
            return clip != nil
        }
        .task(id: clip) { inspectClip() }
        .onDisappear { recorder.discard() }
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
            Button("Cancel", role: .cancel) { dismiss() }
                .keyboardShortcut(.cancelAction)
            Button("Add Voice") { add() }
                .keyboardShortcut(.defaultAction)
                .disabled(!canAdd)
        }
        .padding([.horizontal, .bottom], 20)
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
        Task {
            defer { isEncoding = false }
            do {
                try await model.encodeVoice(clip: clip, transcript: transcript.trimmingCharacters(in: .whitespacesAndNewlines), name: name.trimmingCharacters(in: .whitespaces))
                dismiss()
            } catch {
                self.error = "Couldn't add the voice: \(SessionModel.describe(error))"
            }
        }
    }
}

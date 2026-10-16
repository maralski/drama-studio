import DramaEngine
import SwiftUI

struct InspectorView: View {
    @Bindable var model: SessionModel
    @State private var designing: Speaker?

    var body: some View {
        Form {
            CastSection(model: model, designing: $designing)
            Section("Performance") {
                HStack {
                    Menu("Insert Cue", systemImage: "theatermasks") { CueMenuItems(model: model) }
                    HelpButton(.performanceCues)
                }
                Text("Cues go at the cursor. A passage cue wraps the selected text.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            AdvancedSection(model: model)
            if ModelSetup.isPreviewEngine {
                Section {
                    Label("Preview engine: audio is a placeholder tone until the Metal engine ships.", systemImage: "info.circle")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
        }
        .formStyle(.grouped)
        .disabled(model.state.isGenerating)
        .sheet(item: $designing) { DesignVoiceSheet(model: model, speaker: $0) }
    }
}

private struct CastSection: View {
    @Bindable var model: SessionModel
    @Binding var designing: Speaker?

    var body: some View {
        Section {
            if model.document == nil {
                Toggle(isOn: $model.screenplayMode) { HelpLabel("Screenplay mode", help: .screenplayMode) }
                    .help("Treat \u{201C}Name: line\u{201D} dialogue as separate speakers")
            }
            switch model.castSpeakers {
            case .success(let speakers):
                ForEach(speakers) { speaker in
                    Picker(selection: voiceBinding(for: speaker)) {
                        if !model.library.voices.isEmpty {
                            ForEach(model.library.voices) { Text($0.name).tag(VoicePickerTag.choice(.library($0.id))) }
                            Divider()
                        }
                        Text("Model default").tag(VoicePickerTag.choice(.modelDefault))
                        Text("Surprise me").tag(VoicePickerTag.choice(.surprise))
                        ForEach(model.castDesigns, id: \.self) { Label($0.name, systemImage: "wand.and.stars").tag(VoicePickerTag.choice(.designed($0))) }
                        Divider()
                        Text("Design a Voice\u{2026}").tag(VoicePickerTag.designNew)
                    } label: {
                        HStack(spacing: 4) {
                            VStack(alignment: .leading, spacing: 1) {
                                Text(speaker.name)
                                Text(speaker.tag).font(.caption.monospaced()).foregroundStyle(.secondary)
                            }
                            HelpButton(.speakerVoice)
                        }
                    }
                    if model.canKeepVoice(for: speaker) {
                        Button("Keep This Voice", systemImage: "square.and.arrow.down") { model.keepVoice(for: speaker) }
                            .help("Save the voice of the first take to the library, so it stays the same in later sessions")
                    }
                }
                ForEach(model.casting.ignored) { ignored in
                    Label(ignored.message, systemImage: "exclamationmark.triangle.fill")
                        .font(.caption)
                        .foregroundStyle(.orange)
                }
            case .failure(.empty):
                Text("Speakers appear here once there is text to narrate.").font(.caption).foregroundStyle(.secondary)
            case .failure(let error):
                Label(error.localizedDescription, systemImage: "exclamationmark.triangle.fill")
                    .font(.caption)
                    .foregroundStyle(.red)
            }
        } header: {
            Text("Cast")
        }
    }

    private func voiceBinding(for speaker: Speaker) -> Binding<VoicePickerTag> {
        Binding(get: { .choice(model.voice(for: speaker)) }, set: { tag in
            switch tag {
            case .choice(let choice): model.setVoice(choice, for: speaker)
            case .designNew: designing = speaker
            }
        })
    }
}

private struct AdvancedSection: View {
    @Bindable var model: SessionModel
    @State private var isExpanded = false

    var body: some View {
        Section {
            DisclosureGroup(isExpanded: $isExpanded) {
                VStack(alignment: .leading, spacing: 6) {
                    HStack {
                        HelpLabel("Expressiveness", help: .expressiveness)
                        Spacer()
                        Button("Default") { model.expressiveness = GenerationConfig.defaultExpressiveness }
                            .controlSize(.small)
                            .disabled(model.expressiveness == GenerationConfig.defaultExpressiveness)
                    }
                    Slider(value: $model.expressiveness, in: 0...1) {
                        Text("Expressiveness")
                    } minimumValueLabel: {
                        Text("Steady")
                    } maximumValueLabel: {
                        Text("Varied")
                    }
                    .labelsHidden()
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    Picker(selection: $model.modelPrecision) {
                        ForEach(ModelSetup.offeredPrecisions(current: model.modelPrecision), id: \.self) { mode in
                            Text(mode.studioTitle).tag(mode)
                        }
                    } label: {
                        HelpLabel("Model precision", help: .modelPrecision)
                    }
                    .padding(.top, 8)
                    Text("Takes effect at the next Generate, which reloads the model.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            } label: {
                Button { withAnimation { isExpanded.toggle() } } label: {
                    Text("Advanced")
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
            }
        }
    }
}

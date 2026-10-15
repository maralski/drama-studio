import DramaEngine
import SwiftUI

struct InspectorView: View {
    @Bindable var model: SessionModel

    var body: some View {
        Form {
            CastSection(model: model)
            Section("Performance") {
                HStack {
                    Menu("Insert Cue", systemImage: "theatermasks") { CueMenuItems(model: model) }
                    HelpButton(.performanceCues)
                }
                Text("Cues go at the cursor. A passage cue wraps the selected text.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            AdvancedSection(expressiveness: $model.expressiveness)
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
    }
}

private struct CastSection: View {
    @Bindable var model: SessionModel

    var body: some View {
        Section {
            Toggle(isOn: $model.screenplayMode) { HelpLabel("Screenplay mode", help: .screenplayMode) }
                .help("Treat \u{201C}Name: line\u{201D} dialogue as separate speakers")
            switch model.prepared {
            case .success(let script):
                ForEach(script.speakers) { speaker in
                    Picker(selection: voiceBinding(for: speaker)) {
                        Text("Model default").tag(ReferenceVoice.ID?.none)
                        ForEach(model.library.voices) { Text($0.name).tag(Optional($0.id)) }
                    } label: {
                        HStack(spacing: 4) {
                            VStack(alignment: .leading, spacing: 1) {
                                Text(speaker.name)
                                Text(speaker.tag).font(.caption.monospaced()).foregroundStyle(.secondary)
                            }
                            HelpButton(.speakerVoice)
                        }
                    }
                }
                ForEach(model.casting.ignored) { speaker in
                    Label("\(speaker.name)'s voice is not used: voices apply in speaker order, so every earlier speaker needs one too.", systemImage: "exclamationmark.triangle.fill")
                        .font(.caption)
                        .foregroundStyle(.orange)
                }
            case .failure(.empty):
                Text("Speakers appear here once there is a script.").font(.caption).foregroundStyle(.secondary)
            case .failure(let error):
                Label(error.localizedDescription, systemImage: "exclamationmark.triangle.fill")
                    .font(.caption)
                    .foregroundStyle(.red)
            }
        } header: {
            Text("Cast")
        }
    }

    private func voiceBinding(for speaker: Speaker) -> Binding<ReferenceVoice.ID?> {
        Binding(get: { model.voice(for: speaker) }, set: { model.setVoice($0, for: speaker) })
    }
}

private struct AdvancedSection: View {
    @Binding var expressiveness: Double
    @State private var isExpanded = false

    var body: some View {
        Section {
            DisclosureGroup(isExpanded: $isExpanded) {
                VStack(alignment: .leading, spacing: 6) {
                    HStack {
                        HelpLabel("Expressiveness", help: .expressiveness)
                        Spacer()
                        Button("Default") { expressiveness = GenerationConfig.defaultExpressiveness }
                            .controlSize(.small)
                            .disabled(expressiveness == GenerationConfig.defaultExpressiveness)
                    }
                    Slider(value: $expressiveness, in: 0...1) {
                        Text("Expressiveness")
                    } minimumValueLabel: {
                        Text("Steady")
                    } maximumValueLabel: {
                        Text("Varied")
                    }
                    .labelsHidden()
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

// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import DramaEngine
import SwiftUI

struct InspectorView: View {
    @Bindable var model: SessionModel
    @State private var designing: Speaker?

    var body: some View {
        Form {
            EngineSection(model: model)
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
                    Label("Preview mode uses placeholder tones instead of speech.", systemImage: "info.circle")
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

/// Which engine speaks new takes. Engines without their files show but
/// can't be chosen; Settings says how to add them.
private struct EngineSection: View {
    @Bindable var model: SessionModel

    var body: some View {
        Section {
            Picker("Engine", selection: $model.engine) { EngineChoices(model: model) }
                .accessibilityLabel("Speech engine")
            if let limits = model.engine.limits {
                Text(limits).font(.caption).foregroundStyle(.secondary)
            }
            let missing = SpeechEngineID.allCases.filter { !model.availableEngines.contains($0) }
            if !missing.isEmpty {
                HStack(alignment: .firstTextBaseline) {
                    Text("\(ListFormatter.localizedString(byJoining: missing.map(\.capabilities.name))) isn't installed. Add it in Settings.")
                        .font(.caption).foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                    Spacer(minLength: 4)
                    SettingsLink { Text("Settings\u{2026}") }
                        .controlSize(.small)
                        .help("Open Settings to download or locate another engine")
                }
            }
        }
    }
}

/// One item per engine, the uninstalled ones disabled; the Inspector's
/// picker and the Speech menu share it.
struct EngineChoices: View {
    let model: SessionModel

    var body: some View {
        ForEach(SpeechEngineID.allCases, id: \.self) { engine in
            let installed = model.availableEngines.contains(engine)
            Text(installed ? engine.capabilities.name : "\(engine.capabilities.name) (not installed)")
                .tag(engine)
                .selectionDisabled(!installed)
        }
    }
}

extension SpeechEngineID {
    /// What the engine does differently from one with every capability, from its capabilities table; nil for none.
    var limits: String? {
        let capabilities = capabilities
        var lacks: [String] = []
        if !capabilities.passageTags { lacks.append("has no passage cues such as <whisper>") }
        if let seconds = capabilities.maxCloneSeconds { lacks.append("clones best from clips of up to \(Int(seconds)) seconds") }
        if case .only(let languages) = capabilities.languages { lacks.append("speaks \(languages.count) languages") }
        if !capabilities.history { lacks.append("doesn't build on earlier takes") }
        if !capabilities.expressiveness { lacks.append("ignores Expressiveness") }
        var sentences: [String] = []
        if case .fixed = capabilities.cues { sentences.append("\(capabilities.name) performs fewer cues and leaves out the rest with a notice.") }
        if !lacks.isEmpty { sentences.append((sentences.isEmpty ? capabilities.name : "It") + " " + ListFormatter.localizedString(byJoining: lacks) + ".") }
        return sentences.isEmpty ? nil : sentences.joined(separator: " ")
    }
}

private struct CastSection: View {
    @Bindable var model: SessionModel
    @Binding var designing: Speaker?

    var body: some View {
        Section {
            if model.document == nil {
                Toggle(isOn: $model.screenplayMode) { HelpLabel("Screenplay mode", help: .screenplayMode) }
                    .accessibilityLabel("Screenplay mode")
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
                            }
                            HelpButton(.speakerVoice)
                        }
                    }
                    .accessibilityLabel("Voice for \(speaker.name)")
                    if model.canKeepVoice(for: speaker) {
                        Button("Save Session Voice", systemImage: "square.and.arrow.down") { model.keepVoice(for: speaker) }
                            .help("Save the checked voice to the library for later sessions")
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
                let capabilities = model.engine.capabilities
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
                    if !capabilities.expressiveness {
                        Text("\(capabilities.name) ignores Expressiveness. Switch to an engine that uses it to change it.")
                            .font(.caption).foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
                .disabled(!capabilities.expressiveness)
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

struct InspectorSheet: View {
    let model: SessionModel
    let dismiss: () -> Void

    var body: some View {
        VStack(spacing: 0) {
            Text("Voice and Performance").font(.headline)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(20)
            InspectorView(model: model)
            HStack {
                Spacer()
                Button("Done", action: dismiss).keyboardShortcut(.defaultAction)
            }
            .dialogActions()
        }
        .frame(width: 380, height: 500)
    }
}

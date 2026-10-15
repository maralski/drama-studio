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
            TuningSection(config: $model.config)
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

private struct TuningSection: View {
    @Binding var config: GenerationConfig

    var body: some View {
        Section("Sampling") {
            SliderRow(title: "Temperature", help: .temperature, value: $config.temperature, range: 0.1...1.5, step: 0.05)
            SliderRow(title: "Top-p", help: .topP, value: $config.topP, range: 0.05...1, step: 0.05)
            Stepper(value: $config.topK, in: 1...200) {
                LabeledContent { Text(config.topK.formatted()) } label: { HelpLabel("Top-k", help: .topK) }
            }
        }
        Section("Chunking") {
            Picker(selection: $config.chunking) {
                Text("Per sentence").tag(GenerationConfig.Chunking.sentences)
                Text("Reference").tag(GenerationConfig.Chunking.reference)
            } label: {
                HelpLabel("Chunking", help: .chunking)
            }
            Stepper(value: $config.chunkBytes, in: 100...1000, step: 50) {
                LabeledContent { Text("\(config.chunkBytes) bytes") } label: { HelpLabel("Chunk size", help: .chunkBytes) }
            }
            .disabled(config.chunking != .reference)
            Stepper(value: $config.maxNewFrames, in: 64...2048, step: 32) {
                LabeledContent {
                    Text("\(config.maxNewFrames) \u{00B7} \(seconds(frames: config.maxNewFrames)) s")
                } label: {
                    HelpLabel("Max frames", help: .maxFrames)
                }
            }
            Toggle(isOn: Binding(get: { config.stopAfterSilenceSeconds != nil }, set: { config.stopAfterSilenceSeconds = $0 ? GenerationConfig().stopAfterSilenceSeconds : nil })) {
                HelpLabel("Stop after silence", help: .stopAfterSilence)
            }
            if let silence = config.stopAfterSilenceSeconds {
                Stepper(value: Binding(get: { silence }, set: { config.stopAfterSilenceSeconds = $0 }), in: 0.5...10, step: 0.5) {
                    LabeledContent("Silence", value: "\(silence.formatted(.number.precision(.fractionLength(1)))) s")
                }
            }
            SliderRow(title: "Retry below", help: .retryThreshold, value: $config.minSpokenRatio, range: 0...0.9, step: 0.05)
        }
        Section {
            Toggle(isOn: Binding(get: { config.seed != nil }, set: { config.seed = $0 ? UInt64.random(in: 0...999_999) : nil })) {
                HelpLabel("Fixed seed", help: .seed)
            }
            if config.seed != nil {
                HStack {
                    TextField("Seed", value: Binding(get: { config.seed ?? 0 }, set: { config.seed = $0 }), format: .number.grouping(.never))
                        .labelsHidden()
                    Button("New Seed", systemImage: "dice") { config.seed = UInt64.random(in: 0...999_999) }
                        .labelStyle(.iconOnly)
                        .help("Draw a new seed")
                }
            } else {
                Text("Each take draws a new seed.").font(.caption).foregroundStyle(.secondary)
            }
            Button("Restore Defaults") { config = GenerationConfig() }
                .disabled(config == GenerationConfig())
        } header: {
            Text("Seed")
        }
    }

    private func seconds(frames: Int) -> String {
        (Double(frames * ModelConstants.samplesPerFrame) / Double(ModelConstants.sampleRate)).formatted(.number.precision(.fractionLength(1)))
    }
}

private struct SliderRow: View {
    let title: String
    let help: SettingHelp
    @Binding var value: Double
    let range: ClosedRange<Double>
    let step: Double

    var body: some View {
        LabeledContent {
            HStack {
                Slider(value: $value, in: range, step: step).labelsHidden()
                Text(value, format: .number.precision(.fractionLength(2)))
                    .monospacedDigit()
                    .frame(width: 34, alignment: .trailing)
            }
        } label: {
            HelpLabel(title, help: help)
        }
    }
}

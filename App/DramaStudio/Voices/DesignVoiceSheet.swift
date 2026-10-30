// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import DramaEngine
import SwiftUI

/// The fields of the design sheet.
struct DesignDraft: Equatable {
    var name = ""
    var description = ""
    var scene = ""
    var line = ""

    init(editing design: VoiceDesign?, line: String) {
        name = design?.name ?? ""
        description = design?.description ?? ""
        scene = design?.scene ?? ""
        self.line = line
    }

    mutating func fill(_ preset: CastPreset) {
        name = preset.design.name
        description = preset.design.description
        scene = preset.design.scene ?? ""
        line = preset.line
    }

    /// Nil until there is a name and a description.
    var design: VoiceDesign? {
        let name = name.trimmingCharacters(in: .whitespacesAndNewlines)
        let description = description.trimmingCharacters(in: .whitespacesAndNewlines)
        let scene = scene.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !name.isEmpty, !description.isEmpty else { return nil }
        return VoiceDesign(name: name, description: description, scene: scene.isEmpty ? nil : scene)
    }
}

struct DesignVoiceSheet: View {
    let model: SessionModel
    let speaker: Speaker
    @State private var draft: DesignDraft
    @Environment(\.dismiss) private var dismiss

    init(model: SessionModel, speaker: Speaker) {
        self.model = model
        self.speaker = speaker
        let firstLine = model.script.split(separator: "\n").map { $0.trimmingCharacters(in: .whitespaces) }.first { !$0.isEmpty } ?? ""
        _draft = State(initialValue: DesignDraft(editing: model.voice(for: speaker).design, line: String(firstLine.prefix(200))))
    }

    private var heardThisDesign: Bool { model.audition != nil && model.audition?.design == draft.design }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Form {
                Section {
                    LazyVGrid(columns: Array(repeating: GridItem(.flexible(), spacing: 6), count: 4), spacing: 6) {
                        ForEach(Examples.cast) { preset in
                            Button { draft.fill(preset) } label: {
                                Text(preset.design.name).frame(maxWidth: .infinity)
                            }
                            .help(preset.design.description)
                        }
                    }
                } header: {
                    Text("Start from a character")
                }
                Section {
                    TextField("Name", text: $draft.name, prompt: Text("Captain, Nana, \u{2026}"))
                    TextField("Description", text: $draft.description, prompt: Text("Age, texture, pace, mood"), axis: .vertical)
                        .lineLimit(2...4)
                    TextField("Scene", text: $draft.scene, prompt: Text("Optional: where the line is spoken"), axis: .vertical)
                        .lineLimit(1...3)
                } header: {
                    HelpLabel("Voice", help: .voiceDesign)
                }
                Section {
                    TextField("Sample line", text: $draft.line, prompt: Text("A line to hear the voice say"), axis: .vertical)
                        .lineLimit(2...4)
                    HStack {
                        Button("Try a Line", systemImage: "play.circle") {
                            if let design = draft.design { model.tryLine(draft.line, as: design) }
                        }
                        .disabled(draft.design == nil || DirectorsNote.spokenWords(draft.line).isEmpty || model.state.isGenerating)
                        if model.isAuditioning {
                            ProgressView().controlSize(.small)
                            Text("Generating the line\u{2026}").foregroundStyle(.secondary)
                        } else if heardThisDesign {
                            Label("Use Voice keeps the voice you just heard.", systemImage: "checkmark.circle")
                                .foregroundStyle(.secondary)
                        }
                    }
                    .font(.callout)
                } header: {
                    Text("Try it")
                } footer: {
                    Text("Trying a line doesn't change the script or the takes.")
                }
            }
            .formStyle(.grouped)
            HStack {
                Spacer()
                Button("Cancel", role: .cancel) {
                    model.cancelAudition()
                    dismiss()
                }
                .keyboardShortcut(.cancelAction)
                Button("Use Voice") {
                    if let design = draft.design { model.useDesign(design, for: speaker) }
                    dismiss()
                }
                .keyboardShortcut(.defaultAction)
                .disabled(draft.design == nil || model.isAuditioning)
            }
            .dialogActions()
        }
        .frame(width: 520, height: 580)
    }
}

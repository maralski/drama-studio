// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import DramaEngine
import SwiftUI

/// The built-in voices under one heading, such as Ming's English voices.
struct BuiltInVoiceGroup: Identifiable, Equatable {
    var engine: SpeechEngineID
    var name: String
    var voices: [BuiltInVoice]
    var id: String { "\(engine.rawValue):\(name)" }

    /// `voices` under their headings, in the order each heading first appears.
    /// A search keeps a voice whose name, emotion, heading or engine contains it.
    static func groups(of voices: [BuiltInVoice], matching search: String = "") -> [BuiltInVoiceGroup] {
        let search = search.trimmingCharacters(in: .whitespaces)
        var groups: [BuiltInVoiceGroup] = []
        for voice in voices {
            let fields = [voice.name, voice.note ?? "", voice.group, voice.engine.capabilities.name]
            guard search.isEmpty || fields.contains(where: { $0.localizedCaseInsensitiveContains(search) }) else { continue }
            if let index = groups.firstIndex(where: { $0.engine == voice.engine && $0.name == voice.group }) {
                groups[index].voices.append(voice)
            } else {
                groups.append(BuiltInVoiceGroup(engine: voice.engine, name: voice.group, voices: [voice]))
            }
        }
        return groups
    }
}

/// Picks one of the installed engines' built-in voices for a speaker.
struct BuiltInVoiceSheet: View {
    let model: SessionModel
    let speaker: Speaker
    @State private var search = ""
    @State private var selection: BuiltInVoice?
    @Environment(\.dismiss) private var dismiss

    init(model: SessionModel, speaker: Speaker) {
        self.model = model
        self.speaker = speaker
        _selection = State(initialValue: model.voice(for: speaker).builtIn)
    }

    var body: some View {
        let groups = BuiltInVoiceGroup.groups(of: model.builtInVoices, matching: search)
        VStack(alignment: .leading, spacing: 0) {
            VStack(alignment: .leading, spacing: 8) {
                Text("Choose a Built-in Voice").font(.headline)
                TextField("Search", text: $search, prompt: Text("Search by name, language or emotion"))
                    .textFieldStyle(.roundedBorder)
                    .accessibilityLabel("Search built-in voices")
            }
            .padding(20)
            List(selection: $selection) {
                ForEach(groups) { group in
                    Section("\(group.engine.capabilities.name): \(group.name) (\(group.voices.count))") {
                        ForEach(group.voices, id: \.self) { voice in
                            HStack {
                                Text(voice.name)
                                Spacer()
                                if let note = voice.note { Text(note).foregroundStyle(.secondary) }
                            }
                            .tag(voice)
                            .accessibilityElement(children: .combine)
                            .accessibilityAddTraits(.isButton)
                            .accessibilityAction { use(voice) }
                        }
                    }
                }
            }
            .overlay {
                if groups.isEmpty { ContentUnavailableView.search(text: search) }
            }
            if let selection, selection.engine != model.engine {
                Label("\(selection.name) speaks only on \(selection.engine.capabilities.name). Switch the engine to \(selection.engine.capabilities.name) before you generate.",
                      systemImage: "exclamationmark.triangle.fill")
                    .font(.caption)
                    .foregroundStyle(.orange)
                    .padding(.horizontal, 20)
                    .padding(.vertical, 8)
            }
            HStack {
                Text("A built-in voice speaks only on its own engine.").font(.caption).foregroundStyle(.secondary)
                Spacer()
                Button("Cancel", role: .cancel) { dismiss() }
                    .keyboardShortcut(.cancelAction)
                Button("Use Voice") {
                    if let selection { use(selection) }
                }
                .keyboardShortcut(.defaultAction)
                .disabled(selection == nil)
            }
            .dialogActions()
        }
        .frame(width: 460, height: 560)
    }

    private func use(_ voice: BuiltInVoice) {
        model.setVoice(.builtIn(voice), for: speaker)
        dismiss()
    }
}

import DramaEngine
import SwiftUI

/// The pronunciation rules, saved on every change.
struct PronunciationSettingsView: View {
    @State private var rows: [Row] = []
    @State private var selection: Set<Row.ID> = []
    @State private var problem: String?
    @State private var loaded = false

    private struct Row: Identifiable, Equatable {
        let id = UUID()
        var rule: PronunciationRule
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .firstTextBaseline, spacing: 4) {
                Text("Rules rewrite words in the script before the model hears them. Speaker tags, cues in brackets, and markup are never changed.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
                HelpButton(.pronunciation)
            }
            Table($rows, selection: $selection) {
                TableColumn("Match") { $row in
                    TextField("Word", text: $row.rule.match, prompt: Text("Nguyen"))
                }
                TableColumn("Say as") { $row in
                    TextField("Spelling", text: $row.rule.sayAs, prompt: Text("win"))
                }
                TableColumn("Whole word") { $row in
                    Toggle("Whole word", isOn: $row.rule.wholeWord).labelsHidden()
                }
                .width(80)
                TableColumn("Case sensitive") { $row in
                    Toggle("Case sensitive", isOn: $row.rule.caseSensitive).labelsHidden()
                }
                .width(100)
            }
            .frame(minHeight: 220)
            HStack {
                Button("Add Rule", systemImage: "plus") {
                    let row = Row(rule: PronunciationRule(match: "", sayAs: ""))
                    rows.append(row)
                    selection = [row.id]
                }
                Button("Remove", systemImage: "minus") {
                    rows.removeAll { selection.contains($0.id) }
                    selection = []
                }
                .disabled(selection.isEmpty)
                Spacer()
                if let problem {
                    Label(problem, systemImage: "exclamationmark.triangle.fill").font(.caption).foregroundStyle(.orange)
                } else {
                    Text("Saved for every take and section you generate.").font(.caption).foregroundStyle(.secondary)
                }
            }
            .controlSize(.small)
        }
        .padding(20)
        .onAppear(perform: load)
        .onChange(of: rows) { save() }
    }

    private func load() {
        do {
            rows = try PronunciationDictionary.load().rules.map { Row(rule: $0) }
            problem = nil
        } catch {
            problem = "Couldn't read the rules: \(error.localizedDescription)"
        }
        loaded = true
    }

    private func save() {
        guard loaded else { return }
        do {
            try PronunciationDictionary(rules: rows.map(\.rule)).save()
            problem = nil
        } catch {
            problem = "Couldn't save the rules: \(error.localizedDescription)"
        }
    }
}

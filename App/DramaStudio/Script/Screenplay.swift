import DramaEngine
import Foundation

struct Speaker: Identifiable, Equatable {
    let index: Int
    let name: String
    var id: Int { index }
    var tag: String { "<|speaker:\(index)|>" }
}

struct PreparedScript: Equatable {
    var text: String
    var speakers: [Speaker]
}

enum ScriptError: LocalizedError, Equatable {
    case empty
    case noDialogue
    case tooManySpeakers([String])

    var errorDescription: String? {
        switch self {
        case .empty: "The script is empty."
        case .noDialogue: "Screenplay mode found no \u{201C}Name: line\u{201D} dialogue."
        case .tooManySpeakers(let names): "Drama 3 supports \(Screenplay.maxSpeakers) speakers; this script has \(names.count): \(names.joined(separator: ", "))."
        }
    }
}

enum Screenplay {
    static let maxSpeakers = ModelConstants.maxSpeakerID + 1

    static func prepare(_ script: String, asScreenplay: Bool) -> Result<PreparedScript, ScriptError> {
        let text = script.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return .failure(.empty) }
        if asScreenplay { return convert(text) }
        let tagged = taggedSpeakers(in: text)
        let speakers = tagged.isEmpty ? [Speaker(index: 0, name: "Narrator")] : tagged.map { Speaker(index: $0, name: "Speaker \($0)") }
        return .success(PreparedScript(text: text, speakers: speakers))
    }

    /// Turns `Name: line` dialogue into `<|speaker:N|>line`, numbering names
    /// in order of first appearance. Unattributed lines continue the current
    /// speaker; lines before the first name go to a narrator.
    static func convert(_ script: String) -> Result<PreparedScript, ScriptError> {
        var names: [String] = []
        var keys: [String: Int] = [:]
        var turns: [(speaker: Int, text: String)] = []
        var current: Int?
        var sawDialogue = false

        func speaker(named name: String) -> Int {
            let key = name.lowercased()
            if let index = keys[key] { return index }
            names.append(name)
            keys[key] = names.count - 1
            return names.count - 1
        }

        for line in script.split(whereSeparator: \.isNewline).map({ $0.trimmingCharacters(in: .whitespaces) }) where !line.isEmpty {
            if let match = line.firstMatch(of: dialogueLine) {
                sawDialogue = true
                let index = speaker(named: String(match.1).trimmingCharacters(in: .whitespaces))
                current = index
                let spoken = (match.2.map(String.init) ?? "").trimmingCharacters(in: .whitespaces)
                if !spoken.isEmpty { turns.append((index, spoken)) }
            } else {
                let index = current ?? speaker(named: "Narrator")
                current = index
                if let last = turns.last, last.speaker == index {
                    turns[turns.count - 1].text += " " + line
                } else {
                    turns.append((index, line))
                }
            }
        }

        guard sawDialogue else { return .failure(.noDialogue) }
        guard names.count <= maxSpeakers else { return .failure(.tooManySpeakers(names)) }
        let text = turns.map { "<|speaker:\($0.speaker)|>\($0.text)" }.joined(separator: "\n")
        return .success(PreparedScript(text: text, speakers: names.enumerated().map { Speaker(index: $0.offset, name: $0.element) }))
    }

    /// True when most lines are `Name: line` dialogue from at least two names,
    /// so an imported file can switch Screenplay Mode on by itself.
    static func looksLikeScreenplay(_ text: String) -> Bool {
        let lines = text.split(whereSeparator: \.isNewline).map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
        let names = lines.compactMap { $0.firstMatch(of: dialogueLine).map { String($0.1).lowercased() } }
        return Set(names).count >= 2 && names.count * 2 >= lines.count
    }

    static func taggedSpeakers(in text: String) -> [Int] {
        Set(text.matches(of: speakerTag).compactMap { Int($0.1) }).sorted()
    }

    nonisolated(unsafe) private static let dialogueLine = #/^(\p{L}[\p{L}\p{N} .'’-]{0,39}?)\s*(?:\([^)]*\))?\s*:(?:\s+(.*))?$/#
    nonisolated(unsafe) private static let speakerTag = #/<\|speaker:(\d+)\|>/#
}

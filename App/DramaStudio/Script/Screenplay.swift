// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
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
        case .noDialogue: "Screenplay Mode needs dialogue such as \"Alex: Hello.\" Add a name, a colon and a space before the spoken words, or turn Screenplay Mode off. See Screenplay Mode help for examples."
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
            if let match = dialogue(in: line) {
                sawDialogue = true
                let index = speaker(named: match.name.trimmingCharacters(in: .whitespaces))
                current = index
                let spoken = match.spoken.trimmingCharacters(in: .whitespaces)
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

        guard sawDialogue, !turns.isEmpty else { return .failure(.noDialogue) }
        guard names.count <= maxSpeakers else { return .failure(.tooManySpeakers(names)) }
        let text = turns.map { "<|speaker:\($0.speaker)|>\($0.text)" }.joined(separator: "\n")
        return .success(PreparedScript(text: text, speakers: names.enumerated().map { Speaker(index: $0.offset, name: $0.element) }))
    }

    /// True when most lines are `Name: line` dialogue from at least two names,
    /// so an imported file can switch Screenplay Mode on by itself.
    static func looksLikeScreenplay(_ text: String) -> Bool {
        let lines = text.split(whereSeparator: \.isNewline).map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
        let names = lines.compactMap { dialogue(in: $0)?.name.lowercased() }
        return Set(names).count >= 2 && names.count * 2 >= lines.count
    }

    static func taggedSpeakers(in text: String) -> [Int] {
        Set(text.matches(of: speakerTag).compactMap { Int($0.1) }).sorted()
    }

    private static func dialogue(in line: String) -> (name: String, spoken: String)? {
        // Swift Regex preserves grapheme-cluster matching for Unicode names.
        if !line.utf8.allSatisfy({ $0 < 128 }) {
            return line.firstMatch(of: unicodeDialogueLine).map { (String($0.1), $0.2.map(String.init) ?? "") }
        }
        let text = line as NSString
        guard let match = dialogueLine.firstMatch(in: line, range: NSRange(location: 0, length: text.length)) else { return nil }
        let spoken = match.range(at: 2)
        return (text.substring(with: match.range(at: 1)), spoken.location == NSNotFound ? "" : text.substring(with: spoken))
    }

    nonisolated(unsafe) private static let unicodeDialogueLine = #/^(\p{L}[\p{L}\p{N} .'’-]{0,39}?)\s*(?:\([^)]*\))?\s*:(?:\s+(.*))?$/#
    private static let dialogueLine = try! NSRegularExpression(pattern: #"^(\p{L}[\p{L}\p{N} .'’-]{0,39}?)\s*(?:\([^)]*\))?\s*:(?:\s+(.*))?$"#)
    nonisolated(unsafe) private static let speakerTag = #/<\|speaker:(\d+)\|>/#
}

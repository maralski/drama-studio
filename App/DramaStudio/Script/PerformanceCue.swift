// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation

/// Inline performance tags Drama 3 understands. Add a row to extend the
/// Insert Cue menu; its kind picks the menu section and the written form.
struct PerformanceCue: Identifiable, Hashable {
    enum Form: Hashable { case marker, paired }

    enum Kind: String, CaseIterable {
        case reaction = "Reactions"
        case style = "Opening Styles"
        case passage = "Passages"

        var title: String { rawValue }

        var form: Form {
            switch self {
            case .reaction, .style: .marker
            case .passage: .paired
            }
        }
    }

    let tag: String
    let title: String
    let kind: Kind

    var id: String { tag }
    var form: Form { kind.form }
    var opening: String { form == .marker ? "[\(tag)]" : "<\(tag)>" }
    var closing: String { form == .marker ? "" : "</\(tag)>" }

    /// The cue study's best defaults still lost a clause in 2 of 40 runs.
    static let reliabilityNote = "Cues still cost a clause in about one take in twenty, most often right after a sentence end. Listen before you export."

    static let all: [PerformanceCue] = [
        PerformanceCue(tag: "laughs", title: "Laughs", kind: .reaction),
        PerformanceCue(tag: "chuckles", title: "Chuckles", kind: .reaction),
        PerformanceCue(tag: "sighs", title: "Sighs", kind: .reaction),
        PerformanceCue(tag: "gasps", title: "Gasps", kind: .reaction),
        PerformanceCue(tag: "pause", title: "Pause", kind: .reaction),
        PerformanceCue(tag: "clears throat", title: "Clears Throat", kind: .reaction),
        PerformanceCue(tag: "hesitates", title: "Hesitates", kind: .reaction),
        PerformanceCue(tag: "coughs", title: "Coughs", kind: .reaction),
        PerformanceCue(tag: "whispers", title: "Whispers", kind: .reaction),
        PerformanceCue(tag: "with bright eager excitement", title: "Excited", kind: .style),
        PerformanceCue(tag: "icy calm", title: "Icy Calm", kind: .style),
        PerformanceCue(tag: "heavy, deflated sadness", title: "Sad", kind: .style),
        PerformanceCue(tag: "sarcastic and deadpan", title: "Deadpan", kind: .style),
        PerformanceCue(tag: "breathless and terrified", title: "Terrified", kind: .style),
        PerformanceCue(tag: "furious, barely holding it together", title: "Furious", kind: .style),
        PerformanceCue(tag: "warm, gentle bedtime voice", title: "Bedtime", kind: .style),
        PerformanceCue(tag: "low and menacing", title: "Menacing", kind: .style),
        PerformanceCue(tag: "whisper", title: "Whisper Passage", kind: .passage),
        PerformanceCue(tag: "slow", title: "Slow Passage", kind: .passage),
        PerformanceCue(tag: "emphasis", title: "Emphasis", kind: .passage),
    ]

    static func cues(of kind: Kind) -> [PerformanceCue] { all.filter { $0.kind == kind } }

    static var pairedTagNames: [String] { all.filter { $0.form == .paired }.map(\.tag) }

    func insert(into text: String, at range: Range<Int>) -> (text: String, selection: Range<Int>) {
        let chars = Array(text)
        let before = String(chars[..<range.lowerBound])
        let selected = String(chars[range])
        let after = String(chars[range.upperBound...])
        let lead = before.last.map { $0.isWhitespace ? "" : " " } ?? ""
        let trail = after.first.map { $0.isWhitespace ? "" : " " } ?? ""
        switch form {
        case .marker:
            let inserted = lead + opening + (selected.isEmpty ? trail : " ")
            let start = before.count + inserted.count
            return (before + inserted + selected + after, start..<(start + selected.count))
        case .paired:
            let start = before.count + lead.count + opening.count
            return (before + lead + opening + selected + closing + trail + after, start..<(start + selected.count))
        }
    }
}

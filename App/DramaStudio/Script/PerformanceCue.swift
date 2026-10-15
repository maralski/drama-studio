import Foundation

/// Inline performance tags Drama 3 understands. Add a row to extend the
/// Insert Cue menu.
struct PerformanceCue: Identifiable, Hashable {
    enum Form: Hashable { case marker, paired }

    let tag: String
    let title: String
    let form: Form

    var id: String { tag }
    var opening: String { form == .marker ? "[\(tag)]" : "<\(tag)>" }
    var closing: String { form == .marker ? "" : "</\(tag)>" }

    static let all: [PerformanceCue] = [
        PerformanceCue(tag: "laughs", title: "Laughs", form: .marker),
        PerformanceCue(tag: "sighs", title: "Sighs", form: .marker),
        PerformanceCue(tag: "gasps", title: "Gasps", form: .marker),
        PerformanceCue(tag: "coughs", title: "Coughs", form: .marker),
        PerformanceCue(tag: "clears throat", title: "Clears Throat", form: .marker),
        PerformanceCue(tag: "whispers", title: "Whispers", form: .marker),
        PerformanceCue(tag: "whisper", title: "Whisper Passage", form: .paired),
    ]

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

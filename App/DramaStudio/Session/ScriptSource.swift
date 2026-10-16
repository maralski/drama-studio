import Foundation

/// What the session narrates: a plain script, or an imported document read section by section.
enum ScriptSource: Equatable {
    case plain(String)
    case document(NarrationDocument)

    var document: NarrationDocument? {
        if case .document(let document) = self { document } else { nil }
    }

    var plainText: String {
        if case .plain(let text) = self { text } else { "" }
    }
}

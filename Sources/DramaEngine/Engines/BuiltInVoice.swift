// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation

/// A voice an engine ships with. Unlike a ``ReferenceVoice`` it has no
/// recording and no transcript: it belongs to one engine, which knows it by
/// `name`, and no other engine can speak in it.
/// ``SpeechEngineID/builtInVoices`` lists the ones an engine has.
public struct BuiltInVoice: Sendable, Hashable, Codable {
    public let engine: SpeechEngineID
    /// The engine's own name for the voice, such as `US_Emma`.
    public let name: String

    public init(engine: SpeechEngineID, name: String) {
        self.engine = engine
        self.name = name
    }

    /// The name with its engine, such as `ming:US_Emma`, which no library
    /// voice's name can stand in for.
    public var qualifiedName: String { "\(engine.rawValue):\(name)" }

    /// The heading the voice is listed under, such as "English".
    public var group: String {
        switch engine {
        case .drama3: engine.capabilities.name
        case .ming: MingVoices.group(of: name)
        }
    }

    /// What the name alone does not say, such as the emotion a voice was recorded in.
    public var note: String? {
        switch engine {
        case .drama3: nil
        case .ming: MingVoices.note(on: name)
        }
    }

    /// The built-in voice `key` names: `engine:name` for one engine's voice,
    /// or a bare name looked up in every engine's list. An exact match wins
    /// over one that differs only in case.
    public static func named(_ key: String) -> BuiltInVoice? {
        let qualified = qualifier(of: key)
        let candidates = qualified?.engine.builtInVoices ?? SpeechEngineID.allCases.flatMap(\.builtInVoices)
        let name = qualified?.name ?? key
        return candidates.first { $0.name == name } ?? candidates.first { $0.name.caseInsensitiveCompare(name) == .orderedSame }
    }

    /// The engine and name of an `engine:name` key; nil when the key does not start with an engine's id and a colon.
    public static func qualifier(of key: String) -> (engine: SpeechEngineID, name: String)? {
        guard let colon = key.firstIndex(of: ":"), let engine = SpeechEngineID(rawValue: key[..<colon].lowercased()) else { return nil }
        return (engine, String(key[key.index(after: colon)...]))
    }

    /// - Throws: ``PromptError/engineUnsupported(_:_:)`` when `engine` is not
    ///   the voice's engine, or the voice's engine has no voice of this name.
    public func check(on engine: SpeechEngineID) throws {
        let owner = self.engine.capabilities.name
        guard engine == self.engine else {
            throw PromptError.engineUnsupported(engine, "speak in \(name), a built-in \(owner) voice; switch to \(owner) or choose another voice")
        }
        guard engine.capabilities.builtInVoices.contains(name) else {
            throw PromptError.engineUnsupported(engine, "speak in \(name); it has no built-in voice of that name")
        }
    }
}

public extension SpeechEngineID {
    /// The engine's built-in voices, in the engine's own order; empty for an engine without them.
    var builtInVoices: [BuiltInVoice] { capabilities.builtInVoices.map { BuiltInVoice(engine: self, name: $0) } }
}

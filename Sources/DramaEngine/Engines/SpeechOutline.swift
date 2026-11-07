// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation

/// How an engine will cut and perform a request, known before it runs and
/// without its weights: the script line of each generation turn, how long
/// each should take, and what the engine cannot perform as written. The app
/// reads it for script lines, the estimate and notices; each synthesizer
/// reports the same notices in its result.
public struct SpeechOutline: Sendable, Equatable {
    public let engine: SpeechEngineID
    /// The script text of each turn, cues as written.
    public let lines: [String]
    /// Expected spoken seconds per turn, at 16 characters per second or the
    /// slowest reference voice's pace.
    public let expectedSeconds: [Double]
    public let notices: [SpeechNotice]

    /// - Throws: ``PromptError`` for a request the engine cannot take.
    public init(_ request: SpeechRequest, engine: SpeechEngineID) throws {
        self.engine = engine
        if request.character != nil && !request.references.isEmpty { throw PromptError.voiceConflict }
        let text = request.text
        guard !UnicodeWhitespace.strip(text).isEmpty else { throw PromptError.emptyText("text") }
        if request.note.flatMap(DirectorsNote.clean) != nil, DirectorsNote.sentences(text).isEmpty { throw PromptError.emptyText("line") }
        var config = request.config
        if engine == .ming { config.chunking = .sentences }
        lines = GenerationPlan.lines(text: text, config: config)
        let voices = request.references.map { (DirectorsNote.spokenWords($0.transcript), $0.durationSeconds) }
        expectedSeconds = lines.map { SpokenPace.expectedSeconds(line: $0, voices: voices) }
        var notices: [SpeechNotice] = []
        let foreign = request.previousTakes.filter { $0.context.engine != engine }
        if let other = foreign.first { notices.append(.otherEngineTakes(other.context.engine, count: foreign.count)) }
        if engine == .ming { notices += MingCues.translate(lines: lines, note: request.note).notices }
        self.notices = notices
    }
}

/// How long a line takes to speak: 16 characters per second, or slower when a
/// reference voice speaks its own transcript more slowly.
package enum SpokenPace {
    package static func expectedSeconds(line: String, voices: [(transcript: String, seconds: Double)]) -> Double {
        let characters = DirectorsNote.spokenWords(line).count
        let secondsPerCharacter = voices.reduce(1 / FrameGenerator.charactersPerSecond) { pace, voice in
            let count = voice.transcript.count
            guard count > 0 else { return pace }
            return max(pace, voice.seconds / Double(count))
        }
        return Double(characters) * secondsPerCharacter
    }
}

// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation

/// How one request becomes prompts: the voice layout, the user turn of every
/// chunk, and the earlier takes that precede them as history.
package struct GenerationPlan: Sendable, Equatable {
    package var references: [Drama3Prompt.Reference]
    /// Set only for the design layout, where the builder tags speaker zero.
    package var character: Drama3Prompt.Character?
    package var history: [Drama3Prompt.HistoryTurn]
    package var chunks: [Chunk]
    package var streaming: StreamingMode
    /// The whole request's user turn, as its take records it.
    package var turnText: String
    /// The user's cleaned direction, separate from the script and prompt text.
    package var direction: String?

    package static let historyTakes = 2

    /// A slow reference needs time for its delivery and pauses. Use the slowest
    /// reference for a mixed cast, without shortening the default allowance.
    /// This budgets generation and optional silence probes; retries use the
    /// text-only duration because reference audio may include trailing silence.
    package func expectedSeconds(for line: String) -> Double {
        let characters = DirectorsNote.spokenWords(line).count
        let secondsPerCharacter = references.reduce(1 / FrameGenerator.charactersPerSecond) { pace, reference in
            let count = DirectorsNote.spokenWords(reference.transcript).count
            guard count > 0 else { return pace }
            let seconds = Double(reference.codes.first?.count ?? 0)
                * Double(ModelConstants.samplesPerFrame) / Double(ModelConstants.sampleRate)
            return max(pace, seconds / Double(count))
        }
        return Double(characters) * secondsPerCharacter
    }

    package struct Chunk: Sendable, Equatable {
        /// The user turn text passed to the prompt builder.
        package var turn: String
        /// The script text the turn speaks, which sets the frame budget.
        package var line: String
    }

    package init(text: String, references: [Drama3Prompt.Reference], character: VoiceDesign? = nil, previousTakes: [Take] = [],
                note: String? = nil, config: GenerationConfig) throws {
        if character != nil && !references.isEmpty { throw PromptError.voiceConflict }
        let note = note.flatMap(DirectorsNote.clean)
        direction = note
        let body: String
        if let note {
            guard !DirectorsNote.sentences(text).isEmpty else { throw PromptError.emptyText("line") }
            body = DirectorsNote.retakeText(line: text, note: note)
        } else {
            body = text
        }
        streaming = config.streaming
        if config.wordVerification == .required || (config.maxRetries > 0 && config.minSpokenRatio > 0) {
            streaming = .sentence
        }
        self.references = references
        self.character = nil
        if references.isEmpty, let voice = previousTakes.first {
            self.references = [Drama3Prompt.Reference(transcript: DirectorsNote.spokenText(ofTurn: voice.text, character: character), codes: Drama3Prompt.HistoryTurn(voice).codes)]
        } else if references.isEmpty {
            self.character = character
        }
        var pieces: [String]
        switch config.wordVerification == .required ? .sentences : config.chunking {
        case .reference: pieces = Drama3Chunking.split(text, maxBytes: config.chunkBytes)
        case .sentences: pieces = Drama3Chunking.splitSentencesTagged(text)
        }
        if pieces.isEmpty { pieces = [text] }
        chunks = pieces.map { piece in
            Chunk(turn: note.map { DirectorsNote.retakeText(line: piece, note: $0) } ?? piece, line: piece)
        }
        let turnText = character?.turn(body) ?? body
        self.turnText = turnText
        // A take of the very turn being generated, as history, makes the model
        // replay that take's audio: measured envelope correlation 0.93 to 0.99.
        history = previousTakes.filter { $0.text != turnText }.suffix(Self.historyTakes).map(Drama3Prompt.HistoryTurn.init)
        if let character {
            history = history.map { Drama3Prompt.HistoryTurn(text: character.removingName(from: $0.text), codes: $0.codes) }
        }
    }

    package init(_ request: SpeechRequest) throws {
        try self.init(text: request.text, references: request.references.map(Drama3Prompt.Reference.init),
                      character: request.character, previousTakes: request.previousTakes, note: request.note, config: request.config)
    }
}

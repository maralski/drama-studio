// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation

/// How one request becomes prompts: the voice layout, the user turn of every
/// chunk, and the earlier takes that precede them as history.
package struct GenerationPlan: Sendable, Equatable {
    package var references: [Drama3Prompt.Reference]
    /// Set only for the design layout, where the builder writes `Name: `.
    package var character: Drama3Prompt.Character?
    package var history: [Take]
    package var chunks: [Chunk]
    package var streaming: StreamingMode
    /// The whole request's user turn, as its take records it.
    package var turnText: String

    package static let historyTakes = 2

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
        let body: String
        if let note {
            guard !DirectorsNote.sentences(text).isEmpty else { throw PromptError.emptyText("line") }
            body = DirectorsNote.retakeText(line: text, note: note)
        } else {
            body = text
        }
        streaming = config.streaming
        if config.maxRetries > 0, config.minSpokenRatio > 0,
           body.range(of: #"\[[^\[\]]*\]|</?[A-Za-z][^<>]*>"#, options: .regularExpression) != nil {
            streaming = .sentence
        }
        var turn: (String) -> String = { $0 }
        self.references = references
        self.character = nil
        if references.isEmpty, let voice = previousTakes.first {
            self.references = [Drama3Prompt.Reference(transcript: DirectorsNote.spokenText(ofTurn: voice.text, character: character), codes: voice.codes)]
            if let character { turn = character.turn }
        } else if references.isEmpty {
            self.character = character
        }
        if note != nil {
            chunks = [Chunk(turn: turn(body), line: text)]
        } else {
            var pieces: [String]
            switch config.chunking {
            case .reference: pieces = Drama3Chunking.split(text, maxBytes: config.chunkBytes)
            case .sentences: pieces = Drama3Chunking.splitSentencesTagged(text)
            }
            if pieces.isEmpty { pieces = [text] }
            chunks = pieces.map { Chunk(turn: turn($0), line: $0) }
        }
        let turnText = character?.turn(body) ?? body
        self.turnText = turnText
        // A take of the very turn being generated, as history, makes the model
        // replay that take's audio: measured envelope correlation 0.93 to 0.99.
        history = Array(previousTakes.filter { $0.text != turnText }.suffix(Self.historyTakes))
    }

    package init(_ request: SpeechRequest) throws {
        try self.init(text: request.text, references: request.references.map { Drama3Prompt.Reference(transcript: $0.transcript, codes: $0.codes) },
                      character: request.character, previousTakes: request.previousTakes, note: request.note, config: request.config)
    }
}

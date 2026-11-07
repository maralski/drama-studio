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
        if note != nil, DirectorsNote.sentences(text).isEmpty { throw PromptError.emptyText("line") }
        streaming = config.effectiveStreaming
        self.references = references
        self.character = nil
        let dramaTakes = previousTakes.compactMap(Drama3Prompt.HistoryTurn.init)
        if references.isEmpty, let voice = dramaTakes.first {
            self.references = [Drama3Prompt.Reference(transcript: DirectorsNote.spokenText(ofTurn: voice.text, character: character), codes: voice.codes)]
        } else if references.isEmpty {
            self.character = character
        }
        let pieces = Self.lines(text: text, config: config)
        chunks = pieces.map { piece in
            Chunk(turn: note.map { DirectorsNote.retakeText(line: piece, note: $0) } ?? piece, line: piece)
        }
        let turnText = Self.turnText(text: text, note: note, character: character)
        self.turnText = turnText
        // A take of the very turn being generated, as history, makes the model
        // replay that take's audio: measured envelope correlation 0.93 to 0.99.
        history = Array(dramaTakes.filter { $0.text != turnText }.suffix(Self.historyTakes))
        if let character {
            history = history.map { Drama3Prompt.HistoryTurn(text: character.removingName(from: $0.text), codes: $0.codes) }
        }
    }

    /// The whole request as its take records it: a retake's direction and
    /// cues, then a character's name. Both engines record takes this way, so a
    /// take of the very turn being generated is recognized whatever made it.
    package static func turnText(text: String, note: String?, character: VoiceDesign?) -> String {
        let body = note.flatMap(DirectorsNote.clean).map { DirectorsNote.retakeText(line: text, note: $0) } ?? text
        return character?.turn(body) ?? body
    }

    /// The script text of each generation turn: sentences under required word
    /// verification, else the configured chunking.
    package static func lines(text: String, config: GenerationConfig) -> [String] {
        let pieces: [String]
        switch config.wordVerification == .required ? .sentences : config.chunking {
        case .reference: pieces = Drama3Chunking.split(text, maxBytes: config.chunkBytes)
        case .sentences: pieces = Drama3Chunking.splitSentencesTagged(text)
        }
        return pieces.isEmpty ? [text] : pieces
    }

    package init(_ request: SpeechRequest) throws {
        try self.init(text: request.text, references: request.references.map(Drama3Prompt.Reference.init),
                      character: request.character, previousTakes: request.previousTakes, note: request.note, config: request.config)
    }
}

extension GenerationConfig {
    /// Sentence streaming whenever an attempt may be replaced: under required
    /// word verification, or while retries and the spoken-ratio check are on.
    package var effectiveStreaming: StreamingMode {
        wordVerification == .required || (maxRetries > 0 && minSpokenRatio > 0) ? .sentence : streaming
    }
}

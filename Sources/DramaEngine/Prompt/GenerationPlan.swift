import Foundation

/// How one request becomes prompts: the voice layout, the user turn of every
/// chunk, and the earlier takes that precede them as history.
public struct GenerationPlan: Sendable, Equatable {
    public var references: [Drama3Prompt.Reference]
    /// Set only for the design layout, where the builder writes `Name: `.
    public var character: Drama3Prompt.Character?
    public var history: [Take]
    public var chunks: [Chunk]
    /// The whole request's user turn, as its take records it.
    public var turnText: String

    public static let historyTakes = 2

    public struct Chunk: Sendable, Equatable {
        /// The user turn text passed to the prompt builder.
        public var turn: String
        /// The script text the turn speaks, which sets the frame budget.
        public var line: String
    }

    public init(text: String, references: [Drama3Prompt.Reference], character: VoiceDesign? = nil, previousTakes: [Take] = [],
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
            case .sentences: pieces = Drama3Chunking.splitSentencesTagged(text, cueFold: config.cueFold)
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

    public init(_ request: SpeechRequest) throws {
        try self.init(text: request.text, references: request.references.map { Drama3Prompt.Reference(transcript: $0.transcript, codes: $0.codes) },
                      character: request.character, previousTakes: request.previousTakes, note: request.note, config: request.config)
    }
}

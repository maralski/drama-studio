// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation
package import DramaTTS

// DramaTTS holds the Drama 3 engine. DramaEngine's public API never names a
// DramaTTS type, so a binary DramaEngine framework needs no second module.
// Inside the package the engine's types keep the names they had here.
package typealias FrameGenerator = DramaFrameGenerator
package typealias GenerationPlan = DramaGenerationPlan
package typealias DirectorsNote = DramaDirectorsNote
package typealias ChunkAttempts = DramaChunkAttempts
package typealias ChunkAttempt = DramaChunkAttempt
package typealias PCM16 = DramaPCM16
typealias UnicodeWhitespace = DramaWhitespace

extension ModelLocation {
    package var checkpoint: DramaCheckpoint { DramaCheckpoint(directory: directory) }
}

extension VoiceDesign {
    package var drama: DramaVoiceDesign { DramaVoiceDesign(name: name, description: description, scene: scene) }
}

extension StreamingMode {
    package var drama: DramaStreamingMode {
        switch self {
        case .sentence: .sentence
        case .lowLatency(let stepFrames): .lowLatency(stepFrames: stepFrames)
        }
    }
}

extension FrameCap {
    package var drama: DramaFrameCap {
        switch self {
        case .fixed(let frames): .fixed(frames)
        case .fromText(let margin, let floorSeconds): .fromText(margin: margin, floorSeconds: floorSeconds)
        }
    }
}

extension GenerationConfig {
    /// The engine's options. Required word verification becomes a reviewed
    /// generation; the review itself is the caller's ``LocalWordVerifier``.
    package var drama: DramaSynthesisOptions {
        var options = DramaSynthesisOptions()
        options.reviewsAttempts = wordVerification == .required
        options.temperature = temperature
        options.topK = topK
        options.topP = topP
        options.frameCap = frameCap.drama
        options.seed = seed
        options.chunkBytes = chunkBytes
        options.chunking = switch chunking {
        case .reference: .reference
        case .sentences: .sentences
        }
        options.stopAfterSilenceSeconds = stopAfterSilenceSeconds
        options.minSpokenRatio = minSpokenRatio
        options.maxRetries = maxRetries
        options.stopOnLoop = stopOnLoop
        options.retryWithoutEOSBelow = retryWithoutEOSBelow
        options.capSecondsPerCue = capSecondsPerCue
        options.streaming = streaming.drama
        return options
    }

    /// Sentence streaming whenever an attempt may be replaced: under required
    /// word verification, or while retries and the spoken-ratio check are on.
    package var effectiveStreaming: StreamingMode {
        wordVerification == .required || (maxRetries > 0 && minSpokenRatio > 0) ? .sentence : streaming
    }
}

extension SpeechTimings {
    package init(_ timings: DramaTimings) {
        self.init()
        prefillSeconds = timings.prefillSeconds
        decodeSeconds = timings.decodeSeconds
        codecSeconds = timings.codecSeconds
        verificationSeconds = timings.verificationSeconds
        promptRows = timings.promptRows
        framesGenerated = timings.framesGenerated
    }
}

/// DramaTTS throws its own errors. The engine's callers get the ones
/// DramaEngine documents.
package enum Drama3Errors {
    package static func translate(_ error: Error) -> Error {
        switch error {
        case let error as DramaPromptError:
            switch error {
            case .emptyText(let part): PromptError.emptyText(part)
            case .structuralToken(let part, let token): PromptError.structuralToken(part, token)
            case .badSpeakerTag(let part): PromptError.badSpeakerTag(part)
            case .badCodes(let part): PromptError.badCodes(part)
            case .tooLong(let rows): PromptError.tooLong(rows)
            case .voiceConflict: PromptError.voiceConflict
            }
        case let error as DramaEngineError:
            switch error {
            case .noMetalDevice: EngineError.noMetalDevice
            case .missingKernel(let name): EngineError.missingKernel(name)
            case .badModel(let message): EngineError.badModel(message)
            case .badSchedule(let message): EngineError.badSchedule(message)
            }
        case let error as DramaCodecError:
            switch error {
            case .missingModel(let url): CodecError.missingModel(url)
            case .badCodes(let message): CodecError.badCodes(message)
            case .unexpectedOutput(let message): CodecError.unexpectedOutput(message)
            case .audio(let message): CodecError.audio(message)
            case .wrongShape(let url, let message): CodecError.wrongShape(url, message)
            }
        case let error as DramaTokenizerError:
            switch error {
            case .invalidFile(let part): TokenizerError.invalidFile(part)
            }
        case let error as DramaReviewError:
            switch error {
            case .unavailable(let reason): SpeechVerificationError.unavailable(reason)
            case .exhausted(let sentence, let attempts): SpeechVerificationError.exhausted(sentence: sentence, attempts: attempts)
            }
        default: error
        }
    }

    package static func translating<T>(_ body: () throws -> T) throws -> T {
        do { return try body() } catch { throw translate(error) }
    }

    package static func translating<T>(_ body: () async throws -> T) async throws -> T {
        do { return try await body() } catch { throw translate(error) }
    }
}

extension DramaGenerationPlan {
    /// The plan of a request for Drama 3.
    /// - Throws: ``PromptError``, including ``PromptError/notEncoded(_:_:)``
    ///   for a voice without a Drama 3 encoding.
    package init(_ request: SpeechRequest) throws {
        let references = try request.references.map(Drama3Prompt.Reference.init).map(\.drama)
        let takes = request.previousTakes.compactMap(Drama3Prompt.HistoryTurn.init).map(\.drama)
        self = try Drama3Errors.translating {
            try DramaGenerationPlan(text: request.text, references: references, character: request.character?.drama,
                                    previousTakes: takes, note: request.note, config: request.config.drama)
        }
    }

    package static func turnText(text: String, note: String?, character: VoiceDesign?) -> String {
        turnText(text: text, note: note, character: character?.drama)
    }

    package static func lines(text: String, config: GenerationConfig) -> [String] {
        lines(text: text, config: config.drama)
    }
}

extension DramaDirectorsNote {
    package static func spokenText(ofTurn text: String, character: VoiceDesign?) -> String {
        spokenText(ofTurn: text, character: character?.drama)
    }
}

extension DramaFrameGenerator {
    package static func frameBudget(for line: String, config: GenerationConfig, expectedSeconds: Double? = nil) -> Int {
        frameBudget(for: line, config: config.drama, expectedSeconds: expectedSeconds)
    }
}

extension DramaChunkAttempts {
    /// The acceptance loop under an app configuration.
    /// - Throws: ``SpeechVerificationError/exhausted(sentence:attempts:)``
    ///   when no attempt is accepted.
    package static func run<Attempt: DramaChunkAttempt>(index: Int, maxRetries: Int, expectedSeconds: Double, config: GenerationConfig,
                                                        shouldContinue: () -> Bool,
                                                        attempt: (Int) throws -> Attempt,
                                                        verify: (Attempt) throws -> Bool,
                                                        retry: (Int, Double) -> Void) throws -> Attempt {
        try Drama3Errors.translating {
            try run(index: index, maxRetries: maxRetries, expectedSeconds: expectedSeconds, config: config.drama,
                    shouldContinue: shouldContinue, attempt: attempt, verify: verify, retry: retry)
        }
    }
}

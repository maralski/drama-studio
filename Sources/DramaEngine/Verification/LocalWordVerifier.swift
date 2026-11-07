// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import AVFoundation
import Foundation
import Speech
import Synchronization

/// Native recognition receives only audio and a language, never the script as
/// hints. Keep this bridge independent of the occupied synthesizer actor.
package struct LocalWordVerifier: Sendable {
    let locale: Locale
    private let independent: IndependentWordRecognizer
    private let owner = UUID()
    private let plannedLines: [String]?

    package static func prepare(plan: GenerationPlan, config: GenerationConfig, shouldContinue: () -> Bool) throws -> Self? {
        try prepare(lines: plan.chunks.map(\.line), config: config, shouldContinue: shouldContinue)
    }

    /// A verifier for the planned `lines`, one per generated unit; nil unless the config requires word verification.
    package static func prepare(lines: [String], config: GenerationConfig, shouldContinue: () -> Bool) throws -> Self? {
        guard config.wordVerification == .required else { return nil }
        return try prepare(text: lines.joined(separator: " "), plannedLines: lines, shouldContinue: shouldContinue)
    }

    package static func prepare(text: String, shouldContinue: () -> Bool) throws -> Self {
        try prepare(text: text, plannedLines: nil, shouldContinue: shouldContinue)
    }

    private static func prepare(text: String, plannedLines: [String]?, shouldContinue: () -> Bool) throws -> Self {
        guard #available(macOS 26, *) else {
            throw SpeechVerificationError.unavailable("checked generation requires macOS 26 or later")
        }
        let deadline = ContinuousClock.now + .seconds(30)
        try WordCoverage.prepare()
        let spoken = DirectorsNote.spokenWords(text)
        guard !spoken.isEmpty else { throw PromptError.emptyText("spoken words") }
        let language = try ScriptLanguage.identify(spoken)
        let locale = try blocking(timeout: ContinuousClock.now.duration(to: deadline), shouldContinue: shouldContinue) {
            let installed = await SpeechTranscriber.installedLocales
            let requested = ScriptLanguage.preferredLocale(language: language, installed: installed, preferences: Locale.preferredLanguages)
                ?? Locale(identifier: language)
            let locale = await SpeechTranscriber.supportedLocale(equivalentTo: requested)
            return try ScriptLanguage.requireLocale(language: language, recognizerAvailable: SpeechTranscriber.isAvailable,
                                                     locale: locale, installed: installed)
        }
        let baseLanguage = Locale(identifier: language).language.languageCode?.identifier ?? language
        let recognitionLanguage = baseLanguage == "nb" ? "no" : (baseLanguage == "fil" ? "tl" : baseLanguage)
        let independent = try IndependentWordRecognizer(language: recognitionLanguage, deadline: deadline, shouldContinue: shouldContinue)
        return Self(locale: locale, independent: independent, plannedLines: plannedLines)
    }

    package func check(samples: [Float], line: String, shouldContinue: () -> Bool) throws -> Bool {
        try check(samples: samples, line: line, boundary: nil, shouldContinue: shouldContinue)
    }

    package struct VerifiedRawUnit: Sendable {
        fileprivate let samples: [Float]
        fileprivate let index: Int
        fileprivate let owner: UUID
        package var verifiedSamples: [Float] { samples }
    }

    private struct AdjacentBoundary {
        let previous: VerifiedRawUnit
        let current: VerifiedRawUnit
        let previousLine: String
        let currentLine: String
    }

    package func checkUnit(samples: [Float], at index: Int, shouldContinue: () -> Bool) throws -> VerifiedRawUnit? {
        guard let plannedLines, plannedLines.indices.contains(index) else {
            throw SpeechVerificationError.unavailable("a generated unit requires its prepared plan")
        }
        guard try check(samples: samples, line: plannedLines[index], shouldContinue: shouldContinue) else { return nil }
        guard shouldContinue() else { independent.cancel(); throw CancellationError() }
        return VerifiedRawUnit(samples: samples, index: index, owner: owner)
    }

    package func checkAdjacent(_ previous: VerifiedRawUnit, _ current: VerifiedRawUnit,
                               shouldContinue: () -> Bool) throws -> Bool {
        guard let plannedLines, previous.owner == owner, current.owner == owner,
              plannedLines.indices.contains(previous.index), plannedLines.indices.contains(current.index),
              current.index == previous.index + 1 else {
            throw SpeechVerificationError.unavailable("adjacent generated units must belong to this prepared plan")
        }
        let left = plannedLines[previous.index], right = plannedLines[current.index]
        return try check(samples: previous.samples + current.samples, line: left + " " + right,
                         boundary: AdjacentBoundary(previous: previous, current: current, previousLine: left, currentLine: right),
                         shouldContinue: shouldContinue)
    }

    private func check(samples: [Float], line: String, boundary: AdjacentBoundary?,
                       shouldContinue: () -> Bool) throws -> Bool {
        guard #available(macOS 26, *) else { throw SpeechVerificationError.unavailable("checked generation requires macOS 26 or later") }
        guard !samples.isEmpty else {
            SpeechAttemptAudit.record(samples: [], line: line, reason: "empty_audio", locale: locale.identifier)
            return false
        }
        let deadline = ContinuousClock.now + .seconds(30)
        let locale = locale
        let recognition: Recognition
        var confirmation: Recognition?
        do {
            recognition = try Self.blocking(timeout: ContinuousClock.now.duration(to: deadline), shouldContinue: shouldContinue) {
                try await Self.transcribe(samples, locale: locale)
            }
            if !recognition.matches(line: line, locale: locale), recognition.canConfirmEndOfAudio(line: line, locale: locale) {
                confirmation = try Self.blocking(timeout: ContinuousClock.now.duration(to: deadline), shouldContinue: shouldContinue) {
                    // Give the recognizer an acoustic endpoint without changing
                    // any generated samples or supplying the requested words.
                    try await Self.transcribe(samples + Array(repeating: 0, count: ModelConstants.sampleRate * 3 / 4), locale: locale)
                }
            }
        } catch {
            independent.cancel()
            SpeechAttemptAudit.record(samples: samples, line: line, reason: error is CancellationError ? "cancelled" : "recognition_error", locale: locale.identifier,
                                      error: error.localizedDescription)
            throw error
        }
        guard shouldContinue() else { independent.cancel(); throw CancellationError() }
        let boundarySpan = boundary.flatMap {
            recognition.boundaryWordSpan(previousLine: $0.previousLine, currentLine: $0.currentLine, locale: locale)
        }
        let nativeMatches = recognition.matches(line: line, locale: locale)
            || (confirmation?.matches(line: line, locale: locale, confidenceFloor: 0.9) ?? false)
            || boundarySpan != nil
        var independentTranscript = ""
        if nativeMatches {
            do {
                independentTranscript = try independent.transcript(pcm16: PCM16.encode(samples), deadline: deadline, shouldContinue: shouldContinue)
            } catch {
                SpeechAttemptAudit.record(samples: samples, line: line, reason: "independent_recognition_error", locale: locale.identifier,
                                          transcript: recognition.text, error: error.localizedDescription)
                throw error
            }
        }
        let matches = nativeMatches && Self.independentMatches(line: line, transcript: independentTranscript, locale: locale)
        if !matches || confirmation != nil || boundarySpan != nil {
            let matchedReason = boundarySpan != nil ? "verified_boundary_word_matched" : "endpoint_confirmation_matched"
            let witness = boundary.map {
                SpeechAttemptAudit.BoundaryWitness(
                    session: owner, previousIndex: $0.previous.index, currentIndex: $0.current.index,
                    previousSamples: $0.previous.samples, currentSamples: $0.current.samples,
                    expectedTokenOrdinal: WordCoverage.expectedTokens($0.previousLine, locale: locale).count,
                    heardTokenOrdinal: boundarySpan.map { span in
                        recognition.spans[..<span].reduce(0) { count, item in
                            count + WordCoverage.recognizedTokens(item.text, line: line, locale: locale, profile: Recognition.profile).count
                        }
                    }, nativeSpanOrdinal: boundarySpan)
            }
            SpeechAttemptAudit.record(samples: samples, line: line, reason: matches ? matchedReason : (nativeMatches ? "independent_word_check_rejected" : "word_check_rejected"), locale: locale.identifier,
                                      transcript: recognition.text, uncertainSpans: recognition.uncertainSpans(),
                                      confirmationTranscript: confirmation?.text ?? "",
                                      confirmationUncertainSpans: confirmation?.uncertainSpans(confidenceFloor: 0.9) ?? [],
                                      independentTranscript: independentTranscript,
                                      independentIdentity: independent.identity,
                                      confidenceSpans: recognition.spans, confirmationConfidenceSpans: confirmation?.spans ?? [],
                                      boundary: witness)
        }
        guard shouldContinue() else { independent.cancel(); throw CancellationError() }
        guard ContinuousClock.now < deadline else { independent.cancel(); throw SpeechVerificationError.timedOut }
        return matches
    }

    /// The independent recognizer writes numbers as it heard them, so none of
    /// Apple's formatting losses are forgiven in its transcript.
    static func independentMatches(line: String, transcript: String, locale: Locale) -> Bool {
        WordCoverage.matches(line: line, transcript: transcript, locale: locale, profile: .exact)
    }

    /// Bounded wait; cancellation does not wait for a late framework callback.
    /// The detached task owns its buffers and cannot commit generation output.
    private static let nativeCapacity = DispatchSemaphore(value: 4)
    package static func blocking<Value: Sendable>(timeout: Duration = .seconds(30), shouldContinue: () -> Bool,
                                                 work: @escaping @Sendable () async throws -> Value) throws -> Value {
        let deadline = ContinuousClock.now + timeout
        guard shouldContinue() else { throw CancellationError() }
        while nativeCapacity.wait(timeout: .now() + .milliseconds(50)) == .timedOut {
            guard shouldContinue() else { throw CancellationError() }
            guard ContinuousClock.now < deadline else { throw SpeechVerificationError.timedOut }
        }
        guard ContinuousClock.now < deadline else { nativeCapacity.signal(); throw SpeechVerificationError.timedOut }
        let completion = Mutex<Result<Value, Error>?>(nil)
        let signal = DispatchSemaphore(value: 0)
        let worker = Task.detached {
            defer { nativeCapacity.signal() }
            let result: Result<Value, Error>
            do { result = .success(try await work()) } catch { result = .failure(error) }
            completion.withLock { $0 = result }
            signal.signal()
        }
        while signal.wait(timeout: .now() + .milliseconds(100)) == .timedOut {
            guard shouldContinue() else { worker.cancel(); throw CancellationError() }
            guard ContinuousClock.now < deadline else { worker.cancel(); throw SpeechVerificationError.timedOut }
        }
        guard shouldContinue() else { worker.cancel(); throw CancellationError() }
        guard ContinuousClock.now < deadline else { worker.cancel(); throw SpeechVerificationError.timedOut }
        return try completion.withLock { try $0!.get() }
    }

    /// Apple SpeechTranscriber's transcript with its word confidences.
    package struct Recognition: Sendable {
        package struct Span: Sendable {
            var text: String
            var confidence: Double
            var hasReportedConfidence: Bool = true
        }
        var text = ""
        var spans: [Span] = []
        static let profile = WordCoverage.RecognizerProfile.apple

        /// Apple reports correct words at 0.35-0.49 often enough that 0.5 rejected
        /// half of all correct generated clips; the independent check still applies.
        static let confidenceFloor = 0.3
        /// A clipped final word also scores below this, so the last word keeps it
        /// and goes through end-of-audio confirmation instead.
        static let endOfAudioFloor = 0.5

        func uncertainSpans(confidenceFloor: Double = Self.confidenceFloor) -> [String] {
            uncertainIndices(confidenceFloor: confidenceFloor).map { spans[$0].text }
        }

        private func uncertainIndices(confidenceFloor: Double = Self.confidenceFloor) -> [Int] {
            let last = spans.lastIndex { $0.text.contains { $0.isLetter || $0.isNumber } }
            return spans.indices.filter {
                spans[$0].confidence < ($0 == last ? max(confidenceFloor, Self.endOfAudioFloor) : confidenceFloor)
            }
        }

        func matches(line: String, locale: Locale, confidenceFloor: Double = Self.confidenceFloor) -> Bool {
            WordCoverage.matches(line: line, transcript: text, locale: locale, profile: Self.profile,
                                 uncertainSpans: uncertainSpans(confidenceFloor: confidenceFloor))
        }

        func canConfirmEndOfAudio(line: String, locale: Locale) -> Bool {
            guard WordCoverage.matches(line: line, transcript: text, locale: locale, profile: Self.profile) else { return false }
            let lexical = spans.filter { !WordCoverage.tokens($0.text, locale: locale, profile: Self.profile).isEmpty }
            guard let last = lexical.last, last.confidence < Self.endOfAudioFloor,
                  WordCoverage.tokens(last.text, locale: locale, profile: Self.profile).count == 1 else { return false }
            return lexical.dropLast().allSatisfy { $0.confidence >= Self.confidenceFloor }
        }

        func boundaryWordSpan(previousLine: String, currentLine: String, locale: Locale) -> Int? {
            let previous = WordCoverage.expectedTokens(previousLine, locale: locale)
            let current = WordCoverage.expectedTokens(currentLine, locale: locale)
            let line = previousLine + " " + currentLine
            guard !previous.isEmpty, current.count > 1,
                  WordCoverage.expectedTokens(line, locale: locale) == previous + current,
                  let firstCurrent = WordCoverage.unambiguousHeardOrdinal(line: line, transcript: text, expectedOrdinal: previous.count,
                                                                          locale: locale, profile: Self.profile) else { return nil }
            let spanWords = spans.map { WordCoverage.recognizedTokens($0.text, line: line, locale: locale, profile: Self.profile) }
            guard spans.map(\.text).joined() == text,
                  spanWords.flatMap({ $0 }) == WordCoverage.recognizedTokens(text, line: line, locale: locale, profile: Self.profile) else { return nil }
            let uncertain = uncertainIndices()
            var ordinal = 0
            for (index, span) in spans.enumerated() {
                defer { ordinal += spanWords[index].count }
                // A number token can stand for several spoken words, so it never qualifies.
                guard ordinal == firstCurrent, spanWords[index].count == 1, !WordCoverage.isNumber(spanWords[index][0]),
                      span.hasReportedConfidence, span.confidence.isFinite,
                      span.confidence > 0, uncertain.contains(index) else { continue }
                let remaining = uncertain.filter { $0 != index }.map { spans[$0].text }
                if WordCoverage.matches(line: line, transcript: text, locale: locale, profile: Self.profile,
                                        uncertainSpans: remaining) { return index }
            }
            return nil
        }
    }

    @available(macOS 26, *)
    private static func transcribe(_ samples: [Float], locale: Locale) async throws -> Recognition {
        try Task.checkCancellation()
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("drama-word-check-\(UUID().uuidString).wav")
        defer { try? FileManager.default.removeItem(at: url) }
        guard FileManager.default.createFile(atPath: url.path, contents: PCM16.wav(samples), attributes: [.posixPermissions: 0o600]) else {
            throw SpeechVerificationError.unavailable("cannot write the recognition audio buffer")
        }
        let file = try AVAudioFile(forReading: url)
        let transcriber = SpeechTranscriber(locale: locale, transcriptionOptions: [], reportingOptions: [],
                                            attributeOptions: [.transcriptionConfidence])
        let analyzer = SpeechAnalyzer(modules: [transcriber])
        let reader = Task {
            var recognition = Recognition()
            for try await result in transcriber.results {
                try Task.checkCancellation()
                recognition.text += String(result.text.characters)
                for run in result.text.runs {
                    recognition.spans.append(.init(text: String(result.text[run.range].characters),
                                                   confidence: Double(run.transcriptionConfidence ?? 0),
                                                   hasReportedConfidence: run.transcriptionConfidence != nil))
                }
            }
            return recognition
        }
        return try await withTaskCancellationHandler {
            do {
                _ = try await analyzer.analyzeSequence(from: file)
                try await analyzer.finalizeAndFinishThroughEndOfInput()
                return try await reader.value
            } catch {
                reader.cancel()
                await analyzer.cancelAndFinishNow()
                throw error
            }
        } onCancel: {
            reader.cancel()
            Task { await analyzer.cancelAndFinishNow() }
        }
    }
}

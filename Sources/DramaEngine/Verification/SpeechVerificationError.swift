// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation

public enum SpeechVerificationError: Error, LocalizedError, Sendable {
    case unavailable(String)
    /// The request's language, not this Mac, is why its words can't be checked:
    /// the script mixes languages, its language is uncertain, or no recognizer knows it.
    case language(String)
    case exhausted(sentence: Int, attempts: Int)
    case timedOut

    public var errorDescription: String? {
        switch self {
        case .unavailable(let reason), .language(let reason):
            return "Cannot check the generated words: \(reason)."
        case .exhausted(let sentence, let attempts):
            return "Sentence \(sentence) did not pass the spoken-word check after \(attempts) attempts. Generation stopped without creating a completed take."
        case .timedOut:
            return "The local spoken-word check timed out. Generation stopped without creating a completed take."
        }
    }
}

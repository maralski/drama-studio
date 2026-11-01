// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation

public enum SpeechVerificationError: Error, LocalizedError, Sendable {
    case unavailable(String)
    case exhausted(sentence: Int, attempts: Int)
    case timedOut

    public var errorDescription: String? {
        switch self {
        case .unavailable(let reason):
            return "Cannot check the generated words: \(reason). Quick preview is available for unchecked audio."
        case .exhausted(let sentence, let attempts):
            return "Sentence \(sentence) did not pass the spoken-word check after \(attempts) attempts. Generation stopped without creating a completed take."
        case .timedOut:
            return "The local spoken-word check timed out. Generation stopped without creating a completed take."
        }
    }
}

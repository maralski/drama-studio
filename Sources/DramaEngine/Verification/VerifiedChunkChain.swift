// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation

/// Accepts generated chunks in order: each one's words, and its join with the
/// previous accepted chunk. Both engines run their attempts through one of
/// these. Without a verifier every chunk is accepted and nothing is replaced.
package struct VerifiedChunkChain {
    private let verifier: LocalWordVerifier?
    private var previous: LocalWordVerifier.VerifiedRawUnit?
    private var pending: LocalWordVerifier.VerifiedRawUnit?

    package init(_ verifier: LocalWordVerifier?) { self.verifier = verifier }

    /// Whether the chunk's audio speaks its line and joins the previous chunk.
    /// The last accepted attempt is pending until ``commit()``.
    package mutating func accept(_ samples: [Float], at index: Int, shouldContinue: () -> Bool) throws -> Bool {
        guard let verifier else { throw SpeechVerificationError.unavailable("the spoken-word verifier is missing") }
        pending = nil
        guard let current = try verifier.checkUnit(samples: samples, at: index, shouldContinue: shouldContinue) else { return false }
        if index > 0 {
            guard let previous else { throw SpeechVerificationError.unavailable("the previous generated unit is not verified") }
            guard try verifier.checkAdjacent(previous, current, shouldContinue: shouldContinue) else { return false }
        }
        pending = current
        return true
    }

    /// Makes the pending attempt the chain's last accepted chunk and returns
    /// its verified samples; nil without a verifier.
    /// - Throws: when a verifier is set and no attempt is pending.
    package mutating func commit() throws -> [Float]? {
        guard verifier != nil else { return nil }
        guard let pending else { throw SpeechVerificationError.unavailable("the selected generated unit is not verified") }
        previous = pending
        self.pending = nil
        return pending.verifiedSamples
    }
}

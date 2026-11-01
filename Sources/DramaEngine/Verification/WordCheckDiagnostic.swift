// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation

/// Offline packaging gate, shared by the app and portable CLI. Runs the real
/// installed-resource resolver and both audio-only recognizers without Drama.
package enum WordCheckDiagnostic {
    package static func run(_ arguments: [String]) -> Int32 {
        guard arguments.count == 3, ["accepted", "rejected"].contains(arguments[2]) else {
            print("usage: check-words <WAV> <script> <accepted|rejected>")
            return 1
        }
        do {
            let verifier = try LocalWordVerifier.prepare(text: arguments[1], shouldContinue: { true })
            let samples = try DramaCodec.loadAudio(url: URL(fileURLWithPath: arguments[0]))
            let accepted = try verifier.check(samples: samples, line: arguments[1], shouldContinue: { true })
            let data = try JSONSerialization.data(withJSONObject: ["accepted": accepted, "policy_revision": WordVerificationPolicy.currentRevision,
                                                                  "policy_definition": WordVerificationPolicy.currentDefinition])
            print(String(decoding: data, as: UTF8.self))
            return accepted == (arguments[2] == "accepted") ? 0 : 1
        } catch {
            print("Word-check diagnostic failed: \(error.localizedDescription)")
            return 1
        }
    }
}

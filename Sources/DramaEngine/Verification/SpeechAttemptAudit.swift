// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation
import CryptoKit
import WordRecognitionProtocol

/// Opt-in local diagnostics. Shipping builds omit the bundle key.
package enum SpeechAttemptAudit {
    package struct BoundaryWitness {
        let session: UUID
        let previousIndex: Int
        let currentIndex: Int
        let previousSamples: [Float]
        let currentSamples: [Float]
        let expectedTokenOrdinal: Int
        let heardTokenOrdinal: Int?
        let nativeSpanOrdinal: Int?
    }

    package static func record(samples: [Float], line: String, reason: String, frames: [Frame] = [],
                               locale: String = "", transcript: String = "", uncertainSpans: [String] = [], error: String = "",
                               independentTranscript: String = "",
                               independentIdentity: RecognitionReady? = nil,
                               confidenceSpans: [LocalWordVerifier.Recognition.Span] = [],
                               boundary: BoundaryWitness? = nil) {
        let path = (Bundle.main.object(forInfoDictionaryKey: "DSWordCheckAuditDirectory") as? String)
            ?? (Bundle.main.bundleURL.pathExtension != "app" ? ProcessInfo.processInfo.environment["DS_WORD_CHECK_AUDIT_DIR"] : nil)
        guard let path else { return }
        do {
            func digest(_ samples: [Float]) -> String {
                SHA256.hash(data: PCM16.encode(samples)).map { String(format: "%02x", $0) }.joined()
            }
            let witness: [String: Any] = boundary.map {
                var fields: [String: Any] = [
                    "session": $0.session.uuidString,
                    "previous_unit_index": $0.previousIndex, "current_unit_index": $0.currentIndex,
                    "sample_boundary": $0.previousSamples.count,
                    "previous_sample_count": $0.previousSamples.count, "current_sample_count": $0.currentSamples.count,
                    "previous_pcm_sha256": digest($0.previousSamples), "current_pcm_sha256": digest($0.currentSamples),
                    "pair_pcm_sha256": digest(samples), "expected_token_ordinal": $0.expectedTokenOrdinal
                ]
                if let ordinal = $0.heardTokenOrdinal { fields["heard_token_ordinal"] = ordinal }
                if let ordinal = $0.nativeSpanOrdinal { fields["native_span_ordinal"] = ordinal }
                return fields
            } ?? [:]
            let directory = URL(filePath: path, directoryHint: .isDirectory)
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
            let id = UUID().uuidString
            let details: [String: Any] = ["time": ISO8601DateFormatter().string(from: Date()), "line": line, "reason": reason,
                                          "locale": locale, "transcript": transcript, "uncertain_spans": uncertainSpans,
                                          "independent_transcript": independentTranscript,
                                          "policy_revision": WordRecognitionAssets.policyRevision,
                                          "policy_definition": WordRecognitionAssets.policyDefinition,
                                          "boundary_witness": witness,
                                          "independent_identity": independentIdentity.map { [
                                            "protocol_version": $0.protocolVersion, "configuration_id": $0.configurationID,
                                            "model_id": $0.modelID, "model_sha256": $0.modelSHA256
                                          ] as [String: Any] } ?? [:],
                                          "confidence_spans": confidenceSpans.map { ["text": $0.text, "confidence": $0.confidence, "has_reported_confidence": $0.hasReportedConfidence] as [String: Any] },
                                          "frames": frames, "error": error]
            let data = try JSONSerialization.data(withJSONObject: details, options: [.prettyPrinted, .sortedKeys])
            _ = FileManager.default.createFile(atPath: directory.appendingPathComponent(id + ".json").path, contents: data, attributes: [.posixPermissions: 0o600])
            if !samples.isEmpty {
                _ = FileManager.default.createFile(atPath: directory.appendingPathComponent(id + ".wav").path, contents: PCM16.wav(samples), attributes: [.posixPermissions: 0o600])
            }
        } catch {
            // Diagnostics must never alter the acceptance decision or error.
        }
    }
}

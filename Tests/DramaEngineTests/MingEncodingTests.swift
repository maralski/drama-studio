// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import XCTest
@testable import DramaEngine

final class MingEncodingTests: XCTestCase {
    private func reference(rows: Int = 3, transcript: String = "Hello there.") throws -> MingVoiceEncoding {
        try MingVoiceEncoding(transcript: transcript, sampleCount: rows * 3528,
                              latents: (0..<rows * 64).map { Float($0) * 0.01 - 0.5 },
                              speakerEmbedding: (0..<192).map { Float($0) / 192 })
    }

    func testAMingVoiceRoundTripsThroughFormatFiveJSONUnderItsOwnKey() throws {
        let voice = ReferenceVoice(name: "Ada", transcript: "Hello there.", encodings: [try .ming(reference: reference())], clipSampleCount: 10_584)
        let data = try JSONEncoder().encode(voice)
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let encodings = try XCTUnwrap(json["encodings"] as? [String: Any])
        let ming = try XCTUnwrap((encodings["ming"] as? [String: Any])?["ming"] as? [String: Any])
        let stored = try XCTUnwrap(ming["reference"] as? [String: Any])
        XCTAssertEqual(Set(stored.keys), ["transcript", "sampleCount", "latents", "speakerEmbedding"])
        XCTAssertTrue(stored["latents"] is String, "latents are base64, not a JSON array")
        XCTAssertEqual(try JSONDecoder().decode(ReferenceVoice.self, from: data), voice)
        XCTAssertEqual(voice.encodings[.ming]?.sampleCount, 3 * 3528)
        XCTAssertEqual(voice.durationSeconds, 10_584 / 44_100)
    }

    func testAMingTakeRoundTripsAndReportsTheWholeTakeLength() throws {
        let take = Take(text: "Hello there. And more.", context: .ming(reference: try reference(), sampleCount: 90_000))
        let data = try JSONEncoder().encode(take)
        let decoded = try JSONDecoder().decode(Take.self, from: data)
        XCTAssertEqual(decoded, take)
        XCTAssertEqual(decoded.sampleCount, 90_000)
        XCTAssertEqual(decoded.context.engine, .ming)
        XCTAssertEqual(decoded.context.voiceEncoding, .ming(reference: try reference()))
        XCTAssertFalse(String(decoding: data, as: UTF8.self).contains("\"codes\""))
    }

    func testOldTakesAndVoicesStillDecodeAsDramaThree() throws {
        let codes = (0..<10).map { book in [Int32(book), Int32(book + 1)] }
        let take = try JSONDecoder().decode(Take.self, from: try JSONSerialization.data(withJSONObject: ["text": "Hi.", "codes": codes]))
        XCTAssertEqual(take.context, .drama3(codes: codes))
        let phase4 = try JSONSerialization.data(withJSONObject: [
            "id": UUID().uuidString, "name": "Old", "transcript": "Hi.", "encodings": ["drama3": ["drama3": ["codes": codes]]],
        ])
        let voice = try JSONDecoder().decode(ReferenceVoice.self, from: phase4)
        XCTAssertEqual(voice.encodings[.drama3], .drama3(codes: codes))
        XCTAssertNil(voice.encodings[.ming])
    }

    func testAReferenceClipLosesItsTrailingSilenceButKeepsItsSpeech() {
        let frame = ModelConstants.samplesPerFrame
        let speech = (0..<frame * 20).map { sin(Float($0) * 2 * .pi * 220 / 44_100) * 0.3 }
        let silence = [Float](repeating: 0.001, count: frame * 6)
        let trimmed = MingSynthesizer.trimmed(speech + silence)
        XCTAssertEqual(trimmed.count, speech.count, "six silent frames go, the speech stays whole")
        XCTAssertEqual(trimmed, speech)
        XCTAssertEqual(MingSynthesizer.trimmed(speech), speech, "a clip that ends in speech is unchanged")
        XCTAssertEqual(MingSynthesizer.trimmed(speech + silence + speech).count, speech.count * 2 + silence.count, "an inner pause is not trailing")
    }

    func testBadShapesAreRejectedAtTheBoundary() throws {
        XCTAssertThrowsError(try MingVoiceEncoding(transcript: "x", sampleCount: 10, latents: [Float](repeating: 0, count: 65), speakerEmbedding: [Float](repeating: 0, count: 192)))
        XCTAssertThrowsError(try MingVoiceEncoding(transcript: "x", sampleCount: 10, latents: [Float](repeating: 0, count: 64), speakerEmbedding: [Float](repeating: 0, count: 191)))
        XCTAssertThrowsError(try MingVoiceEncoding(transcript: "x", sampleCount: 0, latents: [Float](repeating: 0, count: 64), speakerEmbedding: [Float](repeating: 0, count: 192)))
        var json = try XCTUnwrap(JSONSerialization.jsonObject(with: try JSONEncoder().encode(try reference())) as? [String: Any])
        json["speakerEmbedding"] = Data([0, 0, 0, 0]).base64EncodedString()
        XCTAssertThrowsError(try JSONDecoder().decode(MingVoiceEncoding.self, from: try JSONSerialization.data(withJSONObject: json))) { error in
            XCTAssertTrue(error is DecodingError, "\(error)")
        }
    }

    func testAVoiceCanHoldBothEncodings() throws {
        var voice = ReferenceVoice(name: "Both", transcript: "Hi.", encodings: [.drama3(codes: (0..<10).map { _ in [1, 2] })])
        voice.store(try .ming(reference: reference()))
        XCTAssertEqual(Set(voice.encodings.keys), [.drama3, .ming])
        let decoded = try JSONDecoder().decode(ReferenceVoice.self, from: try JSONEncoder().encode(voice))
        XCTAssertEqual(decoded, voice)
    }
}

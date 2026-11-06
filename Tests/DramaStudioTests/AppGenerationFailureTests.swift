// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import DramaEngine
import XCTest
@testable import DramaStudio

private struct IncompleteSpeechSynthesizer: SpeechSynthesizer {
    var isLoaded: Bool { true }
    func load(onPhase: @escaping @Sendable (String) -> Void) async throws {}
    var engineDescription: EngineDescription { EngineDescription(engine: .drama3, modelID: "fishaudio/drama-3", snapshot: "test", isMock: true) }
    var isTuned: Bool? { false }
    func encodeVoice(samples: [Float], transcript: String) async throws -> VoiceEncoding { throw CancellationError() }
    func synthesize(_ request: SpeechRequest) -> AsyncThrowingStream<SpeechEvent, Error> {
        AsyncThrowingStream { out in
            out.yield(.chunkStarted(index: 0, count: 2, text: "The first sentence passed."))
            out.yield(.audio(samples: Array(repeating: 0.1, count: 44_100), chunkIndex: 0))
            out.yield(.chunkStarted(index: 1, count: 2, text: "[coughs] The second sentence failed."))
            out.finish(throwing: SpeechVerificationError.exhausted(sentence: 2, attempts: 3))
        }
    }
}

@MainActor
final class AppGenerationFailureTests: XCTestCase {
    func testLaterWordCheckFailurePreservesOnlyAnIncompleteTakeAndNoReusableVoice() async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let session = SessionModel(synthesizer: IncompleteSpeechSynthesizer(), library: VoiceLibrary(directory: directory))
        session.playsWhileGenerating = false
        session.script = "The first sentence passed. [coughs] The second sentence failed."
        let voice = VoiceDesign(name: "The Duke", description: "silky theatrical villain")
        session.setVoice(.designed(voice), forSpeaker: 0)
        session.generate()
        let deadline = ContinuousClock.now + .seconds(5)
        while session.state.isGenerating, ContinuousClock.now < deadline { try await Task.sleep(for: .milliseconds(10)) }
        guard case .failed(let message) = session.state else { return XCTFail("Failure became \(session.state)") }
        XCTAssertTrue(message.contains("Sentence 2"))
        XCTAssertEqual(session.takes.count, 1)
        XCTAssertTrue(session.takes[0].isIncomplete)
        XCTAssertEqual(session.takes[0].samples.count, 44_100)
        XCTAssertNil(session.result)
        XCTAssertTrue(session.sessionVoices.isEmpty, "Failed audio must not become the designed voice's reference.")
    }
    func testFailedAuditionStopsPlaybackAndCannotBecomeAReusableVoice() async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let session = SessionModel(synthesizer: IncompleteSpeechSynthesizer(), library: VoiceLibrary(directory: directory))
        session.script = "The first sentence passed. [coughs] The second sentence failed."
        let design = VoiceDesign(name: "The Duke", description: "silky theatrical villain")
        let speaker = try session.prepared.get().speakers[0]
        session.tryLine(session.script, as: design)
        let deadline = ContinuousClock.now + .seconds(5)
        while session.state.isGenerating, ContinuousClock.now < deadline { try await Task.sleep(for: .milliseconds(10)) }
        guard case .failed = session.state else { return XCTFail("Audition did not fail") }
        XCTAssertNil(session.audition)
        XCTAssertTrue(session.auditionAudio.isEmpty)
        XCTAssertFalse(session.isPlayingAudition)
        XCTAssertTrue(session.takes.isEmpty)
        session.useDesign(design, for: speaker)
        XCTAssertTrue(session.sessionVoices.isEmpty)
    }

}

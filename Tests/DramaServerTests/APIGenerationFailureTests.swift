// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import DramaEngine
import DramaServer
import XCTest

private struct WordCheckFailureSynthesizer: SpeechSynthesizer {
    var isLoaded: Bool { true }
    func load(onPhase: @escaping @Sendable (String) -> Void) async throws {}
    func encodeReference(audioURL: URL, transcript: String, name: String) async throws -> ReferenceVoice { throw CancellationError() }
    func decode(frames: [Frame]) async throws -> [Float] { [] }
    func synthesize(_ request: SpeechRequest) -> AsyncThrowingStream<SpeechEvent, Error> {
        AsyncThrowingStream { out in
            out.yield(.chunkStarted(index: 0, count: 2, text: "First sentence."))
            out.yield(.audio(samples: Array(repeating: 0.1, count: 2048), chunkIndex: 0))
            out.finish(throwing: SpeechVerificationError.exhausted(sentence: 2, attempts: 3))
        }
    }
}

final class APIGenerationFailureTests: XCTestCase {
    func testFileFailureDoesNotReturnAudioOrCacheACompletedTake() async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let service = APIService(synthesizer: WordCheckFailureSynthesizer(), voices: VoiceDirectory(url: directory),
                                 token: nil, version: "test", pronunciation: { PronunciationDictionary() })
        let request = HTTPRequest(method: "POST", path: "/v1/audio/speech",
                                  body: Data(#"{"input":"First sentence. [coughs] Second sentence.","response_format":"pcm"}"#.utf8))
        let response = await service.respond(to: request)
        XCTAssertEqual(response.status, .internalServerError)
        XCTAssertNil(response.headers["X-Drama-Take-Id"])
        XCTAssertTrue(String(decoding: response.bodyData ?? Data(), as: UTF8.self).contains("Sentence 2"))
    }

    func testEventStreamEndsWithErrorAndNeverAnnouncesDoneOrCachesTheTake() async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let service = APIService(synthesizer: WordCheckFailureSynthesizer(), voices: VoiceDirectory(url: directory),
                                 token: nil, version: "test", pronunciation: { PronunciationDictionary() })
        let response = await service.respond(to: HTTPRequest(method: "GET", path: "/v1/audio/speech/stream",
                                                             query: ["input": "First sentence. [coughs] Second sentence."]))
        guard case .chunked(let stream) = response.body else { return XCTFail("No event stream") }
        var reader = EventReader(stream)
        var names: [String] = []
        do { while let event = try await reader.next() { names.append(event.name) } }
        catch { XCTAssertTrue(error is SpeechVerificationError) }
        XCTAssertTrue(names.contains("audio"), "An earlier accepted sentence may already have streamed.")
        XCTAssertEqual(names.last, "error")
        XCTAssertFalse(names.contains("done"))
        let id = try XCTUnwrap(response.headers["X-Drama-Take-Id"])
        let take = await service.respond(to: HTTPRequest(method: "GET", path: "/v1/takes/" + id))
        XCTAssertEqual(take.status, .notFound)
    }
}

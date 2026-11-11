// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import DramaEngine
import DramaServer
import Network
import XCTest

private struct WordCheckFailureSynthesizer: SpeechSynthesizer {
    var isLoaded: Bool { true }
    func load(onPhase: @escaping @Sendable (String) -> Void) async throws {}
    var engineDescription: EngineDescription { EngineDescription(engine: .drama3, modelID: "fishaudio/drama-3", snapshot: "test", isMock: true) }
    var isTuned: Bool? { false }
    func encodeVoice(samples: [Float], transcript: String) async throws -> VoiceEncoding { throw CancellationError() }
    func synthesize(_ request: SpeechRequest) -> AsyncThrowingStream<SpeechEvent, Error> {
        AsyncThrowingStream { out in
            out.yield(.chunkStarted(index: 0, count: 2, text: "First sentence."))
            out.yield(.audio(samples: Array(repeating: 0.1, count: 2048), chunkIndex: 0))
            out.finish(throwing: SpeechVerificationError.exhausted(sentence: 2, attempts: 3))
        }
    }
}

private struct UncheckableLanguageSynthesizer: SpeechSynthesizer {
    var isLoaded: Bool { true }
    func load(onPhase: @escaping @Sendable (String) -> Void) async throws {}
    var engineDescription: EngineDescription { EngineDescription(engine: .ming, modelID: "inclusionAI/Ming-3.0-tts-0.5B", snapshot: "test", isMock: true) }
    var isTuned: Bool? { false }
    func encodeVoice(samples: [Float], transcript: String) async throws -> VoiceEncoding { throw CancellationError() }
    func synthesize(_ request: SpeechRequest) -> AsyncThrowingStream<SpeechEvent, Error> {
        AsyncThrowingStream { $0.finish(throwing: SpeechVerificationError.language("this script mixes languages; generate each language separately")) }
    }
}

final class APIGenerationFailureTests: XCTestCase {
    /// The request's language, not the server, is at fault, so the client hears 400 and why.
    func testALanguageTheCheckCannotHearIsABadRequest() async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let service = APIService(synthesizer: UncheckableLanguageSynthesizer(), voices: VoiceDirectory(url: directory),
                                 token: nil, version: "test", pronunciation: { PronunciationDictionary() })
        let response = await service.respond(to: HTTPRequest(method: "POST", path: "/v1/audio/speech",
                                                             body: Data(#"{"input":"Hello there, friend.","response_format":"pcm"}"#.utf8)))
        XCTAssertEqual(response.status, .badRequest)
        XCTAssertTrue(String(decoding: response.bodyData ?? Data(), as: UTF8.self).contains("mixes languages"))
    }

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
        while let event = try await reader.next() { names.append(event.name) }
        XCTAssertTrue(names.contains("audio"), "An earlier accepted sentence may already have streamed.")
        XCTAssertEqual(names.last, "error")
        XCTAssertFalse(names.contains("done"))
        let id = try XCTUnwrap(response.headers["X-Drama-Take-Id"])
        let take = await service.respond(to: HTTPRequest(method: "GET", path: "/v1/takes/" + id))
        XCTAssertEqual(take.status, .notFound)
    }

    /// Catches a failed live session or stream cut off without the chunk
    /// that completes its body, which curl reports as a transfer closed
    /// early. Raw PCM cannot say it failed, so there the incomplete body is
    /// the message.
    func testAFailedEventStreamIsACompleteBodyOverTCPAndFailedPCMIsNot() async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let service = APIService(synthesizer: WordCheckFailureSynthesizer(), voices: VoiceDirectory(url: directory),
                                 token: nil, version: "test", pronunciation: { PronunciationDictionary() })
        let ports = Ports()
        let server = APIServer(service: service, port: 0) { state in
            if case .listening(let port) = state { ports.append(port) }
        }
        server.start()
        defer { server.stop() }
        let deadline = Date().addingTimeInterval(5)
        while ports.first == nil, Date() < deadline { try await Task.sleep(for: .milliseconds(20)) }
        let port = try XCTUnwrap(ports.first)
        let error = "event: error\ndata: {\"message\":\"Sentence 2 did not pass the spoken-word check after 3 attempts. Generation stopped without creating a completed take.\"}\n\n"

        let live = try await exchange(port: port, "POST /v1/audio/speech/live HTTP/1.1\r\nAccept: text/event-stream\r\nContent-Length: 25\r\n\r\nFirst sentence.\nGoodbye.\n")
        XCTAssertTrue(live.contains(error), String(live.suffix(300)))
        XCTAssertTrue(live.hasSuffix("\r\n0\r\n\r\n"), String(live.suffix(60)))

        let stream = try await exchange(port: port, "GET /v1/audio/speech/stream?input=First%20sentence.%20Second%20sentence. HTTP/1.1\r\n\r\n")
        XCTAssertTrue(stream.contains(error), String(stream.suffix(300)))
        XCTAssertTrue(stream.hasSuffix("\r\n0\r\n\r\n"), String(stream.suffix(60)))

        let body = #"{"input":"First sentence. Second sentence.","stream":true}"#
        let pcm = try await exchange(port: port, "POST /v1/audio/speech HTTP/1.1\r\nContent-Length: \(body.utf8.count)\r\n\r\n\(body)")
        XCTAssertTrue(pcm.hasPrefix("HTTP/1.1 200"), String(pcm.prefix(60)))
        XCTAssertFalse(pcm.hasSuffix("\r\n0\r\n\r\n"), "a failed PCM stream must not read as complete")
    }

    /// Everything the server sends for one raw request, until it closes the connection.
    private func exchange(port: UInt16, _ request: String) async throws -> String {
        let connection = NWConnection(host: "127.0.0.1", port: NWEndpoint.Port(rawValue: port)!, using: .tcp)
        defer { connection.cancel() }
        connection.start(queue: .global())
        connection.send(content: Data(request.utf8), completion: .contentProcessed { _ in })
        var received = Data()
        while true {
            let (data, done): (Data?, Bool) = try await withCheckedThrowingContinuation { continuation in
                connection.receive(minimumIncompleteLength: 1, maximumLength: 1 << 16) { data, _, isComplete, error in
                    if let error, data == nil { continuation.resume(throwing: error) } else { continuation.resume(returning: (data, isComplete || error != nil)) }
                }
            }
            if let data { received.append(data) }
            if done { return String(decoding: received, as: UTF8.self) }
        }
    }
}

private final class Ports: @unchecked Sendable {
    private let lock = NSLock()
    private var values: [UInt16] = []
    func append(_ port: UInt16) { lock.withLock { values.append(port) } }
    var first: UInt16? { lock.withLock { values.first } }
}

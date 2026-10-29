// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import DramaEngine
import DramaServer
import XCTest

/// One parsed Server-Sent Event.
struct ServerEvent: Equatable {
    var name: String
    var data: String
    var fields: [String: Any] { (try? JSONSerialization.jsonObject(with: Data(data.utf8))) as? [String: Any] ?? [:] }
    static func == (a: ServerEvent, b: ServerEvent) -> Bool { a.name == b.name && a.data == b.data }
}

/// Reads events off a chunked SSE body as they arrive.
struct EventReader {
    var iterator: AsyncThrowingStream<Data, Error>.AsyncIterator
    var buffer = ""

    init(_ stream: AsyncThrowingStream<Data, Error>) { iterator = stream.makeAsyncIterator() }

    mutating func next() async throws -> ServerEvent? {
        while true {
            if let end = buffer.range(of: "\n\n") {
                let block = buffer[..<end.lowerBound]
                buffer = String(buffer[end.upperBound...])
                var event = ServerEvent(name: "", data: "")
                for line in block.split(separator: "\n") {
                    if line.hasPrefix("event: ") { event.name = String(line.dropFirst(7)) }
                    if line.hasPrefix("data: ") { event.data = String(line.dropFirst(6)) }
                }
                return event
            }
            guard let chunk = try await iterator.next() else { return nil }
            buffer += String(decoding: chunk, as: UTF8.self)
        }
    }

    mutating func next(named name: String) async throws -> ServerEvent? {
        while let event = try await next() {
            if event.name == name { return event }
        }
        return nil
    }
}

final class APIStreamingTests: XCTestCase {
    private var directory: URL!
    private var engine: RecordingSynthesizer!
    private var service: APIService!
    private static let twoSentences = "The first sentence is long enough to stand alone. The second sentence stands alone as well."

    override func setUp() async throws {
        directory = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        engine = RecordingSynthesizer()
        service = APIService(synthesizer: engine, voices: VoiceDirectory(url: directory), token: nil, version: "test", pronunciation: { PronunciationDictionary() })
    }

    override func tearDown() async throws {
        try? FileManager.default.removeItem(at: directory)
    }

    private func body(_ response: APIResponse) throws -> AsyncThrowingStream<Data, Error> {
        guard case .chunked(let stream) = response.body else { throw XCTSkip("not chunked: \(String(decoding: response.bodyData ?? Data(), as: UTF8.self))") }
        return stream
    }

    func testPCMStreamSendsPartialsThatAddUpToTheTake() async throws {
        let response = await service.respond(to: HTTPRequest(method: "POST", path: "/v1/audio/speech",
                                                             body: try JSONSerialization.data(withJSONObject: ["input": Self.twoSentences, "stream": true, "seed": 3])))
        XCTAssertEqual(response.status, .ok)
        XCTAssertNotNil(response.headers["X-Drama-Estimated-Seconds"].flatMap(Double.init))
        var chunks: [Data] = []
        for try await chunk in try body(response) { chunks.append(chunk) }
        let sent = await engine.requests
        XCTAssertEqual(sent.first?.config.streaming, .defaultLowLatency, "stream: true asks the engine for low latency")
        XCTAssertGreaterThan(chunks.count, 2, "audio leaves before each sentence is done")
        XCTAssertTrue(chunks.allSatisfy { $0.count % (ModelConstants.samplesPerFrame * 2) == 0 })

        let file = await service.respond(to: HTTPRequest(method: "POST", path: "/v1/audio/speech",
                                                         body: try JSONSerialization.data(withJSONObject: ["input": Self.twoSentences, "response_format": "pcm", "seed": 3])))
        XCTAssertEqual(chunks.reduce(Data(), +), file.bodyData, "the stream carries exactly the file's audio")
        let fileRequest = await engine.requests.last
        XCTAssertEqual(fileRequest?.config.streaming, .sentence)
    }

    func testEventStreamGetSpeaksTheQueryInput() async throws {
        let response = await service.respond(to: HTTPRequest(method: "GET", path: "/v1/audio/speech/stream",
                                                             query: ["input": Self.twoSentences, "expressiveness": "0.25", "seed": "4"]))
        XCTAssertEqual(response.status, .ok)
        XCTAssertEqual(response.headers["Content-Type"], "text/event-stream")
        var reader = EventReader(try body(response))
        var events: [ServerEvent] = []
        while let event = try await reader.next() { events.append(event) }
        let names = events.map(\.name)
        XCTAssertEqual(names.filter { $0 == "sentence" }.count, 2)
        XCTAssertEqual(names.last, "done")
        XCTAssertEqual(names.first, "audio")
        let audio = events.filter { $0.name == "audio" }.compactMap { Data(base64Encoded: $0.data) }
        XCTAssertTrue(audio.allSatisfy { $0.count % (ModelConstants.samplesPerFrame * 2) == 0 })
        let sentences = events.filter { $0.name == "sentence" }
        XCTAssertEqual(sentences.map { $0.fields["index"] as? Int }, [0, 1])
        XCTAssertEqual(sentences.first?.fields["text"] as? String, "The first sentence is long enough to stand alone.")
        let done = try XCTUnwrap(events.last?.fields)
        XCTAssertEqual(done["take_id"] as? String, response.headers["X-Drama-Take-Id"])
        let config = await engine.requests.first?.config
        XCTAssertEqual(config?.temperature ?? 0, 0.6, accuracy: 1e-9)
        XCTAssertEqual(config?.seed, 4)
    }

    func testEventStreamRejectsAMissingInputAndBadNumbers() async {
        let missing = await service.respond(to: HTTPRequest(method: "GET", path: "/v1/audio/speech/stream"))
        XCTAssertEqual(missing.status, .badRequest)
        let bad = await service.respond(to: HTTPRequest(method: "GET", path: "/v1/audio/speech/stream", query: ["input": "Hi.", "seed": "x"]))
        XCTAssertEqual(bad.status, .badRequest)
        XCTAssertEqual(String(decoding: bad.bodyData ?? Data(), as: UTF8.self), "seed must be a number.\n")
    }

    func testLiveSpeaksEachSentenceOnceItIsCompleteAndCarriesHistory() async throws {
        let (input, feed) = AsyncThrowingStream<Data, Error>.makeStream()
        let response = await service.respond(to: HTTPRequest(method: "POST", path: "/v1/audio/speech/live", headers: ["Accept": "text/event-stream"]), body: input)
        XCTAssertEqual(response.status, .ok)
        var reader = EventReader(try body(response))

        feed.yield(Data("The first sentence arrives in two\n".utf8))
        feed.yield(Data("lines. And the second one".utf8))
        feed.yield(Data(" is still being typed\n".utf8))
        let first = try await reader.next(named: "sentence")
        XCTAssertEqual(first?.fields["text"] as? String, "The first sentence arrives in two lines.")
        let afterFirst = await engine.requests.map(\.text)
        XCTAssertEqual(afterFirst, ["The first sentence arrives in two lines."], "the unfinished sentence waits for more text")

        feed.yield(Data("\n".utf8))
        let second = try await reader.next(named: "sentence")
        XCTAssertEqual(second?.fields["text"] as? String, "And the second one is still being typed")
        XCTAssertEqual(second?.fields["index"] as? Int, 1)

        feed.yield(Data("A last line without a newline.".utf8))
        feed.finish()
        let third = try await reader.next(named: "sentence")
        XCTAssertEqual(third?.fields["text"] as? String, "A last line without a newline.")
        let done = try await reader.next(named: "done")
        XCTAssertEqual(done?.fields["sentences"] as? Int, 3)
        XCTAssertEqual((done?.fields["take_ids"] as? [String])?.count, 3)
        let doneAfterEnd = try await reader.next()
        XCTAssertNil(doneAfterEnd, "the response ends with the input")

        let requests = await engine.requests
        XCTAssertEqual(requests.map(\.previousTakes.count), [0, 1, 2], "each sentence hears the ones before it")
        XCTAssertTrue(requests.allSatisfy { $0.config.streaming == .defaultLowLatency })
        let health = await service.respond(to: HTTPRequest(method: "GET", path: "/v1/health"))
        XCTAssertTrue(String(decoding: health.bodyData ?? Data(), as: UTF8.self).contains("\"busy\":false"))
    }

    func testLiveWithoutEventsStreamsRawPCM() async throws {
        let (input, feed) = AsyncThrowingStream<Data, Error>.makeStream()
        let response = await service.respond(to: HTTPRequest(method: "POST", path: "/v1/audio/speech/live"), body: input)
        XCTAssertEqual(response.headers["Content-Type"], "audio/pcm")
        feed.yield(Data("One whole sentence for the raw stream.\n".utf8))
        feed.finish()
        var bytes = 0
        for try await chunk in try body(response) { bytes += chunk.count }
        XCTAssertGreaterThan(bytes, 0)
        XCTAssertEqual(bytes % (ModelConstants.samplesPerFrame * 2), 0)
    }

    func testHealthReportsTheEstimateWhileBusy() async throws {
        let slow = APIService(synthesizer: MockSynthesizer(pace: .init(loadSeconds: 0, framesPerSecond: 40)), voices: VoiceDirectory(url: directory), token: nil, version: "test")
        let response = await slow.respond(to: HTTPRequest(method: "POST", path: "/v1/audio/speech",
                                                          body: try JSONSerialization.data(withJSONObject: ["input": Self.twoSentences, "stream": true])))
        var iterator = try body(response).makeAsyncIterator()
        _ = try await iterator.next()
        let health = await slow.respond(to: HTTPRequest(method: "GET", path: "/v1/health"))
        let object = try XCTUnwrap(try JSONSerialization.jsonObject(with: health.bodyData ?? Data()) as? [String: Any])
        let estimate = try XCTUnwrap(object["estimate"] as? [String: Any], "\(object)")
        XCTAssertGreaterThan(estimate["frames_done"] as? Int ?? 0, 0)
        XCTAssertEqual(estimate["frames_expected"] as? Int, GenerationEstimate(lines: Drama3Chunking.splitSentencesTagged(Self.twoSentences)).framesExpected)
        XCTAssertGreaterThan(estimate["elapsed_seconds"] as? Double ?? 0, 0)
        XCTAssertNotNil(estimate["remaining_seconds"])
        response.abandon()
        var idle = ""
        let deadline = Date().addingTimeInterval(1)
        while !idle.contains("\"busy\":false"), Date() < deadline {
            try await Task.sleep(for: .milliseconds(20))
            idle = String(decoding: await slow.respond(to: HTTPRequest(method: "GET", path: "/v1/health")).bodyData ?? Data(), as: UTF8.self)
        }
        XCTAssertTrue(idle.contains("\"busy\":false"), idle)
        XCTAssertFalse(idle.contains("estimate"), "no estimate while idle")
    }

    func testLiveTextCutsCompleteSentencesOnly() {
        var text = LiveText()
        XCTAssertEqual(text.append(line: "Short one."), ["Short one."])
        XCTAssertEqual(text.append(line: "This sentence is long enough to stand. Tail"), ["This sentence is long enough to stand."],
                       "a short unfinished tail stays behind instead of folding into the sentence")
        XCTAssertEqual(text.append(line: "ends \u{201C}here.\u{201D} Next"), ["Tail ends \u{201C}here.\u{201D}"])
        XCTAssertEqual(text.append(line: ""), ["Next"], "an empty line flushes")
        XCTAssertEqual(text.flush(), [])
        XCTAssertEqual(text.append(line: "<|speaker:1|> Hello there, my old friend. And"), ["<|speaker:1|> Hello there, my old friend."])
        XCTAssertEqual(text.append(line: "goodbye."), ["<|speaker:1|> And goodbye."], "the tail keeps its speaker")
    }

    func testLineSplitterAndIncrementalBody() throws {
        var lines = LineSplitter()
        XCTAssertEqual(lines.feed(Data("one\r\ntw".utf8)), ["one"])
        XCTAssertEqual(lines.feed(Data("o\nthree".utf8)), ["two"])
        XCTAssertEqual(lines.finish(), ["three"])

        let wire = Data("5\r\nhello\r\n7;ext=1\r\n, world\r\n0\r\nTrailer: x\r\n\r\nGARBAGE".utf8)
        for split in 1 ..< wire.count {
            var chunked = IncrementalBody(.chunked)
            let payload = try chunked.feed(wire.prefix(split)) + chunked.feed(wire.dropFirst(split))
            XCTAssertEqual(payload.reduce(Data(), +), Data("hello, world".utf8), "split at \(split)")
            XCTAssertTrue(chunked.isFinished)
        }
        var length = IncrementalBody(.length(4))
        XCTAssertEqual(try length.feed(Data("ab".utf8)), [Data("ab".utf8)])
        XCTAssertFalse(length.isFinished)
        XCTAssertEqual(try length.feed(Data("cdef".utf8)), [Data("cd".utf8)])
        XCTAssertTrue(length.isFinished)
        var broken = IncrementalBody(.chunked)
        XCTAssertThrowsError(try broken.feed(Data("zz\r\n".utf8)))
    }

    func testParserHandsTheLiveRouteItsHead() {
        let raw = Data("POST /v1/audio/speech/live?voice=Narrator HTTP/1.1\r\nTransfer-Encoding: chunked\r\nExpect: 100-continue\r\n\r\n5\r\nhel".utf8)
        guard case .head(let head) = HTTPRequestParser.parseHead(raw) else { return XCTFail("\(HTTPRequestParser.parseHead(raw))") }
        XCTAssertEqual(head.framing, .chunked)
        XCTAssertTrue(head.expectsContinue)
        XCTAssertEqual(head.request.query["voice"], "Narrator")
        XCTAssertEqual(String(decoding: raw.dropFirst(head.bodyStart), as: UTF8.self), "5\r\nhel")
        XCTAssertTrue(APIRequest.streamsBody(head.request))
        XCTAssertFalse(APIRequest.streamsBody(HTTPRequest(method: "POST", path: "/v1/audio/speech")))
    }
}

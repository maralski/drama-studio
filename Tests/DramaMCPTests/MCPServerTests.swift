// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
@testable import DramaMCP
import XCTest

actor FakeTransport: MCPTransport {
    private var pending: [String]
    private(set) var written: [String] = []

    init(_ lines: [String]) { pending = lines }

    func readLine() -> String? { pending.isEmpty ? nil : pending.removeFirst() }
    func write(_ line: String) { written.append(line) }
}

final class StubAPI: DramaAPI, @unchecked Sendable {
    let baseURL = URL(string: "http://127.0.0.1:9")!
    private let lock = NSLock()
    private var replies: [String: APIReply]
    private var recorded: [APICall] = []

    init(_ replies: [String: APIReply]) { self.replies = replies }

    var calls: [APICall] { lock.withLock { recorded } }

    func send(_ call: APICall) async throws -> APIReply {
        lock.withLock { recorded.append(call) }
        return replies["\(call.method) \(call.path)"] ?? APIReply(status: 404, body: Data("No route.".utf8))
    }
}

final class Played: @unchecked Sendable {
    private let lock = NSLock()
    private var urls: [URL] = []
    func add(_ url: URL) { lock.withLock { urls.append(url) } }
    var all: [URL] { lock.withLock { urls } }
}

final class MCPServerTests: XCTestCase {
    private var output: URL!

    override func setUp() {
        output = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
    }

    override func tearDown() {
        try? FileManager.default.removeItem(at: output)
    }

    private func server(_ api: StubAPI, played: Played = Played()) -> MCPServer {
        MCPServer(context: ToolContext(api: api, outputDirectory: output, now: { Date(timeIntervalSince1970: 0) }, play: { played.add($0) }))
    }

    private func parse(_ line: String?) throws -> JSON {
        try JSON(parsing: Data(try XCTUnwrap(line).utf8))
    }

    private func call(_ server: MCPServer, _ tool: String, _ arguments: JSON, id: JSON = 3) async throws -> JSON {
        let request = JSON.object(["jsonrpc": "2.0", "id": id, "method": "tools/call", "params": ["name": .string(tool), "arguments": arguments]])
        return try parse(await server.handle(request.serialized))
    }

    func testInitializeHandshakeOverTheTransport() async throws {
        let transport = FakeTransport([
            #"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"test","version":"1"}}}"#,
            #"{"jsonrpc":"2.0","method":"notifications/initialized"}"#,
            #"{"jsonrpc":"2.0","id":"two","method":"ping"}"#,
        ])
        await server(StubAPI([:])).run(transport)
        let written = await transport.written
        XCTAssertEqual(written.count, 2, "the notification gets no reply: \(written)")
        XCTAssertTrue(written.allSatisfy { !$0.contains("\n") }, "one message per line")
        let replies = try written.map { try parse($0) }
        let initialize = try XCTUnwrap(replies.first { $0["id"] == 1 })
        XCTAssertEqual(initialize["jsonrpc"], "2.0")
        XCTAssertEqual(initialize["result"]?["protocolVersion"], "2025-06-18")
        XCTAssertEqual(initialize["result"]?["serverInfo"]?["name"], "drama-studio")
        XCTAssertNotNil(initialize["result"]?["capabilities"]?["tools"])
        let ping = try XCTUnwrap(replies.first { $0["id"] == "two" })
        XCTAssertEqual(ping["result"], [:])
    }

    func testToolsListDescribesEveryTool() async throws {
        let reply = try parse(await server(StubAPI([:])).handle(#"{"jsonrpc":"2.0","id":2,"method":"tools/list"}"#))
        guard case .array(let tools)? = reply["result"]?["tools"] else { return XCTFail("\(reply)") }
        XCTAssertEqual(tools.compactMap { $0["name"]?.string }, ["synthesize_speech", "retake_with_note", "list_voices", "add_voice", "delete_voice", "studio_status"])
        for tool in tools {
            XCTAssertEqual(tool["inputSchema"]?["type"], "object", "\(tool)")
            XCTAssertGreaterThan(tool["description"]?.string?.count ?? 0, 150, "\(tool["name"]!)")
        }
        XCTAssertEqual(tools[0]["inputSchema"]?["required"], ["text"])
    }

    func testSynthesizeWritesTheWavAndReportsTimings() async throws {
        let wav = Data("RIFF....WAVE".utf8)
        let api = StubAPI(["POST /v1/audio/speech": APIReply(status: 200, headers: ["x-drama-seconds": "1.950", "X-Drama-Frames": "42", "X-Drama-Generation-Seconds": "2.500"], body: wav)])
        let played = Played()
        let reply = try await call(server(api, played: played), "synthesize_speech", ["text": "Hello.", "voice": "Narrator", "expressiveness": 0.8, "play": true])
        let result = try XCTUnwrap(reply["result"])
        XCTAssertEqual(result["isError"], false)
        let structured = try XCTUnwrap(result["structuredContent"])
        let path = try XCTUnwrap(structured["path"]?.string)
        XCTAssertEqual(path, output.appending(path: "\(Self.localStamp).wav").path)
        XCTAssertEqual(try Data(contentsOf: URL(filePath: path)), wav)
        XCTAssertEqual(structured["seconds"], 1.95)
        XCTAssertEqual(structured["frames"], 42)
        XCTAssertEqual(structured["generation_seconds"], 2.5)
        XCTAssertEqual(played.all.map(\.path), [path])
        guard case .array(let content)? = result["content"] else { return XCTFail() }
        XCTAssertEqual(content.first?["type"], "text")
        XCTAssertEqual(try JSON(parsing: Data(content.first?["text"]?.string?.utf8 ?? "".utf8)), structured)

        let sent = try XCTUnwrap(api.calls.first?.json)
        XCTAssertEqual(sent, ["input": "Hello.", "voice": "Narrator", "expressiveness": 0.8, "response_format": "wav"])
    }

    func testCharacterAndRetakeWithNoteSpeakThroughTheSpeechRoute() async throws {
        let api = StubAPI(["POST /v1/audio/speech": APIReply(status: 200, headers: ["X-Drama-Take-Id": "TAKE-2", "X-Drama-Frames": "10"], body: Data("RIFF".utf8))])
        let character: JSON = ["name": "Captain", "description": "gravelly old male voice"]
        let first = try await call(server(api), "synthesize_speech", ["text": "Ahoy.", "character": character])
        XCTAssertEqual(first["result"]?["structuredContent"]?["take_id"], "TAKE-2")
        XCTAssertEqual(api.calls.last?.json, ["input": "Ahoy.", "character": character, "response_format": "wav"])

        let retake = try await call(server(api), "retake_with_note", ["take_id": "TAKE-1", "note": "angrier", "output_path": .string(output.appending(path: "r.wav").path)])
        XCTAssertEqual(retake["result"]?["isError"], false)
        XCTAssertEqual(retake["result"]?["structuredContent"]?["take_id"], "TAKE-2")
        XCTAssertEqual(api.calls.last?.json, ["note": "angrier", "previous_takes": [["take_id": "TAKE-1"]], "response_format": "wav"])
        XCTAssertEqual(try Data(contentsOf: output.appending(path: "r.wav")), Data("RIFF".utf8))

        let both = try await call(server(api), "synthesize_speech", ["text": "Ahoy.", "character": character, "voice": "Narrator"])
        XCTAssertEqual(both["result"]?["isError"], true)
        let missing = try await call(server(api), "retake_with_note", ["take_id": "TAKE-1", "note": "  "])
        XCTAssertEqual(missing["result"]?["isError"], true)
        XCTAssertEqual(api.calls.count, 2, "invalid calls never reach the API")
    }

    private static var localStamp: String {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "yyyy-MM-dd'T'HH-mm-ss.SSS"
        return formatter.string(from: Date(timeIntervalSince1970: 0))
    }

    func testAPIFailuresBecomeToolErrors() async throws {
        let api = StubAPI(["POST /v1/audio/speech": APIReply(status: 429, body: Data("Drama Studio is already generating.\n".utf8))])
        let busy = try await call(server(api), "synthesize_speech", ["text": "Hi."])
        XCTAssertEqual(busy["result"]?["isError"], true)
        XCTAssertNil(busy["error"], "a failed tool is a result, not a protocol error")
        XCTAssertEqual(busy["result"]?["content"], [["type": "text", "text": "Drama Studio answered 429: Drama Studio is already generating."]])

        let missing = try await call(server(api), "synthesize_speech", [:])
        XCTAssertEqual(missing["result"]?["isError"], true)
        XCTAssertTrue(api.calls.count == 1, "missing text never reaches the API")
    }

    func testVoiceToolsProxyTheirRoutes() async throws {
        let clip = output.appending(path: "clip.wav")
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        try Data([1, 2, 3]).write(to: clip)
        let api = StubAPI([
            "GET /v1/voices": APIReply(status: 200, body: Data(#"{"voices":[{"id":"A","name":"Narrator"}]}"#.utf8)),
            "POST /v1/voices": APIReply(status: 201, body: Data(#"{"id":"B","name":"New"}"#.utf8)),
            "DELETE /v1/voices/B": APIReply(status: 200, body: Data(#"{"deleted":"B"}"#.utf8)),
            "GET /v1/health": APIReply(status: 200, body: Data(#"{"status":"ok","engine":"mock","busy":false}"#.utf8)),
            "GET /v1/models": APIReply(status: 200, body: Data(#"{"object":"list","data":[{"id":"fishaudio/drama-3","snapshot":"abc"}]}"#.utf8)),
        ])
        let mcp = server(api)
        let listed = try await call(mcp, "list_voices", [:])
        XCTAssertEqual(listed["result"]?["structuredContent"]?["voices"], [["id": "A", "name": "Narrator"]])
        let added = try await call(mcp, "add_voice", ["audio_path": .string(clip.path), "transcript": "One two three.", "name": "New"])
        XCTAssertEqual(added["result"]?["structuredContent"]?["id"], "B")
        XCTAssertEqual(api.calls.last?.json, ["name": "New", "transcript": "One two three.", "audio_base64": "AQID"])
        let deleted = try await call(mcp, "delete_voice", ["id": "B"])
        XCTAssertEqual(deleted["result"]?["structuredContent"], ["deleted": "B"])
        let status = try await call(mcp, "studio_status", [:])
        XCTAssertEqual(status["result"]?["structuredContent"]?["engine"], "mock")
        XCTAssertEqual(status["result"]?["structuredContent"]?["model"]?["snapshot"], "abc")
        XCTAssertEqual(status["result"]?["structuredContent"]?["api_url"], "http://127.0.0.1:9")
        let unreadable = try await call(mcp, "add_voice", ["audio_path": "/nonexistent.wav", "transcript": "x", "name": "y"])
        XCTAssertEqual(unreadable["result"]?["isError"], true)
    }

    func testProtocolErrors() async throws {
        let mcp = server(StubAPI([:]))
        let malformed = try parse(await mcp.handle("{\"jsonrpc\":\"2.0\",\"id\":1,"))
        XCTAssertEqual(malformed["error"]?["code"], -32700)
        XCTAssertEqual(malformed["id"], .null)
        let invalid = try parse(await mcp.handle(#"{"jsonrpc":"1.0","id":5,"method":"ping"}"#))
        XCTAssertEqual(invalid["error"]?["code"], -32600)
        XCTAssertEqual(invalid["id"], 5)
        let unknownMethod = try parse(await mcp.handle(#"{"jsonrpc":"2.0","id":6,"method":"resources/list"}"#))
        XCTAssertEqual(unknownMethod["error"]?["code"], -32601)
        let unknownTool = try await call(mcp, "sing", [:], id: "x")
        XCTAssertEqual(unknownTool["error"]?["code"], -32602)
        XCTAssertEqual(unknownTool["id"], "x")
        let unknownNotification = await mcp.handle(#"{"jsonrpc":"2.0","method":"notifications/cancelled","params":{"requestId":1}}"#)
        XCTAssertNil(unknownNotification)
    }

    func testToolCallsTakeTurnsWhilePingAnswersAtOnce() async throws {
        final class SlowAPI: DramaAPI, @unchecked Sendable {
            let baseURL = URL(string: "http://127.0.0.1:9")!
            private let lock = NSLock()
            private var inFlight = 0
            private(set) var maxInFlight = 0
            func send(_ call: APICall) async throws -> APIReply {
                lock.withLock { inFlight += 1; maxInFlight = max(maxInFlight, inFlight) }
                try await Task.sleep(for: .milliseconds(200))
                lock.withLock { inFlight -= 1 }
                return APIReply(status: 200, body: Data(#"{"voices":[]}"#.utf8))
            }
        }
        let api = SlowAPI()
        let transport = FakeTransport([
            #"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"list_voices","arguments":{}}}"#,
            #"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"list_voices","arguments":{}}}"#,
            #"{"jsonrpc":"2.0","id":3,"method":"ping"}"#,
        ])
        await MCPServer(context: ToolContext(api: api, outputDirectory: output)).run(transport)
        let ids = try await transport.written.map { try parse($0)["id"] }
        XCTAssertEqual(api.maxInFlight, 1)
        XCTAssertEqual(ids.first, 3, "ping does not wait for the tools: \(ids)")
        XCTAssertEqual(ids.count, 3)
        XCTAssertTrue(ids.contains(1) && ids.contains(2), "\(ids)")
    }

    func testSerializationIsOneLineAndRoundTrips() throws {
        let value: JSON = ["text": "say \"hi\"\n\tthen\u{1}stop \\ ok", "seconds": 2.354, "frames": 42, "ok": true, "none": nil, "list": [1, "two"]]
        let line = value.serialized
        XCTAssertFalse(line.contains("\n"))
        XCTAssertTrue(line.contains("\"seconds\":2.354"), line)
        XCTAssertEqual(try JSON(parsing: Data(line.utf8)), value)
    }

    func testAppCandidatesPreferTheEnvironmentThenTheEnclosingBundle() {
        let inBundle = HTTPDramaAPI.appCandidates(environment: [:], executable: URL(filePath: "/Users/x/Apps/DramaStudio.app/Contents/MacOS/drama-mcp"))
        XCTAssertEqual(inBundle.map(\.path), ["/Users/x/Apps/DramaStudio.app", "/Applications/DramaStudio.app"])
        let fromBuild = HTTPDramaAPI.appCandidates(environment: ["DRAMA_APP_PATH": "/tmp/D.app"], executable: URL(filePath: "/repo/.build/arm64-apple-macosx/release/drama-mcp"))
        XCTAssertEqual(fromBuild.map(\.path), ["/tmp/D.app", "/Applications/DramaStudio.app", "/repo/dist/DramaStudio.app"])
    }
}

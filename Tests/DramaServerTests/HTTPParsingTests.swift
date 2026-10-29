// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import DramaServer
import XCTest

final class HTTPParsingTests: XCTestCase {
    private func raw(_ text: String, body: Data = Data()) -> Data { Data(text.utf8) + body }

    func testParsesRequestLineHeadersQueryAndBody() {
        let body = Data("{\"input\":\"hi\"}".utf8)
        let buffer = raw("POST /v1/audio/speech?x=1&name=a%20b HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: \(body.count)\r\n\r\n", body: body)
        guard case .complete(let request) = HTTPRequestParser.parse(buffer) else { return XCTFail("\(HTTPRequestParser.parse(buffer))") }
        XCTAssertEqual(request.method, "POST")
        XCTAssertEqual(request.path, "/v1/audio/speech")
        XCTAssertEqual(request.query, ["x": "1", "name": "a b"])
        XCTAssertEqual(request.header("CONTENT-TYPE"), "application/json")
        XCTAssertEqual(request.body, body)
    }

    func testWaitsForTheWholeBodyAndReportsWhatTheHeadPromised() {
        let head = "POST /v1/voices HTTP/1.1\r\nContent-Length: 10\r\nExpect: 100-continue\r\n\r\n"
        XCTAssertEqual(HTTPRequestParser.parse(raw("POST /v1/voi")), .incomplete(nil))
        XCTAssertEqual(HTTPRequestParser.parse(raw(head, body: Data("12345".utf8))), .incomplete(HTTPHead(totalLength: head.utf8.count + 10, expectsContinue: true)))
        guard case .complete(let request) = HTTPRequestParser.parse(raw(head, body: Data("1234567890".utf8))) else { return XCTFail() }
        XCTAssertEqual(request.body.count, 10)
    }

    func testRejectsWhatTheServerDoesNotSpeak() {
        let tooBig = HTTPRequestParser.maxBodyBytes + 1
        XCTAssertEqual(HTTPRequestParser.parse(raw("POST / HTTP/1.1\r\nContent-Length: \(tooBig)\r\n\r\n")), .invalid(.payloadTooLarge, "The request body is larger than 64 MB."))
        guard case .invalid(.notImplemented, _) = HTTPRequestParser.parse(raw("POST / HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n")) else { return XCTFail() }
        guard case .invalid(.badRequest, _) = HTTPRequestParser.parse(raw("GARBAGE\r\n\r\n")) else { return XCTFail() }
        guard case .invalid(.badRequest, _) = HTTPRequestParser.parse(raw("GET / HTTP/1.1\r\nContent-Length: ten\r\n\r\n")) else { return XCTFail() }
        guard case .invalid(.headersTooLarge, _) = HTTPRequestParser.parse(Data(repeating: 65, count: HTTPRequestParser.maxHeaderBytes + 1)) else { return XCTFail() }
    }

    func testMultipartSplitsFileAndFields() {
        let boundary = "XyZ"
        let file = Data([0, 1, 2, 13, 10, 3])
        var body = Data()
        body += Data("--XyZ\r\nContent-Disposition: form-data; name=\"name\"\r\n\r\nNarrator\r\n".utf8)
        body += Data("--XyZ\r\nContent-Disposition: form-data; name=\"file\"; filename=\"clip.wav\"\r\nContent-Type: audio/wav\r\n\r\n".utf8) + file
        body += Data("\r\n--XyZ--\r\n".utf8)
        XCTAssertEqual(Multipart.boundary(contentType: "multipart/form-data; boundary=\"XyZ\""), boundary)
        XCTAssertNil(Multipart.boundary(contentType: "application/json"))
        let parts = Multipart.parse(body, boundary: boundary)
        XCTAssertEqual(parts?.map(\.name), ["name", "file"])
        XCTAssertEqual(parts?[0].text, "Narrator")
        XCTAssertEqual(parts?[1].filename, "clip.wav")
        XCTAssertEqual(parts?[1].contentType, "audio/wav")
        XCTAssertEqual(parts?[1].data, file)
        XCTAssertNil(Multipart.parse(Data("no boundary here".utf8), boundary: boundary))
    }

    func testChunkFramingAndResponseHead() {
        XCTAssertEqual(ChunkedEncoding.frame(Data(repeating: 7, count: 26)), Data("1a\r\n".utf8) + Data(repeating: 7, count: 26) + Data("\r\n".utf8))
        XCTAssertEqual(ChunkedEncoding.frame(Data()), Data(), "an empty chunk would end the body early")
        XCTAssertEqual(ChunkedEncoding.terminator, Data("0\r\n\r\n".utf8))
        let fixed = String(decoding: APIResponse.error(APIError(.badRequest, "nope")).head, as: UTF8.self)
        XCTAssertTrue(fixed.hasPrefix("HTTP/1.1 400 Bad Request\r\n"), fixed)
        XCTAssertTrue(fixed.contains("Content-Length: 5\r\n"), fixed)
        XCTAssertTrue(fixed.contains("Connection: close\r\n"), fixed)
        XCTAssertTrue(fixed.hasSuffix("\r\n\r\n"))
        let streamed = String(decoding: APIResponse(status: .ok, body: .chunked(AsyncThrowingStream { $0.finish() })).head, as: UTF8.self)
        XCTAssertTrue(streamed.contains("Transfer-Encoding: chunked\r\n"), streamed)
        XCTAssertFalse(streamed.contains("Content-Length"), streamed)
    }

    func testPCMIsSixteenBitLittleEndianAndClipped() {
        let pcm = AudioEncoding.pcm16([0, 1, -1, 2, -2, 0.5])
        let values = stride(from: 0, to: pcm.count, by: 2).map { Int16(bitPattern: UInt16(pcm[$0]) | UInt16(pcm[$0 + 1]) << 8) }
        XCTAssertEqual(values, [0, 32767, -32767, 32767, -32767, 16384])
        let wav = AudioEncoding.wav([0, 0.25])
        XCTAssertEqual(wav.count, 44 + 4)
        XCTAssertEqual(String(decoding: wav.prefix(4), as: UTF8.self), "RIFF")
        XCTAssertEqual(wav[24..<28].withUnsafeBytes { $0.loadUnaligned(as: UInt32.self) }, 44_100)
        XCTAssertEqual(wav[34], 16)
        XCTAssertEqual(AudioEncoding.fileExtension(sniffing: wav), "wav")
        XCTAssertEqual(AudioEncoding.fileExtension(sniffing: Data([0, 0, 0, 0x20]) + Data("ftypM4A ".utf8)), "m4a")
    }

    func testTokenGuardsEverythingButHealth() {
        let token = "secret"
        let voices = HTTPRequest(method: "GET", path: "/v1/voices")
        XCTAssertTrue(APIToken.authorizes(voices, token: nil))
        XCTAssertFalse(APIToken.authorizes(voices, token: token))
        XCTAssertFalse(APIToken.authorizes(HTTPRequest(method: "GET", path: "/v1/voices", headers: ["Authorization": "Bearer secreT"]), token: token))
        XCTAssertFalse(APIToken.authorizes(HTTPRequest(method: "GET", path: "/v1/voices", headers: ["Authorization": "Basic secret"]), token: token))
        XCTAssertTrue(APIToken.authorizes(HTTPRequest(method: "GET", path: "/v1/voices", headers: ["authorization": "Bearer secret"]), token: token))
        XCTAssertTrue(APIToken.authorizes(HTTPRequest(method: "GET", path: "/v1/health"), token: token))
    }

    func testTokenFileIsCreatedOnceAndPrivate() throws {
        let url = FileManager.default.temporaryDirectory.appending(path: "\(UUID().uuidString)/api-token")
        defer { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }
        let first = try APIToken.loadOrCreate(at: url)
        XCTAssertEqual(first.count, 64)
        XCTAssertEqual(try APIToken.loadOrCreate(at: url), first)
        let permissions = try FileManager.default.attributesOfItem(atPath: url.path)[.posixPermissions] as? Int
        XCTAssertEqual(permissions, 0o600)
    }
}

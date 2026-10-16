import DramaEngine
import DramaServer
import XCTest

/// One pass over a real socket, so the listener, the body accumulation, and
/// chunked framing are exercised the way curl sees them.
final class APIServerSocketTests: XCTestCase {
    func testServesHealthAndStreamsOverTCP() async throws {
        let directory = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let states = Seen()
        let service = APIService(synthesizer: MockSynthesizer(pace: .instant), voices: VoiceDirectory(url: directory), token: "abc", version: "test")
        let server = APIServer(service: service, port: 0) { state in
            if case .listening(let port) = state { states.append(String(port)) } else { states.append("\(state)") }
        }
        server.start()
        let deadline = Date().addingTimeInterval(5)
        var port: String?
        while port == nil, Date() < deadline {
            port = states.values.compactMap { $0 }.first { Int($0) != nil }
            try await Task.sleep(for: .milliseconds(20))
        }
        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = 10
        let session = URLSession(configuration: configuration)
        let base = try XCTUnwrap(port.map { "http://127.0.0.1:\($0)" }, "\(states.values)")

        let (healthData, healthResponse) = try await session.data(from: URL(string: base + "/v1/health")!)
        XCTAssertEqual((healthResponse as? HTTPURLResponse)?.statusCode, 200)
        XCTAssertTrue(String(decoding: healthData, as: UTF8.self).contains("\"engine\":\"mock\""))

        var request = URLRequest(url: URL(string: base + "/v1/audio/speech")!)
        request.httpMethod = "POST"
        request.setValue("Bearer abc", forHTTPHeaderField: "Authorization")
        let text = String(repeating: "This sentence is about forty bytes long. ", count: 10)
        request.httpBody = try JSONSerialization.data(withJSONObject: ["input": text, "stream": true, "response_format": "pcm"])
        let (pcm, streamResponse) = try await session.data(for: request)
        let http = try XCTUnwrap(streamResponse as? HTTPURLResponse)
        XCTAssertEqual(http.statusCode, 200)
        XCTAssertEqual(http.value(forHTTPHeaderField: "X-Drama-Format"), "pcm_s16le_mono")
        XCTAssertGreaterThan(pcm.count, 0)
        XCTAssertEqual(pcm.count % (ModelConstants.samplesPerFrame * 2), 0)

        let bigBody = Data(repeating: 0x20, count: 3 << 20)
        var big = URLRequest(url: URL(string: base + "/v1/voices")!)
        big.httpMethod = "POST"
        big.setValue("Bearer abc", forHTTPHeaderField: "Authorization")
        big.httpBody = bigBody
        let (bigData, bigResponse) = try await session.data(for: big)
        XCTAssertEqual((bigResponse as? HTTPURLResponse)?.statusCode, 400, String(decoding: bigData, as: UTF8.self))

        server.stop()
        try await Task.sleep(for: .milliseconds(100))
        XCTAssertEqual(states.values.last, "stopped")
    }

    func testHangingUpBeforeTheFirstChunkFreesTheEngine() async throws {
        let directory = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let ports = Seen()
        let service = APIService(synthesizer: MockSynthesizer(pace: .init(loadSeconds: 0, framesPerSecond: 20)), voices: VoiceDirectory(url: directory), token: nil, version: "test")
        let server = APIServer(service: service, port: 0) { state in
            if let port = state.port { ports.append(String(port)) }
        }
        server.start()
        defer { server.stop() }
        let deadline = Date().addingTimeInterval(5)
        while ports.values.isEmpty, Date() < deadline { try await Task.sleep(for: .milliseconds(20)) }
        let base = "http://127.0.0.1:\(try XCTUnwrap(ports.values.first ?? nil))"

        func busy() async throws -> Bool {
            let (data, _) = try await URLSession.shared.data(from: URL(string: base + "/v1/health")!)
            return String(decoding: data, as: UTF8.self).contains("\"busy\":true")
        }

        for stream in [true, false] {
            var request = URLRequest(url: URL(string: base + "/v1/audio/speech")!)
            request.httpMethod = "POST"
            request.httpBody = try JSONSerialization.data(withJSONObject: ["input": "A sentence that takes the slow engine several seconds to speak.", "stream": stream])
            let call = Task { try await URLSession.shared.data(for: request) }
            try await Task.sleep(for: .milliseconds(300))
            let busyWhileRunning = try await busy()
            XCTAssertTrue(busyWhileRunning, "stream \(stream)")
            call.cancel()
            var stillBusy = true
            let until = Date().addingTimeInterval(1)
            while stillBusy, Date() < until {
                try await Task.sleep(for: .milliseconds(20))
                stillBusy = try await busy()
            }
            XCTAssertFalse(stillBusy, "stream \(stream): the slot frees once the client hangs up")
        }
    }
}

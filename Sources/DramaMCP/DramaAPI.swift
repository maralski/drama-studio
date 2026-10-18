import Foundation

public struct APICall: Sendable, Equatable {
    public var method: String
    public var path: String
    public var json: JSON?

    public init(_ method: String, _ path: String, json: JSON? = nil) {
        self.method = method
        self.path = path
        self.json = json
    }
}

public struct APIReply: Sendable, Equatable {
    public var status: Int
    public var headers: [String: String]
    public var body: Data

    public init(status: Int, headers: [String: String] = [:], body: Data = Data()) {
        self.status = status
        self.headers = headers
        self.body = body
    }

    public var ok: Bool { (200..<300).contains(status) }
    public var text: String { String(decoding: body, as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines) }
    public func header(_ name: String) -> String? { headers.first { $0.key.caseInsensitiveCompare(name) == .orderedSame }?.value }
}

public protocol DramaAPI: Sendable {
    var baseURL: URL { get }
    func send(_ call: APICall) async throws -> APIReply
    /// The body as it arrives. A success reply comes with an empty body and the
    /// audio in `chunks`; a failure reply carries its whole body and no chunks.
    func bytes(for call: APICall) async throws -> (reply: APIReply, chunks: AsyncThrowingStream<Data, any Error>)
}

extension DramaAPI {
    public func bytes(for call: APICall) async throws -> (reply: APIReply, chunks: AsyncThrowingStream<Data, any Error>) {
        let reply = try await send(call)
        guard reply.ok else { return (reply, AsyncThrowingStream { $0.finish() }) }
        return (APIReply(status: reply.status, headers: reply.headers), AsyncThrowingStream { $0.yield(reply.body); $0.finish() })
    }
}

public struct APIUnavailable: Error, LocalizedError {
    public var message: String
    public var errorDescription: String? { message }
}

public struct HTTPDramaAPI: DramaAPI {
    public let baseURL: URL
    public let tokenURL: URL
    public let appCandidates: [URL]
    public let launchTimeout: Duration
    private let programName: String

    public static var defaultTokenURL: URL {
        URL.applicationSupportDirectory.appending(path: "DramaStudio/api-token")
    }

    public init(environment: [String: String] = ProcessInfo.processInfo.environment, executable: URL = URL(filePath: CommandLine.arguments[0]).resolvingSymlinksInPath()) {
        baseURL = URL(string: environment["DRAMA_API_URL"] ?? "http://127.0.0.1:8780")!
        tokenURL = Self.defaultTokenURL
        appCandidates = Self.appCandidates(environment: environment, executable: executable)
        launchTimeout = .seconds(90)
        programName = executable.lastPathComponent
    }

    static func appCandidates(environment: [String: String], executable: URL) -> [URL] {
        var candidates: [URL] = []
        if let path = environment["DRAMA_APP_PATH"] { candidates.append(URL(filePath: path)) }
        let macOS = executable.deletingLastPathComponent()
        if macOS.lastPathComponent == "MacOS", macOS.deletingLastPathComponent().lastPathComponent == "Contents" {
            candidates.append(macOS.deletingLastPathComponent().deletingLastPathComponent())
        }
        candidates.append(URL(filePath: "/Applications/DramaStudio.app"))
        var directory = macOS
        while directory.path != "/" {
            if directory.lastPathComponent == ".build" {
                candidates.append(directory.deletingLastPathComponent().appending(path: "dist/DramaStudio.app"))
                break
            }
            directory = directory.deletingLastPathComponent()
        }
        return candidates
    }

    public func send(_ call: APICall) async throws -> APIReply {
        try await launchingIfNeeded { try await perform(call) }
    }

    public func bytes(for call: APICall) async throws -> (reply: APIReply, chunks: AsyncThrowingStream<Data, any Error>) {
        let (bytes, response) = try await launchingIfNeeded { try await URLSession.shared.bytes(for: request(call)) }
        var reply = Self.reply(response, body: Data())
        guard reply.ok else {
            for try await byte in bytes { reply.body.append(byte) }
            return (reply, AsyncThrowingStream { $0.finish() })
        }
        // The server sends whole 2048-sample frames, so a 4096-byte chunk never
        // waits on audio that is still generating.
        let chunks = AsyncThrowingStream<Data, any Error> { continuation in
            let task = Task {
                do {
                    var chunk = Data(capacity: 4096)
                    for try await byte in bytes {
                        chunk.append(byte)
                        if chunk.count == 4096 {
                            continuation.yield(chunk)
                            chunk = Data(capacity: 4096)
                        }
                    }
                    if !chunk.isEmpty { continuation.yield(chunk) }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
        return (reply, chunks)
    }

    private func launchingIfNeeded<T>(_ attempt: () async throws -> T) async throws -> T {
        do {
            return try await attempt()
        } catch let error as URLError where [.cannotConnectToHost, .cannotFindHost].contains(error.code) {
            try await launchAndWait()
            return try await attempt()
        }
    }

    /// A file response sends nothing until the whole script is generated, and
    /// the timeout counts idle time, so it allows for an hour-long script.
    private func request(_ call: APICall, timeout: TimeInterval = 3600) -> URLRequest {
        var request = URLRequest(url: baseURL.appending(path: call.path), timeoutInterval: timeout)
        request.httpMethod = call.method
        if let token = try? String(contentsOf: tokenURL, encoding: .utf8).trimmingCharacters(in: .whitespacesAndNewlines), !token.isEmpty {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
        if let json = call.json {
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            request.httpBody = Data(json.serialized.utf8)
        }
        return request
    }

    private func perform(_ call: APICall, timeout: TimeInterval = 3600) async throws -> APIReply {
        let (data, response) = try await URLSession.shared.data(for: request(call, timeout: timeout))
        return Self.reply(response, body: data)
    }

    private static func reply(_ response: URLResponse, body: Data) -> APIReply {
        let http = response as? HTTPURLResponse
        let headers = (http?.allHeaderFields ?? [:]).reduce(into: [String: String]()) { result, pair in
            result[String(describing: pair.key)] = String(describing: pair.value)
        }
        return APIReply(status: http?.statusCode ?? 0, headers: headers, body: body)
    }

    private func launchAndWait() async throws {
        guard let app = appCandidates.first(where: { FileManager.default.fileExists(atPath: $0.path) }) else {
            throw APIUnavailable(message: "Nothing answers at \(baseURL.absoluteString) and Drama Studio is not installed at \(appCandidates.map(\.path).joined(separator: ", ")). Set DRAMA_APP_PATH to the app.")
        }
        FileHandle.standardError.write(Data("\(programName): opening \(app.path)\n".utf8))
        let open = Process()
        open.executableURL = URL(filePath: "/usr/bin/open")
        open.arguments = ["-a", app.path]
        try open.run()
        open.waitUntilExit()
        let deadline = ContinuousClock.now + launchTimeout
        while ContinuousClock.now < deadline {
            if let reply = try? await perform(APICall("GET", "/v1/health"), timeout: 2), reply.ok { return }
            try await Task.sleep(for: .seconds(1))
        }
        throw APIUnavailable(message: "Opened \(app.path), but its API did not answer at \(baseURL.absoluteString) within 90 seconds. Check that the API server is on in Drama Studio's menu bar.")
    }
}

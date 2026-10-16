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

    public static var defaultTokenURL: URL {
        URL.applicationSupportDirectory.appending(path: "DramaStudio/api-token")
    }

    public init(environment: [String: String] = ProcessInfo.processInfo.environment, executable: URL = URL(filePath: CommandLine.arguments[0]).resolvingSymlinksInPath()) {
        baseURL = URL(string: environment["DRAMA_API_URL"] ?? "http://127.0.0.1:8765")!
        tokenURL = Self.defaultTokenURL
        appCandidates = Self.appCandidates(environment: environment, executable: executable)
        launchTimeout = .seconds(90)
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
        do {
            return try await perform(call)
        } catch let error as URLError where [.cannotConnectToHost, .cannotFindHost, .networkConnectionLost].contains(error.code) {
            try await launchAndWait()
            return try await perform(call)
        }
    }

    private func perform(_ call: APICall, timeout: TimeInterval = 600) async throws -> APIReply {
        var request = URLRequest(url: baseURL.appending(path: call.path), timeoutInterval: timeout)
        request.httpMethod = call.method
        if let token = try? String(contentsOf: tokenURL, encoding: .utf8).trimmingCharacters(in: .whitespacesAndNewlines), !token.isEmpty {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
        if let json = call.json {
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            request.httpBody = Data(json.serialized.utf8)
        }
        let (data, response) = try await URLSession.shared.data(for: request)
        let http = response as? HTTPURLResponse
        let headers = (http?.allHeaderFields ?? [:]).reduce(into: [String: String]()) { result, pair in
            result[String(describing: pair.key)] = String(describing: pair.value)
        }
        return APIReply(status: http?.statusCode ?? 0, headers: headers, body: data)
    }

    private func launchAndWait() async throws {
        guard let app = appCandidates.first(where: { FileManager.default.fileExists(atPath: $0.path) }) else {
            throw APIUnavailable(message: "Nothing answers at \(baseURL.absoluteString) and Drama Studio is not installed at \(appCandidates.map(\.path).joined(separator: ", ")). Set DRAMA_APP_PATH to the app.")
        }
        FileHandle.standardError.write(Data("drama-mcp: opening \(app.path)\n".utf8))
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

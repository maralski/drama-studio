import Foundation

public enum HTTPStatus: Int, Sendable {
    case ok = 200
    case created = 201
    case badRequest = 400
    case unauthorized = 401
    case notFound = 404
    case methodNotAllowed = 405
    case payloadTooLarge = 413
    case tooManyRequests = 429
    case headersTooLarge = 431
    case internalServerError = 500
    case notImplemented = 501

    public var reason: String {
        switch self {
        case .ok: "OK"
        case .created: "Created"
        case .badRequest: "Bad Request"
        case .unauthorized: "Unauthorized"
        case .notFound: "Not Found"
        case .methodNotAllowed: "Method Not Allowed"
        case .payloadTooLarge: "Payload Too Large"
        case .tooManyRequests: "Too Many Requests"
        case .headersTooLarge: "Request Header Fields Too Large"
        case .internalServerError: "Internal Server Error"
        case .notImplemented: "Not Implemented"
        }
    }
}

/// Header names are lowercased.
public struct HTTPRequest: Sendable, Equatable {
    public var method: String
    public var path: String
    public var query: [String: String]
    public var headers: [String: String]
    public var body: Data

    public init(method: String, path: String, query: [String: String] = [:], headers: [String: String] = [:], body: Data = Data()) {
        self.method = method
        self.path = path
        self.query = query
        self.headers = Dictionary(headers.map { ($0.key.lowercased(), $0.value) }) { a, b in a + ", " + b }
        self.body = body
    }

    public func header(_ name: String) -> String? { headers[name.lowercased()] }
}

public struct HTTPHead: Sendable, Equatable {
    public var totalLength: Int
    public var expectsContinue: Bool

    public init(totalLength: Int, expectsContinue: Bool) {
        self.totalLength = totalLength
        self.expectsContinue = expectsContinue
    }
}

public enum HTTPParse: Sendable, Equatable {
    /// Nil head while the header block is still arriving.
    case incomplete(HTTPHead?)
    case complete(HTTPRequest)
    case invalid(HTTPStatus, String)
}

public enum HTTPRequestParser {
    public static let maxHeaderBytes = 64 * 1024
    public static let maxBodyBytes = 64 * 1024 * 1024
    private static let headerEnd = Data("\r\n\r\n".utf8)

    public static func parse(_ buffer: Data) -> HTTPParse {
        guard let end = buffer.range(of: headerEnd) else {
            return buffer.count > maxHeaderBytes ? .invalid(.headersTooLarge, "The request headers are larger than 64 KB.") : .incomplete(nil)
        }
        let headBytes = buffer[buffer.startIndex..<end.lowerBound]
        guard headBytes.count <= maxHeaderBytes else { return .invalid(.headersTooLarge, "The request headers are larger than 64 KB.") }
        guard let headText = String(data: headBytes, encoding: .utf8) ?? String(data: headBytes, encoding: .isoLatin1) else {
            return .invalid(.badRequest, "The request head is not text.")
        }
        var lines = headText.components(separatedBy: "\r\n")
        let requestLine = lines.removeFirst().split(separator: " ", omittingEmptySubsequences: true)
        guard requestLine.count == 3, requestLine[2].hasPrefix("HTTP/1.") else {
            return .invalid(.badRequest, "Malformed request line.")
        }
        var headers: [String: String] = [:]
        for line in lines where !line.isEmpty {
            guard let colon = line.firstIndex(of: ":") else { return .invalid(.badRequest, "Malformed header line: \(line)") }
            let name = line[..<colon].trimmingCharacters(in: .whitespaces).lowercased()
            let value = line[line.index(after: colon)...].trimmingCharacters(in: .whitespaces)
            headers[name] = headers[name].map { $0 + ", " + value } ?? value
        }
        if headers["transfer-encoding"] != nil {
            return .invalid(.notImplemented, "Chunked request bodies are not supported; send Content-Length.")
        }
        let length: Int
        if let declared = headers["content-length"] {
            guard let value = Int(declared), value >= 0 else { return .invalid(.badRequest, "Content-Length is not a number.") }
            length = value
        } else {
            length = 0
        }
        guard length <= maxBodyBytes else { return .invalid(.payloadTooLarge, "The request body is larger than 64 MB.") }
        let bodyStart = end.upperBound
        let total = bodyStart - buffer.startIndex + length
        guard buffer.count >= total else {
            return .incomplete(HTTPHead(totalLength: total, expectsContinue: headers["expect"]?.lowercased() == "100-continue"))
        }
        guard let target = URLComponents(string: String(requestLine[1])) else { return .invalid(.badRequest, "Malformed request target.") }
        var query: [String: String] = [:]
        for item in target.queryItems ?? [] { query[item.name] = item.value ?? "" }
        let body = Data(buffer[bodyStart..<(bodyStart + length)])
        return .complete(HTTPRequest(method: String(requestLine[0]).uppercased(), path: target.path, query: query, headers: headers, body: body))
    }
}

public enum ChunkedEncoding {
    public static let terminator = Data("0\r\n\r\n".utf8)

    /// An empty payload frames to nothing, because an empty chunk ends the body.
    public static func frame(_ payload: Data) -> Data {
        guard !payload.isEmpty else { return Data() }
        var framed = Data(String(payload.count, radix: 16).utf8)
        framed.append(contentsOf: [13, 10])
        framed.append(payload)
        framed.append(contentsOf: [13, 10])
        return framed
    }
}

public struct MultipartPart: Sendable, Equatable {
    public var name: String
    public var filename: String?
    public var contentType: String?
    public var data: Data

    public var text: String? { String(data: data, encoding: .utf8) }
}

public enum Multipart {
    public static func boundary(contentType: String) -> String? {
        let parts = contentType.split(separator: ";").map { $0.trimmingCharacters(in: .whitespaces) }
        guard parts.first?.lowercased() == "multipart/form-data" else { return nil }
        for parameter in parts.dropFirst() where parameter.lowercased().hasPrefix("boundary=") {
            return String(parameter.dropFirst("boundary=".count)).trimmingCharacters(in: CharacterSet(charactersIn: "\""))
        }
        return nil
    }

    /// Nil when the body does not follow the boundary structure.
    public static func parse(_ body: Data, boundary: String) -> [MultipartPart]? {
        let delimiter = Data("--\(boundary)".utf8)
        let nextDelimiter = Data("\r\n--\(boundary)".utf8)
        let blank = Data("\r\n\r\n".utf8)
        guard var cursor = body.range(of: delimiter)?.upperBound else { return nil }
        var parts: [MultipartPart] = []
        while true {
            let rest = body[cursor...]
            if rest.starts(with: Data("--".utf8)) { return parts }
            guard rest.starts(with: Data("\r\n".utf8)) else { return nil }
            let headerStart = cursor + 2
            guard let headerEnd = body.range(of: blank, in: headerStart..<body.endIndex),
                  let next = body.range(of: nextDelimiter, in: headerEnd.upperBound..<body.endIndex) else { return nil }
            let headerText = String(decoding: body[headerStart..<headerEnd.lowerBound], as: UTF8.self)
            var name: String?
            var filename: String?
            var contentType: String?
            for line in headerText.components(separatedBy: "\r\n") {
                let lower = line.lowercased()
                if lower.hasPrefix("content-disposition:") {
                    name = dispositionParameter("name", in: line)
                    filename = dispositionParameter("filename", in: line)
                } else if lower.hasPrefix("content-type:") {
                    contentType = line.dropFirst("content-type:".count).trimmingCharacters(in: .whitespaces)
                }
            }
            guard let name else { return nil }
            parts.append(MultipartPart(name: name, filename: filename, contentType: contentType, data: Data(body[headerEnd.upperBound..<next.lowerBound])))
            cursor = next.upperBound
        }
    }

    private static func dispositionParameter(_ key: String, in line: String) -> String? {
        for piece in line.split(separator: ";").map({ $0.trimmingCharacters(in: .whitespaces) }) {
            guard let equals = piece.firstIndex(of: "="), piece[..<equals].lowercased() == key else { continue }
            return piece[piece.index(after: equals)...].trimmingCharacters(in: CharacterSet(charactersIn: "\""))
        }
        return nil
    }
}

// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import DramaEngine
import Foundation

public enum AudioFormat: String, Sendable, Equatable, CaseIterable {
    case wav, pcm, m4a

    public var contentType: String {
        switch self {
        case .wav: "audio/wav"
        case .pcm: "audio/pcm"
        case .m4a: "audio/mp4"
        }
    }
}

public enum Delivery: Sendable, Equatable {
    case file(AudioFormat)
    case pcmStream
    /// Server-Sent Events: `audio`, `sentence`, `done`, and `error`.
    case events

    public var streams: Bool { if case .file = self { false } else { true } }
}

public enum AudioSource: Sendable, Equatable {
    case path(String)
    case bytes(Data)
}

public struct ReferenceClip: Sendable, Equatable {
    public var audio: AudioSource
    public var transcript: String
}

public typealias VoiceKey = String

public enum VoiceCasting: Sendable, Equatable {
    case none
    case voice(VoiceKey)
    case clips([ReferenceClip])
    case speakers([VoiceKey])
}

/// An earlier take named in `previous_takes`.
public enum TakeSource: Sendable, Equatable {
    case cached(UUID)
    case inline(Take)
}

public struct SpeechParameters: Sendable, Equatable {
    /// Nil when the request continues the line of a cached take it names.
    public var text: String?
    public var casting: VoiceCasting
    public var character: VoiceDesign?
    public var note: String?
    public var previousTakes: [TakeSource]
    public var delivery: Delivery
    public var expressiveness: Double
    public var quickPreview: Bool
    public var seed: UInt64?
    public var applyPronunciation: Bool

    public var config: GenerationConfig {
        var config = GenerationConfig.make(expressiveness: expressiveness, quickPreview: quickPreview)
        config.seed = seed
        if delivery.streams { config.streaming = .defaultLowLatency }
        return config
    }
}

public struct NewVoice: Sendable, Equatable {
    public var name: String
    public var transcript: String
    public var audio: AudioSource
    public var fileName: String?
}

public enum APIRequest: Sendable, Equatable {
    case health
    case models
    case listVoices
    case addVoice(NewVoice)
    case deleteVoice(VoiceKey)
    case speech(SpeechParameters)
    /// Text arrives in the request body while the response streams audio.
    case live(SpeechParameters)
    case take(String)
}

public struct APIError: Error, Sendable, Equatable {
    public var status: HTTPStatus
    public var message: String
    public var headers: [String: String] = [:]

    public init(_ status: HTTPStatus, _ message: String, headers: [String: String] = [:]) {
        self.status = status
        self.message = message
        self.headers = headers
    }

    static func bad(_ message: String) -> APIError { APIError(.badRequest, message) }
}

public struct APIResponse: Sendable {
    public enum Body: Sendable {
        case data(Data)
        case chunked(AsyncThrowingStream<Data, Error>)
    }

    public var status: HTTPStatus
    public var headers: [String: String]
    public var body: Body
    /// The producer of a chunked body holds its own continuation, so the
    /// stream outlives a consumer that stops reading; the server calls this
    /// when the client goes away.
    public var abandon: @Sendable () -> Void

    public init(status: HTTPStatus, headers: [String: String] = [:], body: Body, abandon: @escaping @Sendable () -> Void = {}) {
        self.status = status
        self.headers = headers
        self.body = body
        self.abandon = abandon
    }

    public static func json<T: Encodable>(_ value: T, status: HTTPStatus = .ok) -> APIResponse {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        let data = (try? encoder.encode(value)) ?? Data("{}".utf8)
        return APIResponse(status: status, headers: ["Content-Type": "application/json"], body: .data(data))
    }

    public static func error(_ error: APIError) -> APIResponse {
        APIResponse(status: error.status, headers: error.headers.merging(["Content-Type": "text/plain; charset=utf-8"]) { a, _ in a }, body: .data(Data((error.message + "\n").utf8)))
    }

    /// Keep-alive is off, so every response closes its connection.
    public var head: Data {
        var lines = ["HTTP/1.1 \(status.rawValue) \(status.reason)"]
        var all = headers
        switch body {
        case .data(let data): all["Content-Length"] = String(data.count)
        case .chunked: all["Transfer-Encoding"] = "chunked"
        }
        all["Connection"] = "close"
        lines += all.sorted { $0.key < $1.key }.map { "\($0.key): \($0.value)" }
        return Data((lines.joined(separator: "\r\n") + "\r\n\r\n").utf8)
    }

    public var bodyData: Data? {
        if case .data(let data) = body { return data }
        return nil
    }
}

extension APIRequest {
    private struct Route {
        var method: String
        var pattern: [String]
        var parse: (_ captures: [String], _ request: HTTPRequest) throws -> APIRequest
    }

    private static let routes: [Route] = [
        Route(method: "GET", pattern: ["v1", "health"]) { _, _ in .health },
        Route(method: "GET", pattern: ["v1", "models"]) { _, _ in .models },
        Route(method: "GET", pattern: ["v1", "voices"]) { _, _ in .listVoices },
        Route(method: "POST", pattern: ["v1", "voices"]) { _, request in .addVoice(try newVoice(request)) },
        Route(method: "DELETE", pattern: ["v1", "voices", "{id}"]) { captures, _ in .deleteVoice(captures[0]) },
        Route(method: "POST", pattern: ["v1", "audio", "speech"]) { _, request in .speech(try speech(request.body, textField: "input")) },
        Route(method: "POST", pattern: ["v1", "tts"]) { _, request in .speech(try speech(request.body, textField: "text")) },
        Route(method: "GET", pattern: ["v1", "takes", "{id}"]) { captures, _ in .take(captures[0]) },
        Route(method: "GET", pattern: ["v1", "audio", "speech", "stream"]) { _, request in
            .speech(try speech(query: request.query, requireText: true, delivery: .events))
        },
        Route(method: "POST", pattern: liveRoute) { _, request in
            .live(try speech(query: request.query, requireText: false, delivery: wantsEvents(request) ? .events : .pcmStream))
        },
    ]

    private static let liveRoute = ["v1", "audio", "speech", "live"]

    /// The routes whose body the server hands over as it arrives instead of reading it whole.
    public static func streamsBody(_ request: HTTPRequest) -> Bool {
        request.method == "POST" && request.path.split(separator: "/").map(String.init) == liveRoute
    }

    private static func wantsEvents(_ request: HTTPRequest) -> Bool {
        request.header("accept")?.lowercased().contains("text/event-stream") == true
    }

    public static func parse(_ request: HTTPRequest) -> Result<APIRequest, APIError> {
        let segments = request.path.split(separator: "/").map(String.init)
        var allowed: [String] = []
        for route in routes {
            guard let captures = match(route.pattern, segments) else { continue }
            guard route.method == request.method else {
                allowed.append(route.method)
                continue
            }
            do {
                return .success(try route.parse(captures, request))
            } catch let error as APIError {
                return .failure(error)
            } catch {
                return .failure(.bad(String(describing: error)))
            }
        }
        if !allowed.isEmpty {
            return .failure(APIError(.methodNotAllowed, "\(request.path) takes \(allowed.joined(separator: ", ")).", headers: ["Allow": allowed.joined(separator: ", ")]))
        }
        return .failure(APIError(.notFound, "No route for \(request.method) \(request.path)."))
    }

    private static func match(_ pattern: [String], _ segments: [String]) -> [String]? {
        guard pattern.count == segments.count else { return nil }
        var captures: [String] = []
        for (expected, actual) in zip(pattern, segments) {
            if expected.hasPrefix("{") {
                captures.append(actual)
            } else if expected != actual {
                return nil
            }
        }
        return captures
    }

    private struct SpeechBody: Decodable {
        var input: String?
        var text: String?
        var voice: String?
        var references: [ReferenceBody]?
        var response_format: String?
        var stream: Bool?
        var expressiveness: Double?
        var quick_preview: Bool?
        var seed: Int64?
        var speaker_voices: [String: String]?
        var character: CharacterBody?
        var note: String?
        var previous_takes: [TakeBody]?
        var apply_pronunciation: Bool?
    }

    private struct CharacterBody: Decodable {
        var name: String?
        var description: String?
        var scene: String?
    }

    private struct TakeBody: Decodable {
        var take_id: String?
        var text: String?
        var codes_base64: String?
    }

    private struct ReferenceBody: Decodable {
        var audio_path: String?
        var audio_base64: String?
        var text: String?
    }

    private struct VoiceBody: Decodable {
        var name: String?
        var transcript: String?
        var audio_base64: String?
        var audio_path: String?
    }

    static func speech(_ body: Data, textField: String) throws -> SpeechParameters {
        try speech(try decode(SpeechBody.self, from: body), textField: textField, requireText: true)
    }

    /// The speech fields a URL can carry, for EventSource-style GETs and the live route.
    static func speech(query: [String: String], requireText: Bool, delivery: Delivery) throws -> SpeechParameters {
        func number<T: LosslessStringConvertible>(_ key: String) throws -> T? {
            guard let value = query[key] else { return nil }
            guard let number = T(value) else { throw APIError.bad("\(key) must be a number.") }
            return number
        }
        var body = SpeechBody()
        body.input = query["input"]
        body.voice = query["voice"]
        body.expressiveness = try number("expressiveness")
        body.seed = try number("seed")
        body.quick_preview = query["quick_preview"].map { ["1", "true", "yes"].contains($0.lowercased()) }
        if let value = query["apply_pronunciation"]?.lowercased() {
            if ["1", "true", "yes"].contains(value) { body.apply_pronunciation = true }
            else if ["0", "false", "no"].contains(value) { body.apply_pronunciation = false }
            else { throw APIError.bad("apply_pronunciation must be true or false.") }
        }
        if query["character_name"] != nil || query["character_description"] != nil {
            body.character = CharacterBody(name: query["character_name"], description: query["character_description"], scene: query["character_scene"])
        }
        var parameters = try speech(body, textField: "input", requireText: requireText)
        parameters.delivery = delivery
        return parameters
    }

    private static func speech(_ decoded: SpeechBody, textField: String, requireText: Bool) throws -> SpeechParameters {
        let previousTakes = try takes(decoded.previous_takes ?? [])
        let text = (decoded.input ?? decoded.text)?.trimmingCharacters(in: .whitespacesAndNewlines)
        let continuesTake = previousTakes.contains { if case .cached = $0 { true } else { false } }
        guard !requireText || text?.isEmpty == false || (text == nil && continuesTake) else {
            throw APIError.bad("\(textField) is required: the text to speak. Leave it out only to retake the line of a take_id in previous_takes.")
        }
        let character = try character(decoded.character)
        if character != nil, decoded.voice != nil || decoded.references?.isEmpty == false || decoded.speaker_voices?.isEmpty == false {
            throw APIError.bad("character designs a voice; it can't be combined with voice, references, or speaker_voices.")
        }
        let note = decoded.note?.trimmingCharacters(in: .whitespacesAndNewlines)
        let expressiveness = decoded.expressiveness ?? GenerationConfig.defaultExpressiveness
        guard (0...1).contains(expressiveness) else { throw APIError.bad("expressiveness must be between 0 and 1.") }
        var format: AudioFormat?
        if let name = decoded.response_format {
            guard let parsed = AudioFormat(rawValue: name.lowercased()) else {
                throw APIError.bad("response_format must be wav, pcm, or m4a.")
            }
            format = parsed
        }
        let delivery: Delivery
        if decoded.stream == true {
            guard format == nil || format == .pcm else { throw APIError.bad("stream needs response_format pcm.") }
            delivery = .pcmStream
        } else {
            delivery = .file(format ?? .wav)
        }
        if let seed = decoded.seed, seed < 0 { throw APIError.bad("seed must not be negative.") }
        return SpeechParameters(
            text: text,
            casting: try casting(decoded),
            character: character,
            note: note?.isEmpty == false ? note : nil,
            previousTakes: previousTakes,
            delivery: delivery,
            expressiveness: expressiveness,
            quickPreview: decoded.quick_preview ?? false,
            seed: decoded.seed.map(UInt64.init),
            applyPronunciation: decoded.apply_pronunciation ?? true
        )
    }

    private static func character(_ body: CharacterBody?) throws -> VoiceDesign? {
        guard let body else { return nil }
        guard let name = body.name?.trimmingCharacters(in: .whitespacesAndNewlines), !name.isEmpty,
              let description = body.description?.trimmingCharacters(in: .whitespacesAndNewlines), !description.isEmpty else {
            throw APIError.bad("character needs a name and a description.")
        }
        let scene = body.scene?.trimmingCharacters(in: .whitespacesAndNewlines)
        return VoiceDesign(name: name, description: description, scene: scene?.isEmpty == false ? scene : nil)
    }

    private static func takes(_ bodies: [TakeBody]) throws -> [TakeSource] {
        try bodies.enumerated().map { index, body in
            let field = "previous_takes[\(index)]"
            switch (body.take_id, body.text, body.codes_base64) {
            case (let id?, nil, nil):
                guard let uuid = UUID(uuidString: id) else { throw APIError.bad("\(field).take_id is not a take id.") }
                return .cached(uuid)
            case (nil, let text?, let codes?):
                guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { throw APIError.bad("\(field).text must not be empty.") }
                guard let decoded = TakeCodes.decode(codes) else {
                    throw APIError.bad("\(field).codes_base64 must be base64 of 16-bit little-endian codes, 10 codebooks of equal length, each code below 4096.")
                }
                return .inline(Take(text: text, codes: decoded))
            default:
                throw APIError.bad("\(field) needs either take_id, or text and codes_base64.")
            }
        }
    }

    private static func casting(_ body: SpeechBody) throws -> VoiceCasting {
        let references = body.references ?? []
        let speakers = body.speaker_voices ?? [:]
        let given = [body.voice != nil, !references.isEmpty, !speakers.isEmpty].filter { $0 }.count
        guard given <= 1 else { throw APIError.bad("Pass only one of voice, references, or speaker_voices.") }
        if let voice = body.voice { return .voice(voice) }
        if !references.isEmpty {
            return .clips(try references.enumerated().map { index, reference in
                guard let transcript = reference.text?.trimmingCharacters(in: .whitespacesAndNewlines), !transcript.isEmpty else {
                    throw APIError.bad("references[\(index)].text is required: the transcript of the clip.")
                }
                return ReferenceClip(audio: try audio(path: reference.audio_path, base64: reference.audio_base64, field: "references[\(index)]"), transcript: transcript)
            })
        }
        if !speakers.isEmpty {
            var byIndex: [Int: String] = [:]
            for (key, voice) in speakers {
                guard let index = Int(key), (0...ModelConstants.maxSpeakerID).contains(index) else {
                    throw APIError.bad("speaker_voices keys are speaker numbers from 0 to \(ModelConstants.maxSpeakerID); got \"\(key)\".")
                }
                byIndex[index] = voice
            }
            guard byIndex.keys.sorted() == Array(0..<byIndex.count) else {
                throw APIError.bad("speaker_voices must cover speakers 0 through \(byIndex.count - 1) without gaps.")
            }
            return .speakers((0..<byIndex.count).map { byIndex[$0]! })
        }
        return .none
    }

    private static func audio(path: String?, base64: String?, field: String) throws -> AudioSource {
        switch (path, base64) {
        case (let path?, nil):
            return .path(path)
        case (nil, let base64?):
            guard let data = Data(base64Encoded: base64, options: .ignoreUnknownCharacters), !data.isEmpty else {
                throw APIError.bad("\(field).audio_base64 is not valid base64.")
            }
            return .bytes(data)
        default:
            throw APIError.bad("\(field) needs exactly one of audio_path or audio_base64.")
        }
    }

    static func newVoice(_ request: HTTPRequest) throws -> NewVoice {
        if let contentType = request.header("content-type"), let boundary = Multipart.boundary(contentType: contentType) {
            guard let parts = Multipart.parse(request.body, boundary: boundary) else { throw APIError.bad("The multipart body is malformed.") }
            guard let file = parts.first(where: { $0.name == "file" }), !file.data.isEmpty else {
                throw APIError.bad("The multipart body needs a file part with the reference clip.")
            }
            let field = { (name: String) in parts.first { $0.name == name }?.text?.trimmingCharacters(in: .whitespacesAndNewlines) }
            let fallbackName = file.filename.map { URL(filePath: $0).deletingPathExtension().lastPathComponent }
            return try voice(name: field("name") ?? fallbackName, transcript: field("transcript"), audio: .bytes(file.data), fileName: file.filename)
        }
        let body = try decode(VoiceBody.self, from: request.body)
        let fileName = body.audio_path.map { URL(filePath: $0).lastPathComponent }
        return try voice(name: body.name, transcript: body.transcript, audio: try audio(path: body.audio_path, base64: body.audio_base64, field: "the voice"), fileName: fileName)
    }

    private static func voice(name: String?, transcript: String?, audio: AudioSource, fileName: String?) throws -> NewVoice {
        guard let name = name?.trimmingCharacters(in: .whitespacesAndNewlines), !name.isEmpty else { throw APIError.bad("name is required.") }
        guard let transcript = transcript?.trimmingCharacters(in: .whitespacesAndNewlines), !transcript.isEmpty else {
            throw APIError.bad("transcript is required: the words spoken in the clip.")
        }
        return NewVoice(name: name, transcript: transcript, audio: audio, fileName: fileName)
    }

    private static func decode<T: Decodable>(_ type: T.Type, from body: Data) throws -> T {
        guard !body.isEmpty else { throw APIError.bad("The request needs a JSON body.") }
        do {
            return try JSONDecoder().decode(type, from: body)
        } catch let DecodingError.typeMismatch(_, context), let DecodingError.valueNotFound(_, context) {
            let field = context.codingPath.map { $0.intValue.map { "[\($0)]" } ?? $0.stringValue }.joined(separator: ".")
            throw APIError.bad("\(field.isEmpty ? "The body" : field) has the wrong type.")
        } catch {
            throw APIError.bad("The body is not valid JSON.")
        }
    }
}

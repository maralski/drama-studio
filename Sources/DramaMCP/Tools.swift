import Foundation

public struct ToolResult: Sendable, Equatable {
    public var structured: JSON?
    public var text: String
    public var isError: Bool

    public static func success(_ value: JSON) -> ToolResult { ToolResult(structured: value, text: value.serialized, isError: false) }
    public static func failure(_ message: String) -> ToolResult { ToolResult(structured: nil, text: message, isError: true) }

    var json: JSON {
        var result: [String: JSON] = ["content": [["type": "text", "text": .string(text)]], "isError": .bool(isError)]
        if let structured { result["structuredContent"] = structured }
        return .object(result)
    }
}

public struct ToolContext: Sendable {
    public var api: any DramaAPI
    public var outputDirectory: URL
    public var now: @Sendable () -> Date
    public var play: @Sendable (URL) throws -> Void

    public init(api: any DramaAPI, outputDirectory: URL = ToolContext.defaultOutputDirectory, now: @escaping @Sendable () -> Date = { Date() }, play: @escaping @Sendable (URL) throws -> Void = ToolContext.afplay) {
        self.api = api
        self.outputDirectory = outputDirectory
        self.now = now
        self.play = play
    }

    public static var defaultOutputDirectory: URL {
        URL.applicationSupportDirectory.appending(path: "DramaStudio/Output", directoryHint: .isDirectory)
    }

    /// Plays in the background so the tool returns while the audio is still sounding.
    public static let afplay: @Sendable (URL) throws -> Void = { url in
        let process = Process()
        process.executableURL = URL(filePath: "/usr/bin/afplay")
        process.arguments = [url.path]
        try process.run()
    }
}

struct ToolInputError: Error {
    var message: String
}

public struct MCPTool: Sendable {
    public var name: String
    public var description: String
    public var inputSchema: JSON
    var run: @Sendable (_ arguments: [String: JSON], _ context: ToolContext) async throws -> JSON

    var listing: JSON {
        ["name": .string(name), "description": .string(description), "inputSchema": inputSchema]
    }
}

extension MCPTool {
    public static let all: [MCPTool] = [synthesizeSpeech, retakeWithNote, listVoices, addVoice, deleteVoice, studioStatus]

    static let synthesizeSpeech = MCPTool(
        name: "synthesize_speech",
        description: "Speak text with Fish Audio Drama 3 running locally in Drama Studio and save it as a 44.1 kHz 16-bit mono WAV file. Pass voice as an id or name from list_voices to clone that voice, or character (a name, a description of the voice, and optionally a scene) to design one, or neither for the model's own voice. The text may carry performance cues such as [laughs], [sighs] or <whisper>quiet words</whisper>, and multi-speaker scripts tag turns with <|speaker:0|>, <|speaker:1|>. Generation takes about as long as the audio on the Metal engine and the studio runs one generation at a time, so a busy studio returns an error: wait a few seconds and call again. Returns the file path, the audio length in seconds, the frame count, the generation time and a take_id that retake_with_note accepts. Set play to true to also play the file through the Mac's speakers.",
        inputSchema: [
            "type": "object",
            "properties": [
                "text": ["type": "string", "description": "What to say."],
                "voice": ["type": "string", "description": "A voice id or name from list_voices."],
                "character": [
                    "type": "object",
                    "description": "Design a voice from a description instead of cloning one; not with voice.",
                    "properties": [
                        "name": ["type": "string", "description": "What the character is called, such as Captain."],
                        "description": ["type": "string", "description": "How the voice sounds: age, timbre, pace, manner."],
                        "scene": ["type": "string", "description": "Optional: where the line is spoken."],
                    ],
                    "required": ["name", "description"],
                    "additionalProperties": false,
                ],
                "output_path": ["type": "string", "description": "Where to write the WAV. Defaults to ~/Library/Application Support/DramaStudio/Output/<timestamp>.wav."],
                "expressiveness": ["type": "number", "minimum": 0, "maximum": 1, "description": "0 reads flat and steady, 0.5 is the vendor's sampling, 1 is the Fish Audio demo's sampling: lively, keeps every sentence of cue lines, a few more slurred words on prose. Default 0.5."],
                "quick_preview": ["type": "boolean", "description": "Faster rough take: no retries and a tighter length cap."],
                "play": ["type": "boolean", "description": "Play the result through the speakers."],
            ],
            "required": ["text"],
            "additionalProperties": false,
        ],
        run: { arguments, context in
            guard let text = arguments["text"]?.string, !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw ToolInputError(message: "text is required.")
            }
            if arguments["character"] != nil, arguments["voice"] != nil { throw ToolInputError(message: "Pass voice or character, not both.") }
            var body: [String: JSON] = ["input": .string(text), "response_format": "wav"]
            for key in ["voice", "character", "expressiveness", "quick_preview"] { body[key] = arguments[key] }
            return try await speak(body, arguments, context)
        }
    )

    static let retakeWithNote = MCPTool(
        name: "retake_with_note",
        description: "Generate a line again with a director's note, the way a director asks an actor for another take. Pass the take_id that synthesize_speech or an earlier retake_with_note returned, and a short note such as \"softer and warmer\", \"slower, with more pauses\", \"angrier\", \"whispered\" or \"more amused than before; drop the hurt\". The retake keeps the same line and the same voice: a cloned voice keeps its clip, and a designed or model-default voice keeps the first take of the line as its reference, with the latest earlier takes as context. Drama Studio remembers its last 20 takes, so an older take_id returns an error. Returns the WAV path, its length and frame count, and the new take_id to retake again.",
        inputSchema: [
            "type": "object",
            "properties": [
                "take_id": ["type": "string", "description": "The take to redo, from synthesize_speech or retake_with_note."],
                "note": ["type": "string", "description": "How to deliver it differently, at most 120 characters."],
                "output_path": ["type": "string", "description": "Where to write the WAV. Defaults to ~/Library/Application Support/DramaStudio/Output/<timestamp>.wav."],
                "play": ["type": "boolean", "description": "Play the result through the speakers."],
            ],
            "required": ["take_id", "note"],
            "additionalProperties": false,
        ],
        run: { arguments, context in
            guard let id = arguments["take_id"]?.string, !id.isEmpty, let note = arguments["note"]?.string,
                  !note.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw ToolInputError(message: "take_id and note are both required.")
            }
            return try await speak(["note": .string(note), "previous_takes": [["take_id": .string(id)]], "response_format": "wav"], arguments, context)
        }
    )

    private static func speak(_ body: [String: JSON], _ arguments: [String: JSON], _ context: ToolContext) async throws -> JSON {
        let reply = try await context.api.send(APICall("POST", "/v1/audio/speech", json: .object(body)))
        guard reply.ok else { throw ToolInputError(message: "Drama Studio answered \(reply.status): \(reply.text)") }
        let url = arguments["output_path"]?.string.map { URL(filePath: ($0 as NSString).expandingTildeInPath) }
            ?? context.outputDirectory.appending(path: "\(timestamp(context.now())).wav")
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try reply.body.write(to: url)
        let play = arguments["play"]?.bool == true
        if play { try context.play(url) }
        var result: [String: JSON] = [
            "path": .string(url.path),
            "seconds": .double(Double(reply.header("X-Drama-Seconds") ?? "") ?? 0),
            "frames": .int(Int(reply.header("X-Drama-Frames") ?? "") ?? 0),
            "generation_seconds": .double(Double(reply.header("X-Drama-Generation-Seconds") ?? "") ?? 0),
            "played": .bool(play),
        ]
        if let take = reply.header("X-Drama-Take-Id") { result["take_id"] = .string(take) }
        return .object(result)
    }

    static let listVoices = MCPTool(
        name: "list_voices",
        description: "List the reference voices in Drama Studio's voice library, the same library the app shows. Each voice has an id, a name, the transcript of its reference clip and the clip's length in seconds. Pass a voice's id or name as the voice argument of synthesize_speech to speak in that voice, or its id to delete_voice to remove it.",
        inputSchema: ["type": "object", "properties": [:], "additionalProperties": false],
        run: { _, context in try await json(context.api.send(APICall("GET", "/v1/voices"))) }
    )

    static let addVoice = MCPTool(
        name: "add_voice",
        description: "Add a voice to Drama Studio's library by cloning it from a short reference recording. Give the path of an audio file on this Mac (WAV, AIFF, M4A, MP3 or anything else AVFoundation reads; 5 to 30 seconds of one clear speaker works best), the exact transcript of what is said in it, and a name to call the voice by. Only clone a voice with its owner's consent. Returns the new voice with its id, which synthesize_speech then accepts as its voice argument.",
        inputSchema: [
            "type": "object",
            "properties": [
                "audio_path": ["type": "string", "description": "Path of the reference clip."],
                "transcript": ["type": "string", "description": "The words spoken in the clip, verbatim."],
                "name": ["type": "string", "description": "A name for the voice."],
            ],
            "required": ["audio_path", "transcript", "name"],
            "additionalProperties": false,
        ],
        run: { arguments, context in
            guard let path = arguments["audio_path"]?.string, let transcript = arguments["transcript"]?.string, let name = arguments["name"]?.string else {
                throw ToolInputError(message: "audio_path, transcript and name are all required.")
            }
            let url = URL(filePath: (path as NSString).expandingTildeInPath)
            guard let audio = try? Data(contentsOf: url) else { throw ToolInputError(message: "Can't read \(url.path).") }
            return try await json(context.api.send(APICall("POST", "/v1/voices", json: ["name": .string(name), "transcript": .string(transcript), "audio_base64": .string(audio.base64EncodedString())])))
        }
    )

    static let deleteVoice = MCPTool(
        name: "delete_voice",
        description: "Remove a voice from Drama Studio's library for good, by the id list_voices reports. The reference recording it was cloned from is not touched; only the stored voice goes. Returns the id of the deleted voice, or an error when no voice has that id.",
        inputSchema: [
            "type": "object",
            "properties": ["id": ["type": "string", "description": "The voice id from list_voices."]],
            "required": ["id"],
            "additionalProperties": false,
        ],
        run: { arguments, context in
            guard let id = arguments["id"]?.string, !id.isEmpty else { throw ToolInputError(message: "id is required.") }
            return try await json(context.api.send(APICall("DELETE", "/v1/voices/\(id)")))
        }
    )

    static let studioStatus = MCPTool(
        name: "studio_status",
        description: "Report whether Drama Studio is ready to speak: whether the model is loaded (the first generation after launch loads 41 GB of weights and takes a while), which engine runs it (metal, or mock for the preview engine that produces placeholder tones), whether a generation is running right now, whether the kernels are tuned for this Mac, and the model snapshot. Call it before a batch of work or after an error to decide whether to wait.",
        inputSchema: ["type": "object", "properties": [:], "additionalProperties": false],
        run: { _, context in
            let health = try await json(context.api.send(APICall("GET", "/v1/health")))
            let models = try await json(context.api.send(APICall("GET", "/v1/models")))
            var status = health.object ?? [:]
            if case .array(let data)? = models["data"], let model = data.first { status["model"] = model }
            status["api_url"] = .string(context.api.baseURL.absoluteString)
            return .object(status)
        }
    )

    private static func json(_ reply: APIReply) throws -> JSON {
        guard reply.ok else { throw ToolInputError(message: "Drama Studio answered \(reply.status): \(reply.text)") }
        guard let value = try? JSON(parsing: reply.body) else { throw ToolInputError(message: "Drama Studio answered with something other than JSON: \(reply.text.prefix(200))") }
        return value
    }

    private static func timestamp(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "yyyy-MM-dd'T'HH-mm-ss.SSS"
        return formatter.string(from: date)
    }
}

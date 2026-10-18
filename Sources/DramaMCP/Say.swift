import Foundation

/// One `drama-say` invocation, parsed from argv.
public struct SayCommand: Sendable, Equatable {
    public enum Source: Sendable, Equatable { case text(String), file(URL), standardInput }
    public enum Voice: Sendable, Equatable {
        case studio
        case library(String)
        case character(name: String, description: String, scene: String?)
        case speakers([Int: String])
    }
    public enum Output: Sendable, Equatable {
        case play
        case file(URL)
        case standardOutput
    }

    /// `nil` only with `retakeOf`: the retake speaks that take's own line.
    public var source: Source? = .standardInput
    public var voice: Voice = .studio
    public var output: Output = .play
    public var expressiveness: Double?
    public var quickPreview = false
    public var seed: UInt64?
    public var note: String?
    public var retakeOf: String?
    public var listVoices = false
    public var verbose = false

    static let fileFormats = ["wav", "m4a", "pcm"]

    public func request(text: String?) -> APICall {
        var body: [String: JSON] = [:]
        if let text { body["input"] = .string(text) }
        switch voice {
        case .studio: break
        case .library(let key): body["voice"] = .string(key)
        case .character(let name, let description, let scene):
            var character: [String: JSON] = ["name": .string(name), "description": .string(description)]
            if let scene { character["scene"] = .string(scene) }
            body["character"] = .object(character)
        case .speakers(let voices):
            body["speaker_voices"] = .object(Dictionary(uniqueKeysWithValues: voices.map { (String($0.key), .string($0.value)) }))
        }
        switch output {
        case .play:
            body["stream"] = true
            body["response_format"] = "pcm"
        case .file(let url): body["response_format"] = .string(url.pathExtension.lowercased())
        case .standardOutput: body["response_format"] = "pcm"
        }
        if let expressiveness { body["expressiveness"] = .double(expressiveness) }
        if quickPreview { body["quick_preview"] = true }
        if let seed { body["seed"] = .int(Int(seed)) }
        if let note { body["note"] = .string(note) }
        if let retakeOf { body["previous_takes"] = [["take_id": .string(retakeOf)]] }
        return APICall("POST", "/v1/audio/speech", json: .object(body))
    }
}

/// A bad command line. `message` is nil when `-h` asked for the usage text.
public struct SayUsageError: Error, Equatable {
    public var message: String?

    public static let usage = """
    usage: drama-say [options] [text ...]

    Speaks text through the running Drama Studio app. With no text and no -f,
    reads the text from standard input.

      -f FILE                 read the text from FILE (- for standard input)
      -v VOICE                a library voice, by name or id
      -v '?', --list-voices   list the voice library and exit
      --character "Name: description"
                              design a voice from a description
      --scene TEXT            where the character speaks (with --character)
      --speaker N=VOICE       library voice for speaker N of a multi-speaker
                              script; repeat for each speaker
      -o FILE                 write FILE instead of playing (.wav, .m4a, .pcm)
      -o -                    write raw 16-bit mono 44.1 kHz PCM to standard output
      --expressiveness 0..1   0 flat, 0.5 default, 1 lively
      --quick                 faster rough take
      --seed N                fixed sampler seed
      --note TEXT             director's note; with --retake, for that take
      --retake TAKE_ID        generate a take again; without text, its own line
      --verbose               print the take id and timings on standard error
      -h, --help              print this text

    Exit status: 0 spoken, 64 bad usage, 75 the studio is busy (retry), 1 other errors.
    """
}

/// A failure with the exit status a script sees.
public struct SayExit: Error, Equatable {
    public var status: Int32
    public var message: String

    public static let usage: Int32 = 64
    public static let temporaryFailure: Int32 = 75

    static func failure(_ reply: APIReply) -> SayExit {
        let text = reply.text.isEmpty ? "Drama Studio answered \(reply.status)." : reply.text
        guard reply.status == 429 else { return SayExit(status: 1, message: text) }
        return SayExit(status: temporaryFailure, message: "\(text) Retry-After: \(reply.header("Retry-After") ?? "?") s")
    }
}

// MARK: - Parsing

extension SayCommand {
    private struct Draft {
        var command = SayCommand()
        var words: [String] = []
        var file: Source?
        var voiceFlag: String?
        var scene: String?
        var speakers: [Int: String] = [:]

        mutating func setVoice(_ voice: Voice, flag: String) throws {
            if let voiceFlag, voiceFlag != flag {
                throw SayUsageError(message: "\(voiceFlag) and \(flag) both pick the voice; pass one of -v, --character, --speaker.")
            }
            voiceFlag = flag
            command.voice = voice
        }

        func finish() throws -> SayCommand {
            var command = self.command
            if let scene {
                guard case .character(let name, let description, _) = command.voice else {
                    throw SayUsageError(message: "--scene goes with --character.")
                }
                command.voice = .character(name: name, description: description, scene: scene)
            }
            switch (words.isEmpty, file) {
            case (false, .some): throw SayUsageError(message: "give the text as words or with -f, not both.")
            case (false, nil): command.source = .text(words.joined(separator: " "))
            case (true, .some(let file)): command.source = file
            case (true, nil): command.source = command.retakeOf == nil ? .standardInput : nil
            }
            return command
        }
    }

    private enum Flag {
        case toggle((inout Draft) throws -> Void)
        case value((inout Draft, String) throws -> Void)
    }

    private static let flags: [String: Flag] = {
        let input = Flag.value { draft, path in draft.file = path == "-" ? .standardInput : .file(URL(filePath: path)) }
        let voice = Flag.value { draft, key in
            if key == "?" { draft.command.listVoices = true } else { try draft.setVoice(.library(key), flag: "-v") }
        }
        let output = Flag.value { draft, path in
            if path == "-" { draft.command.output = .standardOutput; return }
            let url = URL(filePath: path)
            guard fileFormats.contains(url.pathExtension.lowercased()) else {
                throw SayUsageError(message: "-o \(path): the extension picks the format; use .wav, .m4a or .pcm.")
            }
            draft.command.output = .file(url)
        }
        let help = Flag.toggle { _ in throw SayUsageError(message: nil) }
        return [
            "-f": input, "--input-file": input,
            "-v": voice, "--voice": voice,
            "-o": output, "--output-file": output,
            "-h": help, "--help": help,
            "--list-voices": .toggle { $0.command.listVoices = true },
            "--character": .value { draft, value in
                let parts = value.split(separator: ":", maxSplits: 1).map { $0.trimmingCharacters(in: .whitespaces) }
                guard parts.count == 2, !parts[0].isEmpty, !parts[1].isEmpty else {
                    throw SayUsageError(message: "--character takes \"Name: description\".")
                }
                try draft.setVoice(.character(name: parts[0], description: parts[1], scene: nil), flag: "--character")
            },
            "--scene": .value { draft, scene in draft.scene = scene },
            "--speaker": .value { draft, value in
                let parts = value.split(separator: "=", maxSplits: 1).map(String.init)
                guard parts.count == 2, let speaker = Int(parts[0]), speaker >= 0, !parts[1].isEmpty else {
                    throw SayUsageError(message: "--speaker takes N=VOICE, such as 0=Narrator.")
                }
                guard draft.speakers[speaker] == nil else { throw SayUsageError(message: "--speaker \(speaker) is given twice.") }
                draft.speakers[speaker] = parts[1]
                try draft.setVoice(.speakers(draft.speakers), flag: "--speaker")
            },
            "--expressiveness": .value { draft, value in
                guard let number = Double(value), (0...1).contains(number) else {
                    throw SayUsageError(message: "--expressiveness takes a number from 0 to 1.")
                }
                draft.command.expressiveness = number
            },
            "--quick": .toggle { $0.command.quickPreview = true },
            "--seed": .value { draft, value in
                guard let seed = UInt64(value), seed <= Int64.max else {
                    throw SayUsageError(message: "--seed takes a whole number from 0 to \(Int64.max).")
                }
                draft.command.seed = seed
            },
            "--note": .value { draft, note in draft.command.note = note },
            "--retake": .value { draft, id in draft.command.retakeOf = id },
            "--verbose": .toggle { $0.command.verbose = true },
        ]
    }()

    public static func parse(_ arguments: [String]) throws -> SayCommand {
        var draft = Draft()
        var rest = arguments[...]
        while let argument = rest.popFirst() {
            if argument == "--" {
                draft.words += rest
                break
            }
            guard argument.hasPrefix("-"), argument != "-" else {
                draft.words.append(argument)
                continue
            }
            let (name, inline) = argument.hasPrefix("--") && argument.contains("=")
                ? (String(argument.prefix { $0 != "=" }), String(argument.drop { $0 != "=" }.dropFirst()))
                : (argument, nil as String?)
            switch flags[name] {
            case nil:
                throw SayUsageError(message: "unknown option \(name).")
            case .toggle(let set):
                guard inline == nil else { throw SayUsageError(message: "\(name) takes no value.") }
                try set(&draft)
            case .value(let set):
                guard let value = inline ?? rest.popFirst() else { throw SayUsageError(message: "\(name) needs a value.") }
                try set(&draft, value)
            }
        }
        return try draft.finish()
    }
}

// MARK: - Running

/// The effects a run needs, so tests can stand in for the studio and the terminal.
public struct SayEnvironment: Sendable {
    public var api: any DramaAPI
    public var readStandardInput: @Sendable () -> Data
    public var writeStandardOutput: @Sendable (Data) -> Void
    public var writeStandardError: @Sendable (String) -> Void
    public var play: @Sendable (AsyncThrowingStream<Data, any Error>) async throws -> Void

    public init(
        api: any DramaAPI,
        readStandardInput: @escaping @Sendable () -> Data = { FileHandle.standardInput.readDataToEndOfFile() },
        writeStandardOutput: @escaping @Sendable (Data) -> Void = { FileHandle.standardOutput.write($0) },
        writeStandardError: @escaping @Sendable (String) -> Void = { FileHandle.standardError.write(Data($0.utf8)) },
        play: @escaping @Sendable (AsyncThrowingStream<Data, any Error>) async throws -> Void
    ) {
        self.api = api
        self.readStandardInput = readStandardInput
        self.writeStandardOutput = writeStandardOutput
        self.writeStandardError = writeStandardError
        self.play = play
    }
}

extension SayCommand {
    /// argv in, exit status out. Silent on success, like `say`.
    public static func main(_ arguments: [String], _ environment: SayEnvironment) async -> Int32 {
        do {
            try await parse(arguments).run(environment)
            return 0
        } catch let error as SayUsageError {
            guard let message = error.message else {
                environment.writeStandardOutput(Data((SayUsageError.usage + "\n").utf8))
                return 0
            }
            environment.writeStandardError("drama-say: \(message)\nRun drama-say --help for the options.\n")
            return SayExit.usage
        } catch let error as SayExit {
            environment.writeStandardError("drama-say: \(error.message)\n")
            return error.status
        } catch {
            environment.writeStandardError("drama-say: \(error.localizedDescription)\n")
            return 1
        }
    }

    func run(_ environment: SayEnvironment) async throws {
        if listVoices { return try await printVoices(environment) }
        let text = try source.map { try Self.read($0, environment) }
        if let text, text.isEmpty { throw SayUsageError(message: "no text to speak.") }
        let call = request(text: text)
        let trace = Trace(enabled: verbose, write: environment.writeStandardError)
        switch output {
        case .play:
            let (reply, chunks) = try await environment.api.bytes(for: call)
            guard reply.ok else { throw SayExit.failure(reply) }
            trace.take(reply)
            try await environment.play(trace.watching(chunks))
            trace.finished()
        case .file, .standardOutput:
            let reply = try await environment.api.send(call)
            guard reply.ok else { throw SayExit.failure(reply) }
            trace.take(reply)
            guard case .file(let url) = output else { return environment.writeStandardOutput(reply.body) }
            do {
                try reply.body.write(to: url, options: .atomic)
            } catch {
                throw SayExit(status: 1, message: "can't write \(url.path): \(error.localizedDescription)")
            }
        }
    }

    private static func read(_ source: Source, _ environment: SayEnvironment) throws -> String {
        let data: Data
        switch source {
        case .text(let text): return text.trimmingCharacters(in: .whitespacesAndNewlines)
        case .standardInput: data = environment.readStandardInput()
        case .file(let url):
            do { data = try Data(contentsOf: url) } catch { throw SayExit(status: 1, message: "can't read \(url.path).") }
        }
        return String(decoding: data, as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func printVoices(_ environment: SayEnvironment) async throws {
        let reply = try await environment.api.send(APICall("GET", "/v1/voices"))
        guard reply.ok else { throw SayExit.failure(reply) }
        guard case .array(let voices)? = (try? JSON(parsing: reply.body))?["voices"] else {
            throw SayExit(status: 1, message: "Drama Studio's voice list is not the JSON it should be: \(reply.text.prefix(200))")
        }
        let rows = voices.map { ($0["name"]?.string ?? "", $0["id"]?.string ?? "", $0["seconds"]?.double ?? 0) }
        let width = rows.map(\.0.count).max() ?? 0
        let lines = rows.map { name, id, seconds in
            name.padding(toLength: width, withPad: " ", startingAt: 0) + "  " + id + "  " + String(format: "%.1f s", seconds) + "\n"
        }
        environment.writeStandardOutput(Data(lines.joined().utf8))
    }
}

/// The `--verbose` lines: the take id, and for playback when the first audio
/// reached the speakers, how long the take is and any gaps where playback ran
/// ahead of generation.
private final class Trace: @unchecked Sendable {
    private let enabled: Bool
    private let write: @Sendable (String) -> Void
    private let clock = ContinuousClock()
    private let start: ContinuousClock.Instant
    private var firstAudio: Duration?
    private var samples = 0
    private var playedUntil: Duration = .zero
    private var gaps: Duration = .zero

    init(enabled: Bool, write: @escaping @Sendable (String) -> Void) {
        self.enabled = enabled
        self.write = write
        start = clock.now
    }

    private func line(_ text: String) {
        if enabled { write("drama-say: \(text)\n") }
    }

    private static func seconds(_ duration: Duration) -> String {
        String(format: "%.2f s", Double(duration.components.seconds) + Double(duration.components.attoseconds) / 1e18)
    }

    func take(_ reply: APIReply) {
        var text = "take \(reply.header("X-Drama-Take-Id") ?? "?") after \(Self.seconds(start.duration(to: clock.now)))"
        if let estimate = reply.header("X-Drama-Estimated-Seconds") { text += ", estimated \(estimate) s to generate" }
        line(text)
    }

    func watching(_ chunks: AsyncThrowingStream<Data, any Error>) -> AsyncThrowingStream<Data, any Error> {
        guard enabled else { return chunks }
        return AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    for try await chunk in chunks {
                        self.arrived(bytes: chunk.count)
                        continuation.yield(chunk)
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    private func arrived(bytes: Int) {
        let now = start.duration(to: clock.now)
        let audio = Duration.seconds(Double(bytes / 2) / 44100)
        if firstAudio == nil {
            firstAudio = now
            playedUntil = now
            line("first audio after \(Self.seconds(now))")
        }
        if now > playedUntil {
            gaps += now - playedUntil
            playedUntil = now
        }
        playedUntil += audio
        samples += bytes / 2
    }

    func finished() {
        guard let firstAudio else { return line("no audio") }
        let take = Duration.seconds(Double(samples) / 44100)
        line("take is \(Self.seconds(take)) of audio; first audio after \(Self.seconds(firstAudio)); playback stalled \(Self.seconds(gaps)) waiting on generation; done after \(Self.seconds(start.duration(to: clock.now)))")
    }
}

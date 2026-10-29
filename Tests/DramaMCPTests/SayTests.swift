// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
@testable import DramaMCP
import XCTest

final class Captured: @unchecked Sendable {
    private let lock = NSLock()
    private var out = Data()
    private var err = ""
    private var playedChunks: [Data] = []
    func stdout(_ data: Data) { lock.withLock { out.append(data) } }
    func stderr(_ text: String) { lock.withLock { err += text } }
    func play(_ chunk: Data) { lock.withLock { playedChunks.append(chunk) } }
    var standardOutput: Data { lock.withLock { out } }
    var standardError: String { lock.withLock { err } }
    var played: [Data] { lock.withLock { playedChunks } }
}

final class SayTests: XCTestCase {
    private func command(_ edit: (inout SayCommand) -> Void) -> SayCommand {
        var command = SayCommand()
        edit(&command)
        return command
    }

    /// Catches a flag that is ignored, lands in the wrong field, or swallows the
    /// word after it as text.
    func testEveryFlagFormParsesIntoItsField() throws {
        let cases: [([String], SayCommand)] = [
            (["Hello", "there."], command { $0.source = .text("Hello there.") }),
            ([], command { $0.source = .standardInput }),
            (["-f", "/tmp/line.txt"], command { $0.source = .file(URL(filePath: "/tmp/line.txt")) }),
            (["-f", "-"], command { $0.source = .standardInput }),
            (["--input-file=/tmp/line.txt"], command { $0.source = .file(URL(filePath: "/tmp/line.txt")) }),
            (["-v", "Narrator", "Hi"], command { $0.voice = .library("Narrator"); $0.source = .text("Hi") }),
            (["--voice=Narrator", "Hi"], command { $0.voice = .library("Narrator"); $0.source = .text("Hi") }),
            (["-v", "?"], command { $0.listVoices = true }),
            (["--list-voices"], command { $0.listVoices = true }),
            (["--character", "Captain: gravelly old male voice", "Hi"],
             command { $0.voice = .character(name: "Captain", description: "gravelly old male voice", scene: nil); $0.source = .text("Hi") }),
            (["--scene", "a storm at sea", "--character", "Captain: gravelly", "Hi"],
             command { $0.voice = .character(name: "Captain", description: "gravelly", scene: "a storm at sea"); $0.source = .text("Hi") }),
            (["--speaker", "0=Narrator", "--speaker", "1=abc-123", "Hi"],
             command { $0.voice = .speakers([0: "Narrator", 1: "abc-123"]); $0.source = .text("Hi") }),
            (["-o", "/tmp/out.wav", "Hi"], command { $0.output = .file(URL(filePath: "/tmp/out.wav")); $0.source = .text("Hi") }),
            (["-o", "/tmp/out.M4A", "Hi"], command { $0.output = .file(URL(filePath: "/tmp/out.M4A")); $0.source = .text("Hi") }),
            (["-o", "-", "Hi"], command { $0.output = .standardOutput; $0.source = .text("Hi") }),
            (["--expressiveness", "0.8", "Hi"], command { $0.expressiveness = 0.8; $0.source = .text("Hi") }),
            (["--quick", "Hi"], command { $0.quickPreview = true; $0.source = .text("Hi") }),
            (["--seed", "42", "Hi"], command { $0.seed = 42; $0.source = .text("Hi") }),
            (["--note", "softer", "Hi"], command { $0.note = "softer"; $0.source = .text("Hi") }),
            (["--retake", "T1", "--note", "softer"], command { $0.retakeOf = "T1"; $0.note = "softer"; $0.source = nil }),
            (["--retake", "T1", "A new line."], command { $0.retakeOf = "T1"; $0.source = .text("A new line.") }),
            (["--verbose", "Hi"], command { $0.verbose = true; $0.source = .text("Hi") }),
            (["--", "-5", "degrees"], command { $0.source = .text("-5 degrees") }),
        ]
        for (arguments, expected) in cases {
            XCTAssertEqual(try SayCommand.parse(arguments), expected, "\(arguments)")
        }
    }

    /// Catches a voice form silently overriding another, which would speak in a
    /// voice the user did not ask for.
    func testTheThreeVoiceFormsAreExclusive() {
        let clashes = [
            ["-v", "Narrator", "--character", "Captain: gravelly", "Hi"],
            ["-v", "Narrator", "--speaker", "0=Narrator", "Hi"],
            ["--character", "Captain: gravelly", "--speaker", "0=Narrator", "Hi"],
            ["--speaker", "0=Narrator", "-v", "Narrator", "--character", "Captain: gravelly", "Hi"],
        ]
        for arguments in clashes {
            XCTAssertThrowsError(try SayCommand.parse(arguments), "\(arguments)") { error in
                let message = (error as? SayUsageError)?.message ?? ""
                XCTAssertTrue(message.contains("both pick the voice"), message)
                XCTAssertFalse(message.contains("\n"), "one line: \(message)")
            }
        }
    }

    /// Catches arguments that should stop the command before it reaches the studio.
    func testBadArgumentsAreUsageErrors() {
        let bad: [([String], String)] = [
            (["-o", "out.mp3", "Hi"], "-o out.mp3: the extension picks the format; use .wav, .m4a or .pcm."),
            (["--expressiveness", "2", "Hi"], "--expressiveness takes a number from 0 to 1."),
            (["--seed", "-1", "Hi"], "--seed takes a whole number from 0 to 9223372036854775807."),
            (["--seed", "18446744073709551615", "Hi"], "--seed takes a whole number from 0 to 9223372036854775807."),
            (["--character", "no colon", "Hi"], "--character takes \"Name: description\"."),
            (["--speaker", "zero=Narrator", "Hi"], "--speaker takes N=VOICE, such as 0=Narrator."),
            (["--speaker", "0=A", "--speaker", "0=B", "Hi"], "--speaker 0 is given twice."),
            (["--scene", "a storm", "Hi"], "--scene goes with --character."),
            (["-f", "/tmp/line.txt", "Hi"], "give the text as words or with -f, not both."),
            (["--wat", "Hi"], "unknown option --wat."),
            (["-v"], "-v needs a value."),
            (["--quick=yes", "Hi"], "--quick takes no value."),
        ]
        for (arguments, message) in bad {
            XCTAssertThrowsError(try SayCommand.parse(arguments), "\(arguments)") { error in
                XCTAssertEqual(error as? SayUsageError, SayUsageError(message: message), "\(arguments)")
            }
        }
        XCTAssertThrowsError(try SayCommand.parse(["-h"])) { XCTAssertEqual($0 as? SayUsageError, SayUsageError(message: nil)) }
    }

    /// Catches a body that drifts from the README's speech-body table: a field
    /// renamed, a field sent that the output does not want (stream with a file),
    /// or a value of the wrong JSON kind.
    func testRequestBodyMatchesTheSpeechBodyTable() {
        let file = URL(filePath: "/tmp/out.m4a")
        let cases: [(SayCommand, String?, JSON)] = [
            (SayCommand(), "Hi", ["input": "Hi", "stream": true, "response_format": "pcm"]),
            (command { $0.voice = .library("Narrator"); $0.output = .file(file) }, "Hi",
             ["input": "Hi", "voice": "Narrator", "response_format": "m4a"]),
            (command { $0.voice = .character(name: "Captain", description: "gravelly", scene: "a storm"); $0.output = .standardOutput }, "Hi",
             ["input": "Hi", "character": ["name": "Captain", "description": "gravelly", "scene": "a storm"], "response_format": "pcm"]),
            (command { $0.voice = .character(name: "Captain", description: "gravelly", scene: nil); $0.output = .file(URL(filePath: "x.WAV")) }, "Hi",
             ["input": "Hi", "character": ["name": "Captain", "description": "gravelly"], "response_format": "wav"]),
            (command { $0.voice = .speakers([0: "Narrator", 1: "abc"]) }, "<|speaker:0|>Hi<|speaker:1|>Yo",
             ["input": "<|speaker:0|>Hi<|speaker:1|>Yo", "speaker_voices": ["0": "Narrator", "1": "abc"], "stream": true, "response_format": "pcm"]),
            (command { $0.expressiveness = 0.25; $0.quickPreview = true; $0.seed = 7; $0.output = .file(URL(filePath: "x.pcm")) }, "Hi",
             ["input": "Hi", "expressiveness": 0.25, "quick_preview": true, "seed": 7, "response_format": "pcm"]),
            (command { $0.note = "softer"; $0.retakeOf = "T1"; $0.output = .standardOutput }, nil,
             ["note": "softer", "previous_takes": [["take_id": "T1"]], "response_format": "pcm"]),
            (command { $0.note = "angrier" }, "Hi", ["input": "Hi", "note": "angrier", "stream": true, "response_format": "pcm"]),
        ]
        for (command, text, body) in cases {
            let call = command.request(text: text)
            XCTAssertEqual(call.method, "POST")
            XCTAssertEqual(call.path, "/v1/audio/speech")
            XCTAssertEqual(call.json, body, "\(command)")
        }
    }

    private func run(_ arguments: [String], replies: [String: APIReply], stdin: String = "") async -> (status: Int32, captured: Captured, api: StubAPI) {
        let api = StubAPI(replies)
        let captured = Captured()
        let environment = SayEnvironment(
            api: api,
            readStandardInput: { Data(stdin.utf8) },
            writeStandardOutput: { captured.stdout($0) },
            writeStandardError: { captured.stderr($0) },
            play: { chunks in for try await chunk in chunks { captured.play(chunk) } }
        )
        return (await SayCommand.main(arguments, environment), captured, api)
    }

    private let busy = APIReply(status: 429, headers: ["Retry-After": "5"], body: Data("Drama Studio is already generating; one request runs at a time. Retry shortly.".utf8))

    /// Catches a busy studio reported as a plain failure, which would stop a
    /// script that could have retried, or a message without the wait.
    func testBusyStudioExitsTempFailWithRetryAfter() async {
        for arguments in [["Hi"], ["-o", "-", "Hi"]] {
            let (status, captured, _) = await run(arguments, replies: ["POST /v1/audio/speech": busy])
            XCTAssertEqual(status, 75, "\(arguments)")
            XCTAssertEqual(captured.standardError, "drama-say: Drama Studio is already generating; one request runs at a time. Retry shortly. Retry-After: 5 s\n")
            XCTAssertTrue(captured.played.isEmpty, "nothing plays on a 429")
            XCTAssertTrue(captured.standardOutput.isEmpty, "nothing written on a 429")
        }
    }

    /// Catches a 4xx swallowed or decorated past the server's own message.
    func testBadRequestPrintsTheServerMessageAndExitsOne() async {
        let reply = APIReply(status: 400, body: Data("No voice is called Nobody.\n".utf8))
        let (status, captured, _) = await run(["-v", "Nobody", "Hi"], replies: ["POST /v1/audio/speech": reply])
        XCTAssertEqual(status, 1)
        XCTAssertEqual(captured.standardError, "drama-say: No voice is called Nobody.\n")
    }

    /// Catches noise on success, audio lost on the way to the speakers, a file
    /// not written, and text not read from standard input.
    func testSuccessIsSilentAndDeliversTheAudio() async throws {
        let audio = Data([1, 0, 2, 0, 3, 0])
        let ok = APIReply(status: 200, headers: ["X-Drama-Take-Id": "T9"], body: audio)

        let played = await run([], replies: ["POST /v1/audio/speech": ok], stdin: "  From a pipe.\n")
        XCTAssertEqual(played.status, 0)
        XCTAssertEqual(played.captured.standardError, "")
        XCTAssertEqual(played.captured.played.reduce(Data(), +), audio)
        XCTAssertEqual(played.api.calls.first?.json?["input"], "From a pipe.")

        let piped = await run(["-o", "-", "Hi"], replies: ["POST /v1/audio/speech": ok])
        XCTAssertEqual(piped.status, 0)
        XCTAssertEqual(piped.captured.standardOutput, audio)

        let url = FileManager.default.temporaryDirectory.appending(path: "\(UUID().uuidString).wav")
        defer { try? FileManager.default.removeItem(at: url) }
        let written = await run(["-o", url.path, "Hi"], replies: ["POST /v1/audio/speech": ok])
        XCTAssertEqual(written.status, 0)
        XCTAssertEqual(try Data(contentsOf: url), audio)

        let verbose = await run(["--verbose", "-o", "-", "Hi"], replies: ["POST /v1/audio/speech": ok])
        XCTAssertTrue(verbose.captured.standardError.contains("take T9"), verbose.captured.standardError)
    }

    /// Catches a usage error that reaches the studio or exits with the wrong status.
    func testUsageErrorExits64WithoutCallingTheStudio() async {
        let (status, captured, api) = await run(["-v", "A", "--character", "B: c", "Hi"], replies: [:])
        XCTAssertEqual(status, 64)
        XCTAssertTrue(captured.standardError.hasPrefix("drama-say: -v and --character both pick the voice"), captured.standardError)
        XCTAssertTrue(api.calls.isEmpty)
        let empty = await run([], replies: [:], stdin: " \n")
        XCTAssertEqual(empty.status, 64)
        XCTAssertTrue(empty.api.calls.isEmpty)
    }

    /// Catches the voice list printing something other than name, id and length.
    func testListVoicesPrintsNameIdSeconds() async {
        let voices = APIReply(status: 200, body: Data(#"{"voices":[{"id":"v1","name":"Narrator","seconds":12.34,"transcript":"x"},{"id":"v2","name":"Al","seconds":5}]}"#.utf8))
        let (status, captured, _) = await run(["-v", "?"], replies: ["GET /v1/voices": voices])
        XCTAssertEqual(status, 0)
        XCTAssertEqual(String(decoding: captured.standardOutput, as: UTF8.self), "Narrator  v1  12.3 s\nAl        v2  5.0 s\n")
    }
}

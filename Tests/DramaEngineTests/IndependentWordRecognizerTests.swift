// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Darwin
import XCTest
import WordRecognitionProtocol
@testable import DramaEngine

final class IndependentWordRecognizerTests: XCTestCase {
    private var directory: URL!
    override func setUpWithError() throws {
        directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory.appendingPathComponent("WordVerification"), withIntermediateDirectories: true)
        try Data([0]).write(to: directory.appendingPathComponent("WordVerification/ggml-small.en.bin"))
    }
    override func tearDownWithError() throws { try FileManager.default.removeItem(at: directory) }

    private func worker(_ mode: String) throws {
        let ready = try JSONEncoder().encode(RecognitionReply.ready(.init(model: WordRecognitionAssets.englishModel, languages: ["en"])))
        var length = UInt32(ready.count).bigEndian
        let frame = withUnsafeBytes(of: &length) { Data($0) } + ready
        func hex(_ data: Data) -> String { data.map { String(format: "%02x", $0) }.joined() }
        let script = """
        #!/usr/bin/python3
        import json, os, signal, struct, time, uuid
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        with open(bytes.fromhex("\(hex(Data(directory.appendingPathComponent("pids").path.utf8)))").decode(), "a") as log:
            log.write(str(os.getpid()) + "\\n")
        if "\(mode)" == "startup":
            time.sleep(10)
        else:
            os.write(0, bytes.fromhex("\(hex(frame))"))
            def read_exact(count):
                value = b""
                while len(value) < count:
                    part = os.read(0, count - len(value))
                    if not part: raise SystemExit(0)
                    value += part
                return value
            request = json.loads(read_exact(struct.unpack(">I", read_exact(4))[0]))
            if "\(mode)" == "cancel": time.sleep(10)
            if "\(mode)" == "late": time.sleep(0.3)
            identifier = str(uuid.uuid4()).upper() if "\(mode)" == "wrong-id" else request["id"]
            reply = json.dumps({"transcript": {"id": identifier, "text": "The garden gate is closed."}}).encode()
            os.write(0, struct.pack(">I", len(reply)) + reply)
            time.sleep(10)
        """
        let helper = directory.appendingPathComponent(WordRecognitionAssets.helperName)
        try Data(script.utf8).write(to: helper)
        try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: helper.path)
    }
    private func assertExited(count: Int, file: StaticString = #filePath, line: UInt = #line) throws {
        let pids = try String(contentsOf: directory.appendingPathComponent("pids"), encoding: .utf8)
            .split(separator: "\n").compactMap { Int32($0) }
        XCTAssertEqual(pids.count, count, "Every call must launch, rather than time out waiting for leaked capacity.", file: file, line: line)
        let deadline = ContinuousClock.now + .seconds(2)
        while pids.contains(where: { kill($0, 0) == 0 }), ContinuousClock.now < deadline { usleep(10_000) }
        for pid in pids { XCTAssertEqual(kill(pid, 0), -1, "Child \(pid) still exists", file: file, line: line) }
    }

    func testUnresponsiveChildrenAreReapedAndFreshRequestsCanRun() throws {
        try worker("startup")
        for _ in 0..<6 {
            XCTAssertThrowsError(try IndependentWordRecognizer(language: "en", deadline: .now + .seconds(2),
                                                               shouldContinue: { true }, runtimeDirectory: directory)) {
                guard case SpeechVerificationError.timedOut = $0 else { return XCTFail("Unexpected \($0)") }
            }
        }
        try assertExited(count: 6)
        try worker("valid")
        let verifier = try IndependentWordRecognizer(language: "en", deadline: .now + .seconds(2), shouldContinue: { true }, runtimeDirectory: directory)
        XCTAssertEqual(try verifier.transcript(pcm16: Data([0, 0]), deadline: .now + .seconds(1), shouldContinue: { true }), "The garden gate is closed.")
        verifier.cancel()
        try assertExited(count: 7)
    }

    func testCancellationDuringAnUnresponsiveRequestTerminatesTheChild() throws {
        try worker("cancel")
        let verifier = try IndependentWordRecognizer(language: "en", deadline: .now + .seconds(2), shouldContinue: { true }, runtimeDirectory: directory)
        let cutoff = ContinuousClock.now + .milliseconds(100)
        XCTAssertThrowsError(try verifier.transcript(pcm16: Data([0, 0]), deadline: .now + .seconds(2), shouldContinue: { .now < cutoff })) {
            XCTAssertTrue($0 is CancellationError)
        }
        try assertExited(count: 1)
    }

    func testWrongReplyIdentityIsRejectedAndTheChildIsTerminated() throws {
        try worker("wrong-id")
        let verifier = try IndependentWordRecognizer(language: "en", deadline: .now + .seconds(2), shouldContinue: { true }, runtimeDirectory: directory)
        XCTAssertThrowsError(try verifier.transcript(pcm16: Data([0, 0]), deadline: .now + .seconds(1), shouldContinue: { true })) {
            guard case SpeechVerificationError.unavailable = $0 else { return XCTFail("Unexpected \($0)") }
        }
        try assertExited(count: 1)
    }

    func testMatchingReplyAfterTheSharedDeadlineCannotBeReturned() throws {
        try worker("late")
        let verifier = try IndependentWordRecognizer(language: "en", deadline: .now + .seconds(2), shouldContinue: { true }, runtimeDirectory: directory)
        XCTAssertThrowsError(try verifier.transcript(pcm16: Data([0, 0]), deadline: .now + .milliseconds(100), shouldContinue: { true })) {
            guard case SpeechVerificationError.timedOut = $0 else { return XCTFail("Unexpected \($0)") }
        }
        try assertExited(count: 1)
    }

    /// The verifier decides from this list, without starting whisper, whether
    /// any recognizer can check a language; whisper itself must agree with it.
    func testTheMultilingualLanguageListIsTheOneWhisperReports() throws {
        let (helper, model) = IndependentWordRecognizer.runtimeFiles(language: "es")
        guard FileManager.default.isExecutableFile(atPath: helper.path), FileManager.default.fileExists(atPath: model.path) else {
            throw XCTSkip("Stage the recognizer with tools/stage_word_recognition.py.")
        }
        let recognizer = try IndependentWordRecognizer(language: "es", deadline: .now + .seconds(30), shouldContinue: { true })
        defer { recognizer.cancel() }
        XCTAssertEqual(Set(try XCTUnwrap(recognizer.identity).languages), WordRecognitionAssets.multilingualLanguages)
    }
}

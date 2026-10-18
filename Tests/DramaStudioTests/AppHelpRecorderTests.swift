import DramaEngine
import XCTest
@testable import DramaStudio

final class AppSettingHelpTests: XCTestCase {
    /// Fields the app sets itself, and why none of them is a control.
    private let hiddenConfigFields: [String: String] = [
        "temperature": "Derived from Expressiveness.",
        "topP": "Derived from Expressiveness.",
        "topK": "Fixed at the vendor's 30; top-p already bounds the choices.",
        "maxNewFrames": "The CLI's fixed cap; the app derives frameCap from the text.",
        "chunkBytes": "Only used by reference chunking, which the app never picks.",
        "chunking": "Fixed by measurement: one sentence per turn cut mean WER from 0.141 to 0.063.",
        "stopAfterSilenceSeconds": "Fixed by measurement: with the retry it took the mean quiet tail from 30 frames to 2.",
        "minSpokenRatio": "Fixed by measurement at 0.6; Quick preview sets it to 0.",
        "maxRetries": "Fixed at 2 alongside the retry threshold.",
        "stopOnLoop": "Fixed by measurement: it ends laughter loops and no speech run tripped it.",
        "retryWithoutEOSBelow": "Fixed by measurement at 1.0 alongside the retry threshold; Quick preview turns retries off.",
        "cueFold": "A package-only knob of the eval/cues experiment.",
        "capMinimumSeconds": "A package-only knob of the eval/cues experiment.",
        "capSecondsPerCue": "A package-only knob of the eval/cues experiment.",
        "slowCapMultiplier": "A package-only knob of the eval/cues experiment.",
    ]

    func testEveryGenerationConfigFieldIsExplainedOrDeliberatelyHidden() {
        let stored = Set(Mirror(reflecting: GenerationConfig()).children.compactMap(\.label))
        let cliAliases: Set = ["maxNewFrames"]
        let explained = Set(SettingHelp.allCases.compactMap(\.configField))
        XCTAssertEqual(explained.union(hiddenConfigFields.keys), stored.union(cliAliases))
        XCTAssertTrue(explained.isDisjoint(with: hiddenConfigFields.keys))
    }

    func testStudioConfigKeepsTheMeasuredDefaults() {
        XCTAssertEqual(GenerationConfig.make(expressiveness: GenerationConfig.defaultExpressiveness, quickPreview: false), GenerationConfig())
        XCTAssertNil(GenerationConfig.make(expressiveness: 0.5, quickPreview: false).seed, "every take draws a fresh seed")
    }

    func testExpressivenessSpansTemperatureAndTopPFromHalfToNineTenths() {
        for (value, expected) in [(0.0, 0.5), (1.0, 0.9), (0.25, 0.6)] {
            let config = GenerationConfig.make(expressiveness: value, quickPreview: false)
            XCTAssertEqual(config.temperature, expected, accuracy: 1e-12)
            XCTAssertEqual(config.topP, expected, accuracy: 1e-12)
            XCTAssertEqual(config.topK, 30)
        }
    }

    func testQuickPreviewDropsRetriesAndTightensTheFrameCap() {
        let preview = GenerationConfig.make(expressiveness: 0.5, quickPreview: true)
        XCTAssertEqual(preview.minSpokenRatio, 0)
        XCTAssertEqual(preview.frameCap, .fromText(margin: 1.2, floorSeconds: 1.0))
        var expected = GenerationConfig()
        expected.minSpokenRatio = 0
        expected.frameCap = .fromText(margin: 1.2, floorSeconds: 1.0)
        XCTAssertEqual(preview, expected, "preview changes nothing else")
    }

    func testEntriesAreUnique() {
        for (name, values) in [
            ("title", SettingHelp.allCases.map(\.title)),
            ("text", SettingHelp.allCases.map(\.text)),
            ("config field", SettingHelp.allCases.compactMap(\.configField)),
        ] {
            let duplicates = Dictionary(grouping: values, by: { $0 }).filter { $0.value.count > 1 }.keys
            XCTAssertTrue(duplicates.isEmpty, "duplicate \(name): \(duplicates.sorted())")
        }
    }

    func testEachEntryIsTwoToFourSentences() {
        let sentenceEnd = try! Regex(#"[.!?](\s+|$)"#)
        for topic in SettingHelp.allCases {
            let sentences = topic.text.split(separator: sentenceEnd).filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty }
            XCTAssertTrue((2...4).contains(sentences.count), "\(topic.title) has \(sentences.count) sentences")
        }
    }

    func testHelpWindowOpensOnGettingStartedAndCoversLicense() {
        XCTAssertEqual(SettingHelp.allCases.first, .gettingStarted)
        XCTAssertEqual(SettingHelp.Group.allCases.first, SettingHelp.gettingStarted.group)
        XCTAssertTrue(SettingHelp.license.text.contains("Fish Audio Community License"))
    }

    func testExpressivenessHelpNamesTheEngineDefaults() {
        let defaults = GenerationConfig()
        XCTAssertTrue(SettingHelp.expressiveness.text.contains("temperature \(defaults.temperature) and top-p \(defaults.topP)"))
    }
}

final class AppRecordingLengthTests: XCTestCase {
    func testWarnsOnlyOutsideThreeToThirtySeconds() {
        XCTAssertNotNil(RecordingLength.warning(for: 0.5))
        XCTAssertNotNil(RecordingLength.warning(for: 2.99))
        XCTAssertNil(RecordingLength.warning(for: 3))
        XCTAssertNil(RecordingLength.warning(for: 12))
        XCTAssertNil(RecordingLength.warning(for: 30))
        XCTAssertNotNil(RecordingLength.warning(for: 30.01))
        XCTAssertNotEqual(RecordingLength.warning(for: 1), RecordingLength.warning(for: 40))
    }
}

@MainActor
final class AppVoiceRecorderTests: XCTestCase {
    private var directory: URL!
    private let startTime = Date(timeIntervalSince1970: 1_800_000_000)

    override func setUp() async throws {
        directory = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
    }

    override func tearDown() async throws {
        try? FileManager.default.removeItem(at: directory)
    }

    private func makeRecorder(_ capture: StubCapture, granted: Bool = true, onRequest: @escaping (VoiceRecorder) -> Void = { _ in }) -> VoiceRecorder {
        var recorder: VoiceRecorder!
        recorder = VoiceRecorder(capture: capture, directory: directory, requestAccess: {
            onRequest(recorder)
            return granted
        }, now: { [startTime] in startTime })
        return recorder
    }

    func testGrantedRecordingRunsIdleToRecordingToRecorded() async throws {
        let capture = StubCapture(duration: 7.5)
        var stateDuringRequest: VoiceRecorder.State?
        let recorder = makeRecorder(capture) { stateDuringRequest = $0.state }
        XCTAssertEqual(recorder.state, .idle)

        await recorder.record()
        XCTAssertEqual(stateDuringRequest, .requestingPermission)
        XCTAssertEqual(recorder.state, .recording(startedAt: startTime))
        let url = try XCTUnwrap(capture.url)
        XCTAssertEqual(url.deletingLastPathComponent().standardizedFileURL, directory.standardizedFileURL)
        XCTAssertEqual(url.pathExtension, "caf")

        capture.report(level: 0.4)
        XCTAssertEqual(recorder.level, 0.4)

        recorder.stop()
        XCTAssertEqual(recorder.state, .recorded(url: url, duration: 7.5))
        XCTAssertEqual(recorder.level, 0)
        XCTAssertEqual(capture.stops, 1)
    }

    func testDeniedPermissionNeverTouchesTheMicrophone() async {
        let capture = StubCapture(duration: 5)
        let recorder = makeRecorder(capture, granted: false)
        await recorder.record()
        XCTAssertEqual(recorder.state, .denied)
        XCTAssertNil(capture.url)
    }

    func testCaptureFailureAndSilentCaptureBothFail() async {
        let broken = StubCapture(duration: 5, startError: CocoaError(.fileWriteNoPermission))
        let recorder = makeRecorder(broken)
        await recorder.record()
        guard case .failed(let message) = recorder.state else { return XCTFail("\(recorder.state)") }
        XCTAssertTrue(message.hasPrefix("Couldn't start recording"))

        let silent = makeRecorder(StubCapture(duration: 0))
        await silent.record()
        silent.stop()
        XCTAssertEqual(silent.state, .failed(message: "The microphone delivered no audio."))
    }

    func testStopOutsideRecordingDoesNothing() {
        let capture = StubCapture(duration: 5)
        let recorder = makeRecorder(capture)
        recorder.stop()
        XCTAssertEqual(recorder.state, .idle)
        XCTAssertEqual(capture.stops, 0)
    }

    func testRecordingAgainDeletesTheUnusedTakeButKeptTakesSurvive() async throws {
        let capture = StubCapture(duration: 6)
        let recorder = makeRecorder(capture)
        await recorder.record()
        recorder.stop()
        let first = try XCTUnwrap(capture.url)
        XCTAssertTrue(FileManager.default.fileExists(atPath: first.path))

        try FileManager.default.removeItem(at: directory)
        await recorder.record()
        recorder.stop()
        let second = try XCTUnwrap(capture.url)
        XCTAssertTrue(FileManager.default.fileExists(atPath: second.path))
        recorder.keep()
        XCTAssertEqual(recorder.state, .idle)

        recorder.discard()
        XCTAssertTrue(FileManager.default.fileExists(atPath: second.path), "a kept take belongs to the voice")

        await recorder.record()
        recorder.stop()
        let third = try XCTUnwrap(capture.url)
        recorder.discard()
        XCTAssertFalse(FileManager.default.fileExists(atPath: third.path))
        XCTAssertEqual(recorder.state, .idle)
    }

    func testDiscardWhileRecordingStopsTheMicrophoneAndDeletesTheFile() async throws {
        let capture = StubCapture(duration: 4)
        let recorder = makeRecorder(capture)
        await recorder.record()
        let url = try XCTUnwrap(capture.url)
        recorder.discard()
        XCTAssertEqual(capture.stops, 1)
        XCTAssertEqual(recorder.state, .idle)
        XCTAssertFalse(FileManager.default.fileExists(atPath: url.path))
    }
}

private final class StubCapture: AudioCapturing {
    let duration: TimeInterval
    let startError: Error?
    private(set) var url: URL?
    private(set) var stops = 0
    private var level: (@MainActor (Float) -> Void)?

    init(duration: TimeInterval, startError: Error? = nil) {
        self.duration = duration
        self.startError = startError
    }

    func start(writingTo url: URL, level: @escaping @MainActor (Float) -> Void) throws {
        if let startError { throw startError }
        try Data().write(to: url)
        self.url = url
        self.level = level
    }

    func stop() -> TimeInterval {
        stops += 1
        return duration
    }

    @MainActor func report(level value: Float) {
        level?(value)
    }
}

@MainActor
final class AppGenerationClockTests: XCTestCase {
    func testSummaryShowsAudioFramesSpeedAndGenerationTime() {
        XCTAssertEqual(
            TransportBar.summary(audioSeconds: 23.6, frames: 559, framesPerSecond: 38.2, generationSeconds: 15.2),
            "23.6 s of audio \u{00B7} 559 frames \u{00B7} 38 frames/s \u{00B7} generated in 15.2 s"
        )
        XCTAssertEqual(
            TransportBar.summary(audioSeconds: 4, frames: nil, framesPerSecond: nil, generationSeconds: 2.04),
            "4.0 s of audio (stopped early) \u{00B7} generated in 2.0 s"
        )
        XCTAssertEqual(TransportBar.seconds(12.44), "12.4 s")
    }

    func testClockRunsFromGenerateUntilTheRunSettles() async {
        let session = SessionModel(synthesizer: MockSynthesizer(pace: .instant), library: VoiceLibrary(directory: FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)))
        session.playsWhileGenerating = false
        XCTAssertNil(session.generationClock)
        session.script = "One line to say."
        session.generate()
        let clock = session.generationClock
        XCTAssertNotNil(clock)
        XCTAssertNil(clock?.duration, "still running")
        let deadline = Date().addingTimeInterval(10)
        while session.state.isGenerating, Date() < deadline { try? await Task.sleep(for: .milliseconds(10)) }
        XCTAssertEqual(session.generationClock?.start, clock?.start)
        let duration = session.generationClock?.duration
        XCTAssertNotNil(duration)
        XCTAssertGreaterThanOrEqual(duration ?? -1, 0)

        session.generate()
        XCTAssertNotEqual(session.generationClock?.start, clock?.start, "a new take restarts the clock")
        XCTAssertNil(session.generationClock?.duration)
    }

    func testElapsedFreezesAtTheEnd() {
        let start = Date(timeIntervalSince1970: 100)
        var clock = GenerationClock(start: start)
        XCTAssertEqual(clock.elapsed(at: start.addingTimeInterval(3)), 3)
        clock.end = start.addingTimeInterval(5)
        XCTAssertEqual(clock.elapsed(at: start.addingTimeInterval(60)), 5)
        XCTAssertEqual(clock.duration, 5)
    }
}

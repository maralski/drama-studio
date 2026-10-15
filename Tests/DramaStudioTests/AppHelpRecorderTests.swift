import DramaEngine
import XCTest
@testable import DramaStudio

final class AppSettingHelpTests: XCTestCase {
    /// `maxRetries` has no control in the inspector; the retry threshold's entry mentions it.
    private let unsurfacedConfigFields: Set<String> = ["maxRetries"]

    func testEveryGenerationConfigFieldIsExplainedOrDeliberatelyHidden() {
        let fields = Set(Mirror(reflecting: GenerationConfig()).children.compactMap(\.label))
        let explained = SettingHelp.allCases.compactMap(\.configField)
        XCTAssertEqual(Set(explained), fields.subtracting(unsurfacedConfigFields))
        XCTAssertTrue(unsurfacedConfigFields.isSubset(of: fields), "stale exclusion: \(unsurfacedConfigFields.subtracting(fields))")
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

    func testMeasuredDefaultsMatchTheEngine() {
        let defaults = GenerationConfig()
        XCTAssertTrue(SettingHelp.temperature.text.contains("default is \(defaults.temperature)"))
        XCTAssertTrue(SettingHelp.topP.text.contains("default is \(defaults.topP)"))
        XCTAssertTrue(SettingHelp.topK.text.contains("default is \(defaults.topK)"))
        XCTAssertTrue(SettingHelp.maxFrames.text.contains("default is \(defaults.maxNewFrames) frames"))
        XCTAssertTrue(SettingHelp.chunkBytes.text.contains("default is \(defaults.chunkBytes) bytes"))
        XCTAssertTrue(SettingHelp.retryThreshold.text.contains("default is \(defaults.minSpokenRatio)"))
        XCTAssertTrue(SettingHelp.stopAfterSilence.text.contains("default is \(Int(defaults.stopAfterSilenceSeconds ?? 0)) seconds"))
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

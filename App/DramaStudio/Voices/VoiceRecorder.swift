// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import AVFoundation
import Foundation

/// The audio half of the recorder, so the state machine can be tested without a microphone.
protocol AudioCapturing: AnyObject {
    /// Starts writing mono audio to `url`. `level` receives 0...1 meter readings on the main actor.
    func start(writingTo url: URL, level: @escaping @MainActor (Float) -> Void) throws
    /// Stops and returns the seconds of audio written.
    func stop() -> TimeInterval
}

@MainActor @Observable
final class VoiceRecorder {
    enum State: Equatable {
        case idle
        case requestingPermission
        case denied
        case recording(startedAt: Date)
        case recorded(url: URL, duration: TimeInterval)
        case failed(message: String)
    }

    private(set) var state: State = .idle
    private(set) var level: Float = 0

    @ObservationIgnored private let capture: any AudioCapturing
    @ObservationIgnored private let requestAccess: () async -> Bool
    @ObservationIgnored private let now: () -> Date
    @ObservationIgnored private var pendingURL: URL?
    let directory: URL

    nonisolated static var defaultDirectory: URL {
        URL.applicationSupportDirectory.appending(path: "DramaStudio/Recordings", directoryHint: .isDirectory)
    }

    init(
        capture: any AudioCapturing = MicrophoneCapture(),
        directory: URL = VoiceRecorder.defaultDirectory,
        requestAccess: @escaping () async -> Bool = { await AVCaptureDevice.requestAccess(for: .audio) },
        now: @escaping () -> Date = Date.init
    ) {
        self.capture = capture
        self.directory = directory
        self.requestAccess = requestAccess
        self.now = now
    }

    var isRecording: Bool {
        if case .recording = state { true } else { false }
    }

    func record() async {
        guard !isRecording, state != .requestingPermission else { return }
        discard()
        state = .requestingPermission
        guard await requestAccess() else {
            state = .denied
            return
        }
        guard state == .requestingPermission else { return }
        do {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            let url = directory.appending(path: Self.fileName(at: now()))
            try capture.start(writingTo: url) { [weak self] in self?.level = $0 }
            pendingURL = url
            state = .recording(startedAt: now())
        } catch {
            state = .failed(message: "Couldn't start recording: \(error.localizedDescription)")
        }
    }

    func stop() {
        guard isRecording, let url = pendingURL else { return }
        let duration = capture.stop()
        pendingURL = nil
        level = 0
        state = duration > 0 ? .recorded(url: url, duration: duration) : .failed(message: "The microphone delivered no audio.")
    }

    /// Deletes an unused take and returns to idle. A take handed to the sheet with `keep()` is left on disk.
    func discard() {
        if isRecording { _ = capture.stop() }
        if case .recorded(let url, _) = state { try? FileManager.default.removeItem(at: url) }
        if let pendingURL { try? FileManager.default.removeItem(at: pendingURL) }
        pendingURL = nil
        level = 0
        state = .idle
    }

    func keep() {
        state = .idle
    }

    static func fileName(at date: Date) -> String {
        let stamp = date.formatted(.verbatim(
            "\(year: .defaultDigits)-\(month: .twoDigits)-\(day: .twoDigits) at \(hour: .twoDigits(clock: .twentyFourHour, hourCycle: .zeroBased)).\(minute: .twoDigits).\(second: .twoDigits)",
            timeZone: .current, calendar: .current
        ))
        return "Recording \(stamp).caf"
    }
}

enum RecordingLength {
    static let suggestion = "Read 5 to 20 seconds of clean speech in a quiet room, then type exactly what you said."

    static func warning(for seconds: TimeInterval) -> String? {
        if seconds < 3 {
            "This clip is under 3 seconds, which may be too little for the voice to carry over. 5 to 20 seconds works best."
        } else if seconds > 30 {
            "This clip is over 30 seconds. Encoding takes longer and the extra length rarely helps. 5 to 20 seconds works best."
        } else {
            nil
        }
    }
}

/// Captures the default input through an `AVAudioEngine` tap at the device's
/// native rate, mixed down to mono.
final class MicrophoneCapture: AudioCapturing {
    private let engine = AVAudioEngine()
    private var writer: TapWriter?

    func start(writingTo url: URL, level: @escaping @MainActor (Float) -> Void) throws {
        let input = engine.inputNode
        let format = input.outputFormat(forBus: 0)
        guard format.sampleRate > 0, format.channelCount > 0,
              let mono = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: format.sampleRate, channels: 1, interleaved: false)
        else { throw RecorderError.noInput }
        let writer = TapWriter(file: try AVAudioFile(forWriting: url, settings: mono.settings), format: mono, level: level)
        input.installTap(onBus: 0, bufferSize: 4096, format: format) { buffer, _ in writer.write(buffer) }
        do {
            engine.prepare()
            try engine.start()
        } catch {
            input.removeTap(onBus: 0)
            throw error
        }
        self.writer = writer
    }

    func stop() -> TimeInterval {
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        defer { writer = nil }
        return writer?.finish() ?? 0
    }

    enum RecorderError: LocalizedError {
        case noInput
        var errorDescription: String? { "No microphone is available." }
    }
}

/// Lives on the audio thread: mixes each tap buffer to mono, appends it to the file, and reports its level.
private final class TapWriter: @unchecked Sendable {
    private let file: AVAudioFile
    private let format: AVAudioFormat
    private let level: @MainActor (Float) -> Void
    private let lock = NSLock()
    private var frames: AVAudioFramePosition = 0

    init(file: AVAudioFile, format: AVAudioFormat, level: @escaping @MainActor (Float) -> Void) {
        self.file = file
        self.format = format
        self.level = level
    }

    /// Closes the file so the encoder can read it, and returns the seconds written.
    func finish() -> TimeInterval {
        lock.withLock {
            file.close()
            return Double(frames) / format.sampleRate
        }
    }

    func write(_ buffer: AVAudioPCMBuffer) {
        guard let channels = buffer.floatChannelData, let mono = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: buffer.frameLength),
              let out = mono.floatChannelData?[0] else { return }
        let count = Int(buffer.frameLength)
        let channelCount = Int(buffer.format.channelCount)
        var sumOfSquares: Float = 0
        for i in 0..<count {
            var sample: Float = 0
            for c in 0..<channelCount { sample += channels[c][i] }
            sample /= Float(channelCount)
            out[i] = sample
            sumOfSquares += sample * sample
        }
        mono.frameLength = buffer.frameLength
        lock.withLock {
            guard file.isOpen, (try? file.write(from: mono)) != nil else { return }
            frames += AVAudioFramePosition(count)
        }
        let rms = count > 0 ? (sumOfSquares / Float(count)).squareRoot() : 0
        let meter = max(0, min(1, (20 * log10(max(rms, 1e-6)) + 60) / 60))
        let level = level
        Task { @MainActor in level(meter) }
    }
}

// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import AppKit
import AVFoundation
import SwiftUI

struct RecorderPanel: View {
    let recorder: VoiceRecorder
    let use: (URL) -> Void
    @State private var preview = ClipPreview()

    private static let microphoneSettings = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone")!

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            switch recorder.state {
            case .idle, .requestingPermission:
                Text(RecordingLength.suggestion).font(.caption).foregroundStyle(.secondary)
                recordButton
            case .denied:
                Label("Drama Studio can't use the microphone. Allow it under Privacy & Security > Microphone, then try again.", systemImage: "mic.slash.fill")
                    .foregroundStyle(.orange)
                HStack {
                    Button("Open System Settings") { NSWorkspace.shared.open(Self.microphoneSettings) }
                    recordButton
                }
            case .failed(let message):
                Label(message, systemImage: "exclamationmark.triangle.fill").foregroundStyle(.red)
                recordButton
            case .recording(let startedAt):
                HStack(spacing: 10) {
                    Button("Stop", systemImage: "stop.fill") { recorder.stop() }
                        .tint(.red)
                    ProgressView(value: Double(recorder.level))
                        .tint(recorder.level > 0.9 ? .red : .green)
                        .animation(.linear(duration: 0.08), value: recorder.level)
                    TimelineView(.periodic(from: startedAt, by: 0.1)) { context in
                        secondsLabel(context.date.timeIntervalSince(startedAt))
                    }
                    .frame(width: 52, alignment: .trailing)
                }
            case .recorded(let url, let duration):
                HStack(spacing: 10) {
                    Button(preview.isPlaying ? "Stop" : "Play", systemImage: preview.isPlaying ? "stop.fill" : "play.fill") { preview.toggle(url) }
                    secondsLabel(duration)
                    Spacer()
                    Button("Record Again") {
                        preview.stop()
                        Task { await recorder.record() }
                    }
                    Button("Use This Recording") {
                        preview.stop()
                        recorder.keep()
                        use(url)
                    }
                    .buttonStyle(.borderedProminent)
                }
                if let warning = RecordingLength.warning(for: duration) {
                    Label(warning, systemImage: "exclamationmark.triangle.fill").font(.caption).foregroundStyle(.orange)
                }
            }
        }
        .onDisappear { preview.stop() }
    }

    private func secondsLabel(_ seconds: TimeInterval) -> some View {
        Text("\(seconds.formatted(.number.precision(.fractionLength(1)))) s").monospacedDigit().foregroundStyle(.secondary)
    }

    private var recordButton: some View {
        Button("Start Recording", systemImage: "record.circle") { Task { await recorder.record() } }
            .disabled(recorder.state == .requestingPermission)
    }
}

@MainActor @Observable
private final class ClipPreview: NSObject, AVAudioPlayerDelegate {
    private(set) var isPlaying = false
    @ObservationIgnored private var player: AVAudioPlayer?

    func toggle(_ url: URL) {
        if isPlaying { return stop() }
        player = try? AVAudioPlayer(contentsOf: url)
        player?.delegate = self
        isPlaying = player?.play() ?? false
    }

    func stop() {
        player?.stop()
        player = nil
        isPlaying = false
    }

    nonisolated func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        Task { @MainActor in self.stop() }
    }
}

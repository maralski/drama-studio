// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import DramaEngine
import SwiftUI

struct TransportBar: View {
    @Bindable var model: SessionModel
    @Environment(APIServerController.self) private var server: APIServerController?

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            ViewThatFits(in: .horizontal) {
                HStack(spacing: 10) {
                    playbackButtons
                    Spacer()
                    speedMenu
                    exportMenu
                }
                VStack(alignment: .leading, spacing: 8) {
                    playbackButtons
                    HStack {
                        speedMenu
                        Spacer()
                        exportMenu
                    }
                }
            }
            TimelineView(.animation(paused: !model.state.isPlaying)) { _ in
                    HStack(spacing: 10) {
                        WaveformScrubber(model: model, position: model.position)
                        Text("\(Self.clock(model.position)) / \(Self.clock(model.audio.count))")
                            .font(.callout.monospacedDigit())
                            .foregroundStyle(.secondary)
                    }
                }
            HStack(spacing: 8) {
                StatusLine(model: model)
                if let server { APIChip(server: server) }
                Toggle(isOn: $model.quickPreview) { HelpLabel("Quick preview", help: .quickPreview) }
                    .toggleStyle(.checkbox)
                    .controlSize(.small)
                    .disabled(model.state.isGenerating)
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
        .background(.bar)
        .overlay(alignment: .top) { Divider() }
    }

    private var playbackButtons: some View {
        HStack(spacing: 10) {
            generateButton
            Button("New Take", systemImage: "dice") { model.generate() }
                .labelStyle(.iconOnly)
                .disabled(!model.canTakeAgain)
                .help("Generate the script again for a different read; the takes so far stay below")
            Button(model.state.isPlaying ? "Pause" : "Play", systemImage: model.state.isPlaying ? "pause.fill" : "play.fill") {
                model.togglePlayback()
            }
            .labelStyle(.iconOnly)
            .frame(width: 28)
            .disabled(!model.canPlay)
            .help("Play or pause (Space)")
        }
        .fixedSize()
    }

    @ViewBuilder
    private var generateButton: some View {
        if model.state.isGenerating {
            Button("Stop", systemImage: "stop.fill") { model.stopGeneration() }
                .help("Stop generating (⌘.)")
                .frame(minWidth: 104)
        } else {
            Button(model.document == nil ? "Generate" : "Generate All", systemImage: "waveform") { model.generate() }
                .buttonStyle(.borderedProminent)
                .disabled(!model.canGenerate)
                .help("Generate speech (⌘Return)")
                .frame(minWidth: 104)
        }
    }

    private var speedMenu: some View {
        Picker(selection: $model.playbackRate) {
            ForEach(SessionModel.playbackRates, id: \.self) { rate in
                Text(Self.speed(rate)).tag(rate)
            }
        } label: {
            HelpLabel("Speed", help: .playbackSpeed)
        }
        .accessibilityLabel("Playback speed")
        .pickerStyle(.menu)
        .fixedSize()
        .help("Playback speed; exports keep the original timing")
    }

    private var exportMenu: some View {
        Menu("Export", systemImage: "square.and.arrow.up") {
            ExportMenuItems(model: model)
        } primaryAction: {
            FilePanels.export(model)
        }
        .fixedSize()
        .disabled(!model.canExport)
        .help("\(model.exportSummary) (⌘E)")
    }

    static func speed(_ rate: Double) -> String {
        "\(rate.formatted(.number.precision(.fractionLength(0...2))))x"
    }

    /// The status line after a document run.
    static func sectionSummary(sections: Int, audioSeconds: TimeInterval, stoppedEarly: Bool, generationSeconds: TimeInterval?) -> String {
        var parts = ["\(sections) \(sections == 1 ? "section" : "sections") narrated", "\(seconds(audioSeconds)) of audio"]
        if stoppedEarly { parts[0] += " (stopped early)" }
        if let generationSeconds { parts.append("generated in \(seconds(generationSeconds))") }
        return parts.joined(separator: " \u{00B7} ")
    }

    static func seconds(_ value: TimeInterval) -> String {
        "\(value.formatted(.number.precision(.fractionLength(1)))) s"
    }

    /// A restored project has audio but no generation telemetry or clock.
    static func summary(audioSeconds: TimeInterval, frames: Int?, framesPerSecond: Double?, generationSeconds: TimeInterval?) -> String {
        var parts = ["\(seconds(audioSeconds)) of audio"]
        if let frames, let framesPerSecond {
            parts += ["\(frames) frames", "\(Int(framesPerSecond.rounded())) frames/s"]
        } else if generationSeconds != nil {
            parts[0] += " (stopped early)"
        }
        if let generationSeconds { parts.append("generated in \(seconds(generationSeconds))") }
        return parts.joined(separator: " \u{00B7} ")
    }

    static func clock(_ samples: Int) -> String {
        let seconds = Int(Double(samples) / Double(ModelConstants.sampleRate))
        return String(format: "%d:%02d", seconds / 60, seconds % 60)
    }
}

private struct StatusLine: View {
    let model: SessionModel

    var body: some View {
        HStack(spacing: 8) {
            switch model.state {
            case .idle:
                Text(ModelSetup.isPreviewEngine ? "Preview mode uses placeholder tones instead of speech." : "Ready.")
            case .loading(let progress, let detail, _):
                ProgressView(value: progress).frame(width: 120)
                Text("Loading model \u{2014} \(detail)")
            case .generating(let chunk, let count, _):
                ProgressView(value: chunkProgress(chunk: chunk, of: count)).frame(width: 120)
                TimelineView(.periodic(from: .now, by: 0.1)) { context in
                    let elapsed = model.generationClock.map { " \u{00B7} \(TransportBar.seconds($0.elapsed(at: context.date)))" } ?? ""
                    let left = model.estimateLabel(at: context.date).map { " \u{00B7} \($0)" } ?? ""
                    Text("Generating part \(chunk) of \(count)\(elapsed)\(left)")
                        .monospacedDigit()
                }
            case .ready, .playing:
                Text(summary)
            case .failed(let message):
                Label(message, systemImage: "exclamationmark.octagon.fill")
                    .foregroundStyle(.red)
                    .textSelection(.enabled)
            }
            Spacer()
        }
        .font(.caption)
        .foregroundStyle(.secondary)
        .fixedSize(horizontal: false, vertical: true)
    }

    private func chunkProgress(chunk: Int, of count: Int) -> Double {
        let within = model.telemetry.maxFrames > 0 ? Double(model.telemetry.frame) / Double(model.telemetry.maxFrames) : 0
        return (Double(chunk - 1) + min(within, 1)) / Double(max(count, 1))
    }

    private var summary: String {
        if model.document != nil {
            return TransportBar.sectionSummary(sections: model.narrationTakes.count, audioSeconds: model.duration,
                                               stoppedEarly: model.result == nil && model.generationClock != nil, generationSeconds: model.generationClock?.duration)
        }
        return TransportBar.summary(
            audioSeconds: model.duration,
            frames: model.result?.take.frameCount,
            framesPerSecond: model.result?.timings.framesPerSecond,
            generationSeconds: model.generationClock?.duration
        )
    }
}

/// The transport's Export menu and the File menu share these items.
struct ExportMenuItems: View {
    let model: SessionModel

    var body: some View {
        ForEach(ExportFormat.allCases) { format in
            Button(format.title) { FilePanels.export(model, as: format) }
        }
        .disabled(!model.canExport)
        Divider()
        Button("Per Section\u{2026}") { FilePanels.exportSections(model) }
            .disabled(!model.canExportSections)
        Menu("Captions") {
            Button("SRT\u{2026}") { FilePanels.exportCaptions(model, as: .srt) }
            Button("WebVTT\u{2026}") { FilePanels.exportCaptions(model, as: .webVTT) }
        }
        .disabled(!model.canExportCaptions)
        Button("Slide Narration Package\u{2026}") { FilePanels.exportSlidePackage(model) }
            .disabled(!model.canExportSlidePackage)
    }
}

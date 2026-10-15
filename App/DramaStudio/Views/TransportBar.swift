import DramaEngine
import SwiftUI

struct TransportBar: View {
    let model: SessionModel

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 10) {
                generateButton
                Button(model.state.isPlaying ? "Pause" : "Play", systemImage: model.state.isPlaying ? "pause.fill" : "play.fill") {
                    model.togglePlayback()
                }
                .labelStyle(.iconOnly)
                .frame(width: 28)
                .disabled(!model.canPlay)
                .help("Play or pause (Space)")
                TimelineView(.animation(paused: !model.state.isPlaying)) { _ in
                    HStack(spacing: 10) {
                        WaveformScrubber(model: model, position: model.position)
                        Text("\(Self.clock(model.position)) / \(Self.clock(model.audio.count))")
                            .font(.callout.monospacedDigit())
                            .foregroundStyle(.secondary)
                    }
                }
                exportMenu
            }
            StatusLine(model: model)
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .background(.bar)
        .overlay(alignment: .top) { Divider() }
    }

    @ViewBuilder
    private var generateButton: some View {
        if model.state.isGenerating {
            Button("Stop", systemImage: "stop.fill") { model.stopGeneration() }
                .help("Stop generating (⌘.)")
                .frame(width: 104)
        } else {
            Button("Generate", systemImage: "waveform") { model.generate() }
                .buttonStyle(.borderedProminent)
                .disabled(!model.canGenerate)
                .help("Generate speech (⌘Return)")
                .frame(width: 104)
        }
    }

    private var exportMenu: some View {
        Menu("Export", systemImage: "square.and.arrow.up") {
            ForEach(ExportFormat.allCases) { format in
                Button(format.title) { FilePanels.export(model, as: format) }
            }
        } primaryAction: {
            FilePanels.export(model)
        }
        .fixedSize()
        .disabled(!model.canExport)
        .help("Export audio (⌘E)")
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
                Text(ModelSetup.isPreviewEngine ? "Preview engine: generation produces a placeholder tone until the Metal engine ships." : "Ready.")
            case .loading(let progress, let detail):
                ProgressView(value: progress).frame(width: 120)
                Text("Loading model \u{2014} \(detail)")
            case .generating(let chunk, let count, _):
                ProgressView(value: chunkProgress(chunk: chunk, of: count)).frame(width: 120)
                Text("Chunk \(chunk) of \(count) \u{00B7} frame \(model.telemetry.frame) of \(model.telemetry.maxFrames) \u{00B7} \(model.telemetry.framesPerSecond, format: .number.precision(.fractionLength(1))) frames/s")
                    .monospacedDigit()
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
        .lineLimit(1)
        .truncationMode(.middle)
    }

    private func chunkProgress(chunk: Int, of count: Int) -> Double {
        let within = model.telemetry.maxFrames > 0 ? Double(model.telemetry.frame) / Double(model.telemetry.maxFrames) : 0
        return (Double(chunk - 1) + min(within, 1)) / Double(max(count, 1))
    }

    private var summary: String {
        let seconds = model.duration.formatted(.number.precision(.fractionLength(1)))
        guard let result = model.result else { return "\(seconds) s of audio (stopped early)" }
        let fps = result.timings.framesPerSecond.formatted(.number.precision(.fractionLength(1)))
        return "\(seconds) s of audio \u{00B7} \(result.frames.count) frames \u{00B7} \(fps) frames/s"
    }
}

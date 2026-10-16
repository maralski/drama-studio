import DramaEngine
import SwiftUI

struct TransportBar: View {
    @Bindable var model: SessionModel
    @Environment(APIServerController.self) private var server: APIServerController?

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
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
            HStack(spacing: 8) {
                StatusLine(model: model)
                if let server { APIChip(server: server) }
                Toggle(isOn: $model.quickPreview) { HelpLabel("Quick preview", help: .quickPreview) }
                    .toggleStyle(.checkbox)
                    .controlSize(.small)
                    .disabled(model.state.isGenerating)
            }
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
        .help(model.keptTake == nil ? "Export audio (⌘E)" : "Export the kept take (⌘E)")
    }

    static func seconds(_ value: TimeInterval) -> String {
        "\(value.formatted(.number.precision(.fractionLength(1)))) s"
    }

    /// The status line after a run. `frames` and `framesPerSecond` are nil when generation stopped early.
    static func summary(audioSeconds: TimeInterval, frames: Int?, framesPerSecond: Double?, generationSeconds: TimeInterval?) -> String {
        var parts = ["\(seconds(audioSeconds)) of audio"]
        if let frames, let framesPerSecond {
            parts += ["\(frames) frames", "\(Int(framesPerSecond.rounded())) frames/s"]
        } else {
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
                Text(ModelSetup.isPreviewEngine ? "Preview engine: generation produces a placeholder tone until the Metal engine ships." : "Ready.")
            case .loading(let progress, let detail):
                ProgressView(value: progress).frame(width: 120)
                Text("Loading model \u{2014} \(detail)")
            case .generating(let chunk, let count, _):
                ProgressView(value: chunkProgress(chunk: chunk, of: count)).frame(width: 120)
                TimelineView(.periodic(from: .now, by: 0.1)) { context in
                    let elapsed = model.generationClock.map { " \u{00B7} \(TransportBar.seconds($0.elapsed(at: context.date)))" } ?? ""
                    Text("Chunk \(chunk) of \(count) \u{00B7} frame \(model.telemetry.frame) of \(model.telemetry.maxFrames) \u{00B7} \(model.telemetry.framesPerSecond, format: .number.precision(.fractionLength(1))) frames/s\(elapsed)")
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
        .lineLimit(1)
        .truncationMode(.middle)
    }

    private func chunkProgress(chunk: Int, of count: Int) -> Double {
        let within = model.telemetry.maxFrames > 0 ? Double(model.telemetry.frame) / Double(model.telemetry.maxFrames) : 0
        return (Double(chunk - 1) + min(within, 1)) / Double(max(count, 1))
    }

    private var summary: String {
        TransportBar.summary(
            audioSeconds: model.duration,
            frames: model.result?.frames.count,
            framesPerSecond: model.result?.timings.framesPerSecond,
            generationSeconds: model.generationClock?.duration
        )
    }
}

// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import SwiftUI

struct WaveformScrubber: View {
    let model: SessionModel
    let position: Int
    @State private var peaks: [Float] = []
    @State private var dragFraction: Double?

    var body: some View {
        let progress = dragFraction ?? (model.audio.isEmpty ? 0 : Double(position) / Double(model.audio.count))
        Canvas { context, size in
            let barWidth = size.width / CGFloat(max(peaks.count, 1))
            for (index, peak) in peaks.enumerated() {
                let height = max(1.5, CGFloat(peak) * size.height)
                let rect = CGRect(x: CGFloat(index) * barWidth, y: (size.height - height) / 2, width: max(1, barWidth - 1), height: height)
                let played = CGFloat(index) / CGFloat(peaks.count) < progress
                context.fill(Path(roundedRect: rect, cornerRadius: 0.75), with: .color(played ? .accentColor : .secondary.opacity(0.45)))
            }
            if !peaks.isEmpty {
                for boundary in sectionBoundaries {
                    let x = size.width * boundary
                    context.fill(Path(CGRect(x: x - 0.5, y: 0, width: 1, height: size.height)), with: .color(.primary.opacity(0.5)))
                }
                let x = size.width * progress
                context.fill(Path(CGRect(x: x - 0.75, y: 0, width: 1.5, height: size.height)), with: .color(.primary))
            }
        }
        .frame(height: 34)
        .frame(maxWidth: .infinity)
        .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 5))
        .overlay { GeometryReader { scrubArea(width: $0.size.width) } }
        .task(id: model.audio.count) { peaks = Self.peaks(of: model.audio, bins: 160) }
        .accessibilityElement()
        .accessibilityLabel("Playback position")
        .accessibilityValue("\(Int(progress * 100)) percent")
    }

    /// Where each section after the first begins, as a fraction of the audio.
    private var sectionBoundaries: [Double] {
        guard !model.audio.isEmpty else { return [] }
        return model.narrationTakes.dropFirst().map { Double($0.sampleRange.lowerBound) / Double(model.audio.count) }
    }

    private func scrubArea(width: CGFloat) -> some View {
        Color.clear
            .contentShape(Rectangle())
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { dragFraction = min(1, max(0, $0.location.x / max(width, 1))) }
                    .onEnded { _ in
                        if let dragFraction { model.seek(toFraction: dragFraction) }
                        dragFraction = nil
                    }
            )
            .disabled(!model.canPlay)
    }

    static func peaks(of samples: [Float], bins: Int) -> [Float] {
        guard !samples.isEmpty, bins > 0 else { return [] }
        let size = max(1, (samples.count + bins - 1) / bins)
        let raw = stride(from: 0, to: samples.count, by: size).map { start in
            samples[start..<min(start + size, samples.count)].reduce(0) { max($0, abs($1)) }
        }
        let loudest = raw.max() ?? 0
        return loudest > 0 ? raw.map { $0 / loudest } : raw
    }
}

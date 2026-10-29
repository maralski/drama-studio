// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import AppKit

/// The menu bar mark: the app icon's waveform as a template image, so the
/// menu bar tints it for light, dark and highlighted states.
enum MenuBarIcon {
    /// The bar heights of App/Icon/make_icon.swift, so the two marks match.
    static let heights: [CGFloat] = [150, 300, 460, 300, 520, 360, 200]

    static let image: NSImage = {
        let size = NSSize(width: 20, height: 16)
        let image = NSImage(size: size, flipped: false) { rect in
            let tallest = heights.max()!
            let count = CGFloat(heights.count)
            let gapRatio: CGFloat = 34 / 62
            let barWidth = rect.width / (count + (count - 1) * gapRatio)
            let gap = barWidth * gapRatio
            NSColor.black.setFill()
            for (i, h) in heights.enumerated() {
                let height = rect.height * h / tallest
                let bar = NSRect(x: rect.minX + CGFloat(i) * (barWidth + gap), y: rect.midY - height / 2, width: barWidth, height: height)
                NSBezierPath(roundedRect: bar, xRadius: barWidth / 2, yRadius: barWidth / 2).fill()
            }
            return true
        }
        image.isTemplate = true
        image.accessibilityDescription = "Drama Studio"
        return image
    }()
}

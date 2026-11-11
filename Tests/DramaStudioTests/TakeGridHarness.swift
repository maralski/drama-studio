// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import AppKit
import DramaEngine
import SwiftUI
import XCTest
@testable import DramaStudio

/// A take strip under an editor in a window of its own, driven with
/// synthesized wheel, trackpad and mouse events that reach only that window.
@MainActor
final class TakeGridWindow {
    let host: NSHostingView<AnyView>
    let window: NSWindow

    /// `shown` orders the window in, behind every other window and without
    /// focus, which SwiftUI needs before it follows a mouse drag. It sits past
    /// the edge of every display, so nothing shows.
    init(_ view: some View, size: CGSize, shown: Bool = false) {
        _ = NSApplication.shared
        host = NSHostingView(rootView: AnyView(view))
        window = NSWindow(contentRect: CGRect(origin: .zero, size: size), styleMask: [.borderless], backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        window.contentView = host
        if shown {
            window.setFrameOrigin(CGPoint(x: -30000, y: -30000))
            window.orderBack(nil)
        }
    }

    private static var shot = 0

    func close() { window.close() }

    func settle(_ milliseconds: Int = 500) async throws {
        for _ in 0..<max(1, milliseconds / 25) {
            host.layoutSubtreeIfNeeded()
            try await Task.sleep(for: .milliseconds(25))
        }
    }

    var scroll: NSScrollView {
        get throws { try XCTUnwrap(Self.scrollView(in: host), "the take grid's scroll view") }
    }
    var offset: CGFloat { get throws { try scroll.contentView.bounds.minY } }
    var viewport: CGFloat { get throws { try scroll.contentView.bounds.height } }
    var content: CGFloat { get throws { try XCTUnwrap(scroll.documentView).frame.height } }
    /// The grid in the window's coordinates, whose origin is the bottom left.
    var grid: CGRect { get throws { try scroll.convert(scroll.bounds, to: nil) } }

    private static func scrollView(in view: NSView) -> NSScrollView? {
        (view as? NSScrollView) ?? view.subviews.lazy.compactMap(scrollView(in:)).first
    }

    /// The grid's edges that cut a tile, read from the pixels the window
    /// draws. At rest on whole rows, the line of pixels just inside the top
    /// edge and the line just inside the bottom edge show only the strip's
    /// background, because the grid keeps an inset around its rows.
    func cutEdges() throws -> [String] {
        host.layoutSubtreeIfNeeded()
        let bitmap = try XCTUnwrap(host.bitmapImageRepForCachingDisplay(in: host.bounds))
        host.cacheDisplay(in: host.bounds, to: bitmap)
        let scale = CGFloat(bitmap.pixelsHigh) / host.bounds.height
        if let shots = ProcessInfo.processInfo.environment["DRAMA_GRID_SHOTS"] {
            try FileManager.default.createDirectory(atPath: shots, withIntermediateDirectories: true)
            Self.shot += 1
            try XCTUnwrap(bitmap.representation(using: .png, properties: [:])).write(to: URL(filePath: shots).appending(path: "grid-\(Self.shot).png"))
        }
        let grid = try scroll.convert(scroll.bounds, to: host)
        let top = host.isFlipped ? grid.minY : host.bounds.height - grid.maxY
        func line(_ y: CGFloat) -> [NSColor] {
            stride(from: grid.minX + 1, to: grid.maxX - 20, by: 1).compactMap {
                bitmap.colorAt(x: Int($0 * scale), y: Int(y * scale))?.usingColorSpace(.deviceRGB)
            }
        }
        func cuts(_ y: CGFloat) -> Bool {
            let pixels = line(y)
            guard let background = pixels.first else { return false }
            return pixels.contains {
                abs($0.redComponent - background.redComponent) + abs($0.greenComponent - background.greenComponent)
                    + abs($0.blueComponent - background.blueComponent) > 0.02
            }
        }
        return (cuts(top + 0.5) ? ["a tile is cut at the top edge"] : []) + (cuts(top + grid.height - 0.5) ? ["a tile is cut at the bottom edge"] : [])
    }

    private func scrollEvent(_ delta: Int32, phase: Int64 = 0, momentum: Int64 = 0, continuous: Bool) throws -> NSEvent {
        let event = try XCTUnwrap(CGEvent(scrollWheelEvent2Source: nil, units: continuous ? .pixel : .line, wheelCount: 1, wheel1: delta, wheel2: 0, wheel3: 0))
        event.setIntegerValueField(.scrollWheelEventScrollPhase, value: phase)
        event.setIntegerValueField(.scrollWheelEventMomentumPhase, value: momentum)
        event.setIntegerValueField(.scrollWheelEventIsContinuous, value: continuous ? 1 : 0)
        return try XCTUnwrap(NSEvent(cgEvent: event))
    }

    /// One turn of a mouse wheel with no phases. Negative lines scroll down the grid.
    func wheel(lines: Int32) async throws {
        try scroll.scrollWheel(with: scrollEvent(lines, continuous: false))
        try await settle()
    }

    /// Two fingers on a trackpad moving the grid by each of `points`, then
    /// lifting, with the glide of `momentum` after the lift when given.
    func trackpad(points: [Int32], momentum: [Int32] = []) async throws {
        let scroll = try scroll
        scroll.scrollWheel(with: try scrollEvent(0, phase: 1, continuous: true))
        for delta in points {
            scroll.scrollWheel(with: try scrollEvent(delta, phase: 2, continuous: true))
            try await settle(25)
        }
        scroll.scrollWheel(with: try scrollEvent(0, phase: 4, continuous: true))
        for (index, delta) in momentum.enumerated() {
            scroll.scrollWheel(with: try scrollEvent(delta, momentum: index == 0 ? 1 : 2, continuous: true))
            try await settle(25)
        }
        if !momentum.isEmpty { scroll.scrollWheel(with: try scrollEvent(0, momentum: 3, continuous: true)) }
        try await settle()
    }

    /// Scrolls the way a dragged scroller does: the clip view moves with no gesture.
    func scroller(to offset: CGFloat) async throws {
        let scroll = try scroll
        scroll.contentView.scroll(to: CGPoint(x: 0, y: min(max(offset, 0), max(try content - viewport, 0))))
        scroll.reflectScrolledClipView(scroll.contentView)
        try await settle()
    }

    func mouse(_ type: NSEvent.EventType, at point: CGPoint) throws {
        window.sendEvent(try XCTUnwrap(NSEvent.mouseEvent(
            with: type, location: point, modifierFlags: [], timestamp: ProcessInfo.processInfo.systemUptime,
            windowNumber: window.windowNumber, context: nil, eventNumber: 0, clickCount: 1, pressure: type == .leftMouseUp ? 0 : 1)))
    }

    /// Presses at `point`, moves up the window by `rise` in steps, and reports
    /// the grid's height under the held pointer before releasing.
    @discardableResult
    func drag(from point: CGPoint, rise: CGFloat) async throws -> CGFloat {
        try mouse(.leftMouseDown, at: point)
        try await settle(75)
        let steps = max(2, Int(abs(rise) / 10))
        for step in 1...steps {
            try mouse(.leftMouseDragged, at: CGPoint(x: point.x, y: point.y + rise * CGFloat(step) / CGFloat(steps)))
            try await settle(25)
        }
        try await settle(150)
        let held = try viewport
        try mouse(.leftMouseUp, at: CGPoint(x: point.x, y: point.y + rise))
        try await settle()
        return held
    }
}

@MainActor
enum TakeFixture {
    static func model(in scratch: URL) -> SessionModel {
        let model = SessionModel(synthesizer: MockSynthesizer(pace: .instant), library: VoiceLibrary(directory: scratch),
                                 pronunciationURL: scratch.appending(path: "pronunciation.json"))
        model.playsWhileGenerating = false
        model.script = "A short line for comparing takes."
        return model
    }

    static func finish(_ model: SessionModel) async throws {
        for _ in 0..<500 where model.state.isGenerating { try await Task.sleep(for: .milliseconds(10)) }
        XCTAssertFalse(model.state.isGenerating)
    }

    /// Plain takes, which all make tiles of one height.
    static func plain(_ count: Int, in model: SessionModel) async throws {
        for _ in 0..<count {
            model.generate()
            try await finish(model)
        }
    }

    /// Takes whose tiles differ in height: five with a status line, four
    /// plain, then four with a note long enough to wrap.
    static func mixed(in model: SessionModel) async throws {
        try await plain(5, in: model)
        model.script += " Revised."
        try await plain(4, in: model)
        for _ in 0..<4 {
            model.retake(note: "Speak softly and warmly, with a long pause before the last sentence and a gentle finish.")
            try await finish(model)
        }
    }
}

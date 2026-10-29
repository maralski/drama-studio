// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import AppKit
import SwiftUI

struct ContentView: View {
    /// Wide enough for the sidebar, the editor with its transport bar and the
    /// inspector at their minimum widths. Below it SwiftUI's split view keeps
    /// re-solving the columns' sizes and AppKit aborts the app.
    static let minimumSize = CGSize(width: 1280, height: 640)

    @Bindable var model: SessionModel
    let setup: SetupCoordinator
    @State private var showsInspector = true
    @State private var newVoiceClip: NewVoiceRequest?

    var body: some View {
        if setup.stage == .ready {
            studio
        } else {
            SetupView(setup: setup, model: model)
        }
    }

    private var studio: some View {
        NavigationSplitView {
            VoiceLibraryView(model: model, newVoice: $newVoiceClip)
                .navigationSplitViewColumnWidth(min: 210, ideal: 240, max: 320)
        } detail: {
            ScriptEditorView(model: model)
                .safeAreaInset(edge: .bottom, spacing: 0) {
                    VStack(spacing: 0) {
                        TransportBar(model: model)
                        if !model.visibleTakes.isEmpty { TakesStrip(model: model) }
                    }
                }
                .inspector(isPresented: $showsInspector) {
                    InspectorView(model: model)
                        .inspectorColumnWidth(min: 270, ideal: 300, max: 380)
                }
        }
        .frame(minWidth: Self.minimumSize.width, minHeight: Self.minimumSize.height)
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                Button("Import Script", systemImage: "square.and.arrow.down") { FilePanels.importScript(into: model) }
                    .help("Import a script, or a PowerPoint, PDF, or EPUB document to narrate by section (⌘O)")
            }
            ToolbarItem(placement: .primaryAction) {
                Menu("Examples", systemImage: "text.badge.star") { ExampleMenuItems(model: model) }
                    .help("Insert an example line or cast a character")
            }
            ToolbarItem(placement: .primaryAction) {
                Button("Inspector", systemImage: "sidebar.trailing") { showsInspector.toggle() }
                    .help("Show or hide voice and tuning controls")
            }
        }
        .sheet(item: $newVoiceClip) { request in
            AddVoiceSheet(model: model, clip: request.clip)
        }
        .onSpaceKey { model.togglePlayback() }
    }
}

struct NewVoiceRequest: Identifiable {
    let id = UUID()
    var clip: URL?
}

extension View {
    /// A key monitor rather than a menu shortcut, which would swallow spaces
    /// typed into the editor and text fields.
    func onSpaceKey(perform action: @escaping () -> Void) -> some View {
        modifier(SpaceKeyMonitor(action: action))
    }
}

private struct SpaceKeyMonitor: ViewModifier {
    let action: () -> Void
    @State private var monitor: Any?

    func body(content: Content) -> some View {
        content
            .onAppear {
                monitor = NSEvent.addLocalMonitorForEvents(matching: .keyDown) { event in
                    let plain = event.modifierFlags.intersection(.deviceIndependentFlagsMask).subtracting([.capsLock, .numericPad, .function]).isEmpty
                    guard event.keyCode == 49, plain,
                          let window = event.window, window.attachedSheet == nil, !(window is NSPanel),
                          !(window.firstResponder is NSText) else { return event }
                    action()
                    return nil
                }
            }
            .onDisappear {
                if let monitor { NSEvent.removeMonitor(monitor) }
                monitor = nil
            }
    }
}

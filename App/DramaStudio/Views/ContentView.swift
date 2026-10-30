// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import AppKit
import SwiftUI

struct ContentView: View {
    static let minimumSize = CGSize(width: 900, height: 640)

    @Bindable var model: SessionModel
    let setup: SetupCoordinator
    @State private var showsInspector = true
    @State private var availableWidth: CGFloat = 1360
    @State private var showsVoices = false
    @State private var showsCompactInspector = false
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
            VStack(spacing: 0) {
                if model.document != nil {
                    Picker("Sidebar", selection: $showsVoices) {
                        Text("Sections").tag(false)
                        Text("Voices").tag(true)
                    }
                    .pickerStyle(.segmented)
                    .labelsHidden()
                    .padding(10)
                }
                if let document = model.document, !showsVoices {
                    TimelineView(.animation(minimumInterval: 0.1, paused: !model.state.isPlaying)) { _ in
                        SectionsView(model: model, document: document,
                                     playingID: model.state.isPlaying ? model.playingSegment(at: model.position) : nil)
                    }
                } else {
                    VoiceLibraryView(model: model, newVoice: $newVoiceClip)
                }
            }
            .navigationSplitViewColumnWidth(min: 200, ideal: 260, max: 340)
        } detail: {
            HStack(spacing: 0) {
                VStack(spacing: 0) {
                    ScriptEditorView(model: model)
                        .frame(minHeight: 0, maxHeight: .infinity)
                    VStack(spacing: 0) {
                        TransportBar(model: model)
                        if !model.visibleTakes.isEmpty { TakesStrip(model: model) }
                    }
                    .fixedSize(horizontal: false, vertical: true)
                    .layoutPriority(1)
                }
                .frame(minWidth: 480)
                if showsInspector && availableWidth >= 1100 {
                    Divider()
                    InspectorView(model: model).frame(width: 280)
                }
            }
        }
        .navigationSplitViewStyle(.balanced)
        .onGeometryChange(for: CGFloat.self) { $0.size.width } action: { availableWidth = $0 }
        .onChange(of: model.document != nil) { showsVoices = false }
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
                Button("Inspector", systemImage: "sidebar.trailing") {
                    if availableWidth < 1100 { showsCompactInspector = true }
                    else { showsInspector.toggle() }
                }
                    .help("Show or hide voice and tuning controls")
            }
        }
        .sheet(item: $newVoiceClip) { request in
            AddVoiceSheet(model: model, clip: request.clip)
        }
        .sheet(isPresented: $showsCompactInspector) {
            InspectorSheet(model: model) { showsCompactInspector = false }
        }
        .modifier(ProjectRecovery(model: model))
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

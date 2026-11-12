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
                editorWorkspace
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
                Button(model.document == nil ? "Clear Script" : "Clear Section", systemImage: "eraser") {
                    model.clearScriptInput()
                }
                .disabled(model.editorText.isEmpty || model.state.isGenerating)
                .help("Clear the editor text. Existing takes remain available.")
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

    @ViewBuilder
    private var editorWorkspace: some View {
        if model.visibleTakes.isEmpty {
            editorAndTransport
        } else {
            EditorOverTakes(model: model) { editorAndTransport }
        }
    }

    private var editorAndTransport: some View {
        VStack(spacing: 0) {
            ScriptEditorView(model: model)
                .frame(minHeight: 0, maxHeight: .infinity)
            TransportBar(model: model)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}

/// An editor over the takes. The editor keeps a floor of height, and the
/// takes are told how much the floor leaves them, so the grid shows whole
/// rows within it.
struct EditorOverTakes<Editor: View>: View {
    let model: SessionModel
    @ViewBuilder var editor: Editor
    @State private var height: CGFloat?
    @State private var editorHeight: CGFloat?
    @State private var textHeight: CGFloat?

    var body: some View {
        VStack(spacing: 0) {
            editor.frame(minHeight: EditorFloor.height)
                .onGeometryChange(for: CGFloat.self, of: \.size.height) { editorHeight = $0 }
                .onPreferenceChange(ScriptTextHeight.self) { textHeight = $0 }
            TakesStrip(model: model, room: height.map { $0 - floor }).layoutPriority(1)
        }
        .onGeometryChange(for: CGFloat.self, of: \.size.height) { height = $0 }
    }

    /// The banners, footer and transport around the script text keep their
    /// heights whatever the takes leave, so the floor is theirs plus a few
    /// lines of text.
    private var floor: CGFloat {
        guard let editorHeight, let textHeight else { return EditorFloor.height }
        return max(EditorFloor.height, editorHeight - textHeight + EditorFloor.text)
    }
}

enum EditorFloor {
    /// The least height the editor and transport keep above the takes.
    static let height: CGFloat = 220
    /// The least height the script text keeps when the editor reports it: one line.
    static let text: CGFloat = 40
}

/// The height of the script text inside the editor.
struct ScriptTextHeight: PreferenceKey {
    static let defaultValue: CGFloat? = nil
    static func reduce(value: inout CGFloat?, nextValue: () -> CGFloat?) { value = nextValue() ?? value }
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

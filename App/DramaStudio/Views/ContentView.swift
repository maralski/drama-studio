import AppKit
import SwiftUI

struct ContentView: View {
    @Bindable var model: SessionModel
    @State private var showsInspector = true
    @State private var newVoiceClip: NewVoiceRequest?

    var body: some View {
        NavigationSplitView {
            VoiceLibraryView(model: model, newVoice: $newVoiceClip)
                .navigationSplitViewColumnWidth(min: 210, ideal: 240, max: 320)
        } detail: {
            ScriptEditorView(model: model)
                .safeAreaInset(edge: .bottom, spacing: 0) { TransportBar(model: model) }
                .inspector(isPresented: $showsInspector) {
                    InspectorView(model: model)
                        .inspectorColumnWidth(min: 270, ideal: 300, max: 380)
                }
        }
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                Button("Import Script", systemImage: "square.and.arrow.down") { FilePanels.importScript(into: model) }
                    .help("Import a script (⌘O)")
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

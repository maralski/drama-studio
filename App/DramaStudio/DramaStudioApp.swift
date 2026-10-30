// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import DramaEngine
import SwiftUI

@main
struct DramaStudioApp: App {
    static let mainWindowID = "main"

    @Environment(\.openWindow) private var openWindow

    private func updateWindow() { openWindow(id: "updates") }

    @State private var model: SessionModel
    @State private var server: APIServerController
    @State private var presence: AppPresence
    @State private var updater = AppUpdater()
    @State private var setup: SetupCoordinator

    init() {
        if CommandLine.arguments.contains("--check-resources") { exit(Self.checkResources()) }
        let model = SessionModel(synthesizer: ModelSetup.makeSynthesizer(), library: VoiceLibrary())
        let server = APIServerController(session: model)
        let presence = AppPresence()
        let setup = SetupCoordinator(services: .live(session: model, server: server))
        model.generationRefusal = { [setup] in setup.generationRefusal }
        _model = State(initialValue: model)
        _server = State(initialValue: server)
        _presence = State(initialValue: presence)
        _setup = State(initialValue: setup)
        server.startIfEnabled()
        presence.start()
    }

    var body: some Scene {
        Window("Drama Studio", id: Self.mainWindowID) {
            ContentView(model: model, setup: setup)
                .environment(server)
                .environment(updater)
                .onChange(of: updater.readyVersion) { if updater.readyVersion != nil { updateWindow() } }
                .onReceive(NotificationCenter.default.publisher(for: NSApplication.willTerminateNotification)) { _ in updater.discardOnQuit() }
                .task { await updater.start() }
                .defaultAppStorage(AppHome.defaults)
                .onOpenURL { model.open($0) }
        }
        .defaultSize(width: 1360, height: 820)
        .windowResizability(.contentMinSize)
        .commands {
            StudioCommands(model: model)
            CommandGroup(after: .appInfo) {
                Button("Check for Updates…") {
                    NSApp.activate()
                    updateWindow()
                    Task { await updater.check() }
                }
                .disabled(updater.busy)
            }
        }

        MenuBarExtra {
            MenuBarContent(model: model, server: server, presence: presence)
                .defaultAppStorage(AppHome.defaults)
        } label: {
            Image(nsImage: MenuBarIcon.image)
        }

        Window("Drama Studio Help", id: HelpWindowView.id) {
            HelpWindowView()
        }
        .defaultSize(width: 760, height: 500)

        Window("Software Update", id: "updates") {
            UpdateSettingsView()
                .modifier(UpdateConfirmation())
                .environment(updater)
                .frame(width: 520, height: 260)
        }

        Settings {
            SettingsView(server: server, setup: setup)
                .environment(updater)
                .defaultAppStorage(AppHome.defaults)
        }
    }

    /// Proves the engine's resource bundle resolves from inside the app bundle
    /// by compiling the Metal kernels it ships.
    private static func checkResources() -> Int32 {
        do {
            let context = try MetalContext()
            print("DramaEngine resources OK; kernels compiled for \(context.device.name)")
            return 0
        } catch {
            print("DramaEngine resources failed: \(error)")
            return 1
        }
    }
}

struct StudioCommands: Commands {
    @Bindable var model: SessionModel
    @Environment(\.openWindow) private var openWindow

    var body: some Commands {
        CommandGroup(replacing: .appInfo) {
            Button("About Drama Studio") { AboutPanel.show() }
        }
        CommandGroup(replacing: .newItem) {
            Button("Import Script or Document…") { FilePanels.importScript(into: model) }
                .keyboardShortcut("o")
            Button("Open Project…") { FilePanels.openProject(into: model) }
                .keyboardShortcut("o", modifiers: [.command, .shift])
            Menu("Open Recent Project") {
                ForEach(model.recentProjects, id: \.self) { url in
                    Button(url.deletingPathExtension().lastPathComponent) { model.openProject(from: url) }
                }
            }
            .disabled(model.recentProjects.isEmpty)
        }
        CommandGroup(replacing: .saveItem) {
            Button("Save Project…") { FilePanels.saveProject(model) }
                .keyboardShortcut("s")
                .disabled(model.state.isGenerating)
        }
        CommandGroup(replacing: .importExport) {
            Button("Export Audio…") { FilePanels.export(model) }
                .keyboardShortcut("e")
                .disabled(!model.canExport)
            Menu("Export") { ExportMenuItems(model: model) }
        }
        CommandMenu("Speech") {
            Button("Generate") { model.generate() }
                .keyboardShortcut(.return)
                .disabled(!model.canGenerate)
            Button("Narrate This Section") { model.narrateSelectedSection() }
                .keyboardShortcut(.return, modifiers: [.command, .shift])
                .disabled(!model.canNarrateSection)
            Button("Stop Generating") { model.stopGeneration() }
                .keyboardShortcut(".")
                .disabled(!model.state.isGenerating)
            Divider()
            Button(model.state.isPlaying ? "Pause" : "Play") { model.togglePlayback() }
                .disabled(!model.canPlay)
        }
        CommandGroup(replacing: .help) {
            Button("Drama Studio Help") { openWindow(id: HelpWindowView.id) }
                .keyboardShortcut("?", modifiers: .command)
        }
        CommandMenu("Script") {
            Menu("Insert Cue") { CueMenuItems(model: model) }
            Menu("Examples") { ExampleMenuItems(model: model) }
            Toggle("Screenplay Mode", isOn: $model.screenplayMode)
                .keyboardShortcut("p", modifiers: [.command, .shift])
                .disabled(model.document != nil)
            Button("Flatten to Script") { model.flattenToScript() }
                .disabled(model.document == nil || model.state.isGenerating)
        }
    }
}

struct CueMenuItems: View {
    let model: SessionModel

    var body: some View {
        ForEach(PerformanceCue.Kind.allCases, id: \.self) { kind in
            Section(kind.title) {
                ForEach(PerformanceCue.cues(of: kind)) { cue in
                    Button("\(cue.title)  \(cue.opening)") { model.insert(cue) }
                }
            }
        }
        Section { Text(PerformanceCue.reliabilityNote) }
    }
}

struct ExampleMenuItems: View {
    let model: SessionModel

    var body: some View {
        Section("Lines") {
            ForEach(Examples.lines) { line in
                Button(line.title) { model.insert(line) }
            }
        }
        Section("Cast") {
            ForEach(Examples.cast) { preset in
                Button(preset.design.name) { model.cast(preset) }
            }
        }
    }
}

enum AboutPanel {
    @MainActor
    static func show() {
        let centered = NSMutableParagraphStyle()
        centered.alignment = .center
        let font = NSFont.systemFont(ofSize: NSFont.smallSystemFontSize)
        let credits = NSMutableAttributedString(
            string: "By Peter Marelas\n",
            attributes: [.font: NSFont.boldSystemFont(ofSize: NSFont.smallSystemFontSize), .paragraphStyle: centered, .foregroundColor: NSColor.labelColor]
        )
        credits.append(NSAttributedString(string: "Built with Fish Audio\n", attributes: [.font: font, .paragraphStyle: centered, .foregroundColor: NSColor.labelColor]))
        credits.append(NSAttributedString(string: "Runs Fish Audio Drama 3 locally under the ", attributes: [.font: font, .paragraphStyle: centered, .foregroundColor: NSColor.secondaryLabelColor]))
        credits.append(NSAttributedString(string: "Fish Audio Community License", attributes: [.font: font, .paragraphStyle: centered, .link: ModelSetup.licenseURL]))
        NSApp.orderFrontStandardAboutPanel(options: [.credits: credits])
        NSApp.activate()
    }
}

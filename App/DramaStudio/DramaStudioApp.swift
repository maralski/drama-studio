// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import DramaEngine
import SwiftUI

@main
struct DramaStudioApp: App {
    static let mainWindowID = "main"

    @Environment(\.openSettings) private var openSettings
    @State private var settingsTab: SettingsTab = .general

    private func showUpdates() {
        settingsTab = .updates
        openSettings()
    }

    @NSApplicationDelegateAdaptor(StudioAppDelegate.self) private var appDelegate
    @State private var model: SessionModel
    @State private var server: APIServerController
    @State private var presence: AppPresence
    @State private var updater: AppUpdater
    @State private var setup: SetupCoordinator
    @State private var ming: MingInstall

    init() {
        if CommandLine.arguments.contains("--check-resources") { exit(Self.checkResources()) }
        if CommandLine.arguments.dropFirst().first == "--check-words" {
            exit(WordCheckDiagnostic.run(Array(CommandLine.arguments.dropFirst(2))))
        }
        let model = { let library = VoiceLibrary(directory: AppHome.voices) { TermsAgreement(defaults: AppHome.defaults).isAccepted }; return SessionModel(synthesizer: ModelSetup.makeEngines(library: library), library: library) }()
        let server = APIServerController(session: model)
        let presence = AppPresence()
        let updater = AppUpdater()
        let setup = SetupCoordinator(services: .live(session: model, server: server))
        model.generationRefusal = { [setup] in setup.generationRefusal }
        _model = State(initialValue: model)
        _server = State(initialValue: server)
        _presence = State(initialValue: presence)
        _updater = State(initialValue: updater)
        _setup = State(initialValue: setup)
        _ming = State(initialValue: MingInstall(session: model))
        model.recordUsage = { UsageMonitoring.shared.record($0) }
        UsageMonitoring.shared.record(.launch)
        model.confirmReplacement = { [weak model] message in
            guard let model else { return false }
            return FilePanels.confirmReplacement(model, message: message)
        }
        appDelegate.model = model
        appDelegate.updater = updater
        server.startIfEnabled()
        presence.start()
    }

    var body: some Scene {
        Window("Drama Studio", id: Self.mainWindowID) {
            ContentView(model: model, setup: setup)
                .environment(server)
                .environment(updater)
                .onChange(of: updater.readyVersion) { if updater.readyVersion != nil { showUpdates() } }
                .onReceive(NotificationCenter.default.publisher(for: NSApplication.willTerminateNotification)) { _ in updater.discardOnQuit() }
                .task { await updater.start() }
                .defaultAppStorage(AppHome.defaults)
                .onOpenURL { model.open($0) }
        }
        .defaultSize(width: 1360, height: 820)
        .windowResizability(.contentSize)
        .commands {
            StudioCommands(model: model)
            CommandGroup(after: .appInfo) {
                Button("Check for Updates…") {
                    NSApp.activate()
                    showUpdates()
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
        .windowResizability(.contentMinSize)

        Settings {
            SettingsView(server: server, setup: setup, ming: ming, selection: $settingsTab)
                .modifier(UpdateConfirmation())
                .environment(updater)
                .defaultAppStorage(AppHome.defaults)
        }
    }

    /// Proves the engine's resource bundle resolves from inside the app bundle
    /// by compiling the Metal kernels it ships.
    private static func checkResources() -> Int32 {
        do {
            try WordCoverage.prepare()
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
            Button("Save Project") { FilePanels.saveProject(model) }
                .keyboardShortcut("s")
                .disabled(model.state.isGenerating)
            Button("Save Project As…") { FilePanels.saveProject(model, saveAs: true) }
                .keyboardShortcut("s", modifiers: [.command, .shift])
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
            Button("Generate Changed Sections") { model.generateChangedSections() }
                .disabled(!model.canGenerateChangedSections)
            Button("Narrate This Section") { model.narrateSelectedSection() }
                .keyboardShortcut(.return, modifiers: [.command, .shift])
                .disabled(!model.canNarrateSection)
            Button("Stop Generating") { model.stopGeneration() }
                .keyboardShortcut(".")
                .disabled(!model.state.isGenerating)
            Divider()
            Picker("Engine", selection: $model.engine) { EngineChoices(model: model) }
                .disabled(model.state.isGenerating)
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
            Button(model.document == nil ? "Clear Script" : "Clear Section") { model.clearScriptInput() }
                .disabled(model.editorText.isEmpty || model.state.isGenerating)
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
        for engine in SpeechEngineID.allCases {
            let install = engine.install
            credits.append(NSAttributedString(string: "Runs \(install.publisher) \(engine.capabilities.name) locally under the ", attributes: [.font: font, .paragraphStyle: centered, .foregroundColor: NSColor.secondaryLabelColor]))
            credits.append(NSAttributedString(string: install.license.name + "\n", attributes: [.font: font, .paragraphStyle: centered, .link: install.license.url]))
        }
        NSApp.orderFrontStandardAboutPanel(options: [.credits: credits])
        NSApp.activate()
    }
}

import DramaEngine
import SwiftUI

@main
struct DramaStudioApp: App {
    static let mainWindowID = "main"

    @State private var model: SessionModel
    @State private var server: APIServerController
    @State private var presence: AppPresence

    init() {
        if CommandLine.arguments.contains("--check-resources") { exit(Self.checkResources()) }
        let model = SessionModel(synthesizer: ModelSetup.makeSynthesizer(), library: VoiceLibrary())
        let server = APIServerController(session: model)
        let presence = AppPresence()
        _model = State(initialValue: model)
        _server = State(initialValue: server)
        _presence = State(initialValue: presence)
        server.startIfEnabled()
        presence.start()
    }

    var body: some Scene {
        Window("Drama Studio", id: Self.mainWindowID) {
            ContentView(model: model)
                .environment(server)
                .onOpenURL { model.open($0) }
        }
        .defaultSize(width: 1240, height: 780)
        .commands { StudioCommands(model: model) }

        MenuBarExtra("Drama Studio", systemImage: "waveform.badge.mic") {
            MenuBarContent(model: model, server: server, presence: presence)
        }

        Window("Drama Studio Help", id: HelpWindowView.id) {
            HelpWindowView()
        }
        .defaultSize(width: 760, height: 500)

        Settings {
            SettingsView(server: server)
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
            Button("Import Script…") { FilePanels.importScript(into: model) }
                .keyboardShortcut("o")
        }
        CommandGroup(replacing: .importExport) {
            Button("Export Audio…") { FilePanels.export(model) }
                .keyboardShortcut("e")
                .disabled(!model.canExport)
        }
        CommandMenu("Speech") {
            Button("Generate") { model.generate() }
                .keyboardShortcut(.return)
                .disabled(!model.canGenerate)
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

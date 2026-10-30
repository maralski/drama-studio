// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import AppKit
import UniformTypeIdentifiers

@MainActor
enum FilePanels {
    private static let exportFolderKey = "lastExportFolder"
    private static let exportFormatKey = "lastExportFormat"

    /// Declared in the app's Info.plist; a development build gets a dynamic type with the same extension.
    static let projectType = UTType(filenameExtension: StudioProject.fileExtension, conformingTo: .package) ?? .package

    static var lastExportFormat: ExportFormat {
        AppHome.defaults.string(forKey: exportFormatKey).flatMap(ExportFormat.init) ?? .wav16
    }

    static func importScript(into model: SessionModel) {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = ImportFormat.allCases.map(\.contentType)
        panel.message = "Choose a script or a document to narrate by section."
        guard panel.runModal() == .OK, let url = panel.url else { return }
        model.importScript(from: url)
    }

    static func export(_ model: SessionModel, as format: ExportFormat? = nil) {
        guard model.canExport, confirmExport(model) else { return }
        let format = format ?? lastExportFormat
        guard let url = save(name: "\(model.title).\(format.fileExtension)", type: format.contentType, message: "\(model.exportSummary) · \(format.title), 44.1 kHz mono. Playback speed does not change the export.") else { return }
        AppHome.defaults.set(format.rawValue, forKey: exportFormatKey)
        model.export(to: url, as: format)
    }

    static func exportSections(_ model: SessionModel) {
        guard model.canExportSections, confirmExport(model) else { return }
        let format = lastExportFormat
        guard let folder = save(name: "\(model.title) Sections", type: .folder, message: "A folder with one \(format.title) file per section and a manifest.csv") else { return }
        model.exportSections(to: folder, as: format)
    }

    static func exportCaptions(_ model: SessionModel, as format: Captions.Format) {
        guard model.canExportCaptions, confirmExport(model) else { return }
        let type = format == .srt ? ImportFormat.subRipType : ImportFormat.webVTT.contentType
        guard let url = save(name: "\(model.title).\(format.fileExtension)", type: type, message: "One caption per spoken sentence") else { return }
        model.exportCaptions(to: url, as: format)
    }

    static func exportSlidePackage(_ model: SessionModel) {
        guard model.canExportSlidePackage, confirmExport(model) else { return }
        guard let folder = save(name: "\(model.title) Slide Narration", type: .folder, message: "A folder with one WAV file per slide, named by slide number, and a README on adding them in PowerPoint") else { return }
        model.exportSlidePackage(to: folder, as: .wav16)
    }

    @discardableResult
    static func saveProject(_ model: SessionModel, saveAs: Bool = false) -> Bool {
        if !saveAs, let url = model.projectURL { return model.saveProject(to: url) }
        guard let url = save(name: "\(model.projectName).\(StudioProject.fileExtension)", type: projectType,
                             message: "Save the script, all takes, voices, and export selections") else { return false }
        return model.saveProject(to: url)
    }

    static func confirmReplacement(_ model: SessionModel, message: String) -> Bool {
        if model.state.isGenerating {
            model.notice = "Stop generation before replacing or closing this project."
            let alert = NSAlert()
            alert.messageText = "Generation is still running"
            alert.informativeText = "Stop generation, then save the project before continuing."
            alert.addButton(withTitle: "Return to Studio")
            alert.runModal()
            return false
        }
        let flattening = message.hasPrefix("Flattening")
        guard model.hasUnsavedChanges || flattening else { return true }
        let alert = NSAlert()
        alert.alertStyle = .warning
        alert.messageText = message
        alert.informativeText = flattening
            ? "Flattening removes the section structure and recorded takes from this session. Save a copy to keep them."
            : "Save changes to “\(model.projectName)” before continuing? Your script and recorded takes have unsaved changes."
        alert.addButton(withTitle: flattening ? "Save Copy and Flatten" : "Save")
        alert.addButton(withTitle: "Cancel")
        alert.addButton(withTitle: flattening ? "Flatten Without Saving" : "Discard Changes")
        switch alert.runModal() {
        case .alertFirstButtonReturn: return saveProject(model, saveAs: flattening)
        case .alertThirdButtonReturn: return true
        default: return false
        }
    }

    static func offerRecovery(_ model: SessionModel) -> Bool {
        let alert = NSAlert()
        alert.messageText = "Recover unsaved work?"
        alert.informativeText = "Drama Studio found a recovery draft from an earlier session. It includes the script, recorded takes, and export choices."
        alert.addButton(withTitle: "Recover")
        alert.addButton(withTitle: "Discard Draft")
        if alert.runModal() == .alertFirstButtonReturn {
            return model.openProject(from: RecoveryStore.shared.url, recovering: true)
        }
        return true
    }

    private static func confirmExport(_ model: SessionModel) -> Bool {
        guard let warning = model.exportWarning else { return true }
        let alert = NSAlert()
        alert.alertStyle = .warning
        alert.messageText = "Export existing audio?"
        alert.informativeText = warning
        alert.addButton(withTitle: "Cancel")
        alert.addButton(withTitle: "Export Existing Audio")
        return alert.runModal() == .alertSecondButtonReturn
    }

    static func openProject(into model: SessionModel) {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.allowedContentTypes = [projectType, .folder]
        panel.message = "Choose a Drama Studio project (.\(StudioProject.fileExtension))."
        guard panel.runModal() == .OK, let url = panel.url else { return }
        model.openProject(from: url)
    }

    private static func save(name: String, type: UTType, message: String) -> URL? {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [type]
        panel.nameFieldStringValue = name
        panel.directoryURL = AppHome.defaults.url(forKey: exportFolderKey)
        panel.message = message
        guard panel.runModal() == .OK, let url = panel.url else { return nil }
        AppHome.defaults.set(url.deletingLastPathComponent(), forKey: exportFolderKey)
        return url
    }
}

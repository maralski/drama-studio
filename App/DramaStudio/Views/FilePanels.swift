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
        UserDefaults.standard.string(forKey: exportFormatKey).flatMap(ExportFormat.init) ?? .wav16
    }

    static func importScript(into model: SessionModel) {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = ImportFormat.allCases.map(\.contentType)
        panel.message = "Choose a script (text, Markdown, RTF, HTML, Word, SRT, or WebVTT) or a document to narrate by section (PowerPoint, PDF, or EPUB)."
        guard panel.runModal() == .OK, let url = panel.url else { return }
        model.importScript(from: url)
    }

    static func export(_ model: SessionModel, as format: ExportFormat? = nil) {
        guard model.canExport else { return }
        let format = format ?? lastExportFormat
        guard let url = save(name: "\(model.title).\(format.fileExtension)", type: format.contentType, message: "Export as \(format.title), 44.1 kHz mono") else { return }
        UserDefaults.standard.set(format.rawValue, forKey: exportFormatKey)
        model.export(to: url, as: format)
    }

    static func exportSections(_ model: SessionModel) {
        guard model.canExportSections else { return }
        let format = lastExportFormat
        guard let folder = save(name: "\(model.title) Sections", type: .folder, message: "A folder with one \(format.title) file per section and a manifest.csv") else { return }
        model.exportSections(to: folder, as: format)
    }

    static func exportCaptions(_ model: SessionModel, as format: Captions.Format) {
        guard model.canExportCaptions else { return }
        let type = format == .srt ? ImportFormat.subRipType : ImportFormat.webVTT.contentType
        guard let url = save(name: "\(model.title).\(format.fileExtension)", type: type, message: "One caption per spoken sentence") else { return }
        model.exportCaptions(to: url, as: format)
    }

    static func exportSlidePackage(_ model: SessionModel) {
        guard model.canExportSlidePackage else { return }
        guard let folder = save(name: "\(model.title) Slide Narration", type: .folder, message: "A folder with one WAV file per slide, named by slide number, and a README on adding them in PowerPoint") else { return }
        model.exportSlidePackage(to: folder, as: .wav16)
    }

    static func saveProject(_ model: SessionModel) {
        guard let url = save(name: "\(model.title).\(StudioProject.fileExtension)", type: projectType, message: "Save the script or document, the cast, the settings, and the chosen takes") else { return }
        model.saveProject(to: url)
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
        panel.directoryURL = UserDefaults.standard.url(forKey: exportFolderKey)
        panel.message = message
        guard panel.runModal() == .OK, let url = panel.url else { return nil }
        UserDefaults.standard.set(url.deletingLastPathComponent(), forKey: exportFolderKey)
        return url
    }
}

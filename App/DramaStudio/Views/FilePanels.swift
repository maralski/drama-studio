import AppKit
import UniformTypeIdentifiers

@MainActor
enum FilePanels {
    private static let exportFolderKey = "lastExportFolder"
    private static let exportFormatKey = "lastExportFormat"

    static var lastExportFormat: ExportFormat {
        UserDefaults.standard.string(forKey: exportFormatKey).flatMap(ExportFormat.init) ?? .wav16
    }

    static func importScript(into model: SessionModel) {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = ImportFormat.allCases.map(\.contentType)
        panel.message = "Choose a script: text, Markdown, RTF, HTML, Word, SRT, or WebVTT."
        guard panel.runModal() == .OK, let url = panel.url else { return }
        model.importScript(from: url)
    }

    static func export(_ model: SessionModel, as format: ExportFormat? = nil) {
        guard model.canExport else { return }
        let format = format ?? lastExportFormat
        let panel = NSSavePanel()
        panel.allowedContentTypes = [format.contentType]
        panel.nameFieldStringValue = "\(model.title).\(format.fileExtension)"
        panel.directoryURL = UserDefaults.standard.url(forKey: exportFolderKey)
        panel.message = "Export as \(format.title), 44.1 kHz mono"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        UserDefaults.standard.set(url.deletingLastPathComponent(), forKey: exportFolderKey)
        UserDefaults.standard.set(format.rawValue, forKey: exportFormatKey)
        model.export(to: url, as: format)
    }
}

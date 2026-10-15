import DramaEngine
import Foundation

enum ModelSetup {
    static let downloadCommand = "hf download fishaudio/drama-3"
    static let customDirectoryKey = "modelDirectory"
    static let licenseURL = URL(string: "https://huggingface.co/fishaudio/drama-3/blob/main/LICENSE")!

    /// The Metal engine is not wired in yet; the app runs the mock so the whole
    /// flow works end to end.
    static let isPreviewEngine = true

    static func makeSynthesizer() -> any SpeechSynthesizer {
        MockSynthesizer()
    }

    enum Status: Equatable {
        case ready(URL)
        case incomplete(URL)
        case missing

        var problem: String? {
            switch self {
            case .ready: nil
            case .incomplete(let folder): "The Drama 3 folder at \(folder.path) is missing files. Run \u{201C}\(ModelSetup.downloadCommand)\u{201D} in Terminal to finish the download."
            case .missing: "Drama 3 weights not found. Run \u{201C}\(ModelSetup.downloadCommand)\u{201D} in Terminal, or choose a folder in Settings."
            }
        }
    }

    static func location(customPath: String) -> ModelLocation? {
        customPath.isEmpty ? .huggingFaceCache() : ModelLocation(directory: URL(filePath: customPath, directoryHint: .isDirectory))
    }

    static func status(customPath: String) -> Status {
        guard let location = location(customPath: customPath) else { return .missing }
        return location.isComplete ? .ready(location.directory) : .incomplete(location.directory)
    }
}

import DramaEngine
import Foundation

enum ModelSetup {
    static let downloadCommand = "hf download fishaudio/drama-3"
    static let customDirectoryKey = "modelDirectory"
    static let licenseURL = URL(string: "https://huggingface.co/fishaudio/drama-3/blob/main/LICENSE")!

    /// Set DRAMA_PREVIEW_ENGINE=1 to run the app on the mock instead of the Metal engine.
    static let isPreviewEngine = ProcessInfo.processInfo.environment["DRAMA_PREVIEW_ENGINE"] == "1"

    static let precisionKey = "modelPrecision"
    /// A hidden defaults key: true offers the experimental precision modes too.
    static let showExperimentalPrecisionKey = "showExperimentalPrecision"

    static var storedPrecision: ModelPrecisionMode {
        UserDefaults.standard.string(forKey: precisionKey).flatMap(ModelPrecisionMode.init) ?? .full
    }

    /// The modes the inspector offers: the shipped ones, plus the experimental
    /// ones behind the hidden key, plus whatever is chosen already.
    static func offeredPrecisions(current: ModelPrecisionMode) -> [ModelPrecisionMode] {
        let experimental = UserDefaults.standard.bool(forKey: showExperimentalPrecisionKey)
        return ModelPrecisionMode.allCases.filter { experimental || !$0.isExperimental || $0 == current }
    }

    /// The mock in every mode under the preview engine or without weights.
    static func makeSynthesizer(mode: ModelPrecisionMode = storedPrecision) -> any SpeechSynthesizer {
        if isPreviewEngine { return MockSynthesizer() }
        let customPath = UserDefaults.standard.string(forKey: customDirectoryKey) ?? ""
        guard let location = location(customPath: customPath) else { return MockSynthesizer() }
        return DramaSynthesizer(location: location, codecDirectory: DramaCodec.defaultDirectory, mode: mode)
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

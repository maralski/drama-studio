// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation

package struct RecognitionModelAsset: Sendable {
    package let id: String
    package let fileName: String
    package let sha256: String
    package let bytes: Int
    package let englishOnly: Bool
}

package extension WordRecognitionAssets {
    static func model(for language: String) -> RecognitionModelAsset {
        // The generated manifest contains exactly one model for each route.
        language == "en" ? englishModel : multilingualModel
    }
}

/// The worker receives only delivered PCM and its language. Expected words,
/// voice descriptions, cues, prompts and history have no wire representation.
package struct RecognitionRequest: Codable, Sendable {
    package let id: UUID
    package let language: String
    package let pcm16: Data

    package init(id: UUID, language: String, pcm16: Data) {
        self.id = id; self.language = language; self.pcm16 = pcm16
    }
}

package struct RecognitionReady: Codable, Sendable {
    package let protocolVersion: Int
    package let configurationID: String
    package let modelID: String
    package let modelSHA256: String
    package let languages: [String]

    package init(model: RecognitionModelAsset, languages: [String]) {
        protocolVersion = WordRecognitionAssets.protocolVersion
        configurationID = WordRecognitionAssets.configurationID
        modelID = model.id; modelSHA256 = model.sha256; self.languages = languages
    }
}

package enum RecognitionReply: Codable, Sendable {
    case ready(RecognitionReady)
    case transcript(id: UUID, text: String)
    case failure(id: UUID?, message: String)
}

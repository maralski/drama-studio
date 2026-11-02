// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import DramaEngine
import Foundation

struct ProjectRevision: Equatable {
    var source: ScriptSource
    var voices: [Int: VoiceChoice]
    var recordedVoices: [VoiceChoice: RecordedVoice]
    var screenplay: Bool
    var expressiveness: Double
    var takes: [UUID]
    var sections: [UUID: [UUID]]
    var exportTake: UUID?
    var sectionExports: [UUID: UUID]
}

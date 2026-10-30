// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation
import SwiftUI

struct ProjectSnapshot: Sendable {
    var project: StudioProject
    var audio: [Float]
    var archivedAudio: [UUID: [Float]]

    func save(to url: URL) throws {
        try project.save(to: url, audio: audio, archivedAudio: archivedAudio)
    }
}

actor RecoveryStore {
    static let shared = RecoveryStore(url: AppHome.support.appending(path: "Recovery.dramaproj"))
    nonisolated let url: URL

    init(url: URL) { self.url = url }

    func exists() -> Bool { FileManager.default.fileExists(atPath: url.path) }

    func update(_ snapshot: ProjectSnapshot?) throws {
        guard !Task.isCancelled else { return }
        if let snapshot {
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try snapshot.save(to: url)
        } else if exists() {
            try FileManager.default.removeItem(at: url)
        }
    }
}

private struct RecoveryTrigger: Equatable {
    var revision: ProjectRevision
    var generating: Bool
    var checked: Bool
    var dirty: Bool
}

struct ProjectRecovery: ViewModifier {
    let model: SessionModel
    @State private var checked = false

    func body(content: Content) -> some View {
        content
            .task {
                checked = await RecoveryStore.shared.exists() ? FilePanels.offerRecovery(model) : true
            }
            .task(id: RecoveryTrigger(revision: model.recoveryRevision, generating: model.state.isGenerating,
                                      checked: checked, dirty: model.hasUnsavedChanges)) {
                guard checked, !model.state.isGenerating else { return }
                do {
                    if model.hasUnsavedChanges { try await Task.sleep(for: .seconds(2)) }
                    try Task.checkCancellation()
                    let snapshot = model.hasUnsavedChanges ? model.projectSnapshot : nil
                    try await RecoveryStore.shared.update(snapshot)
                } catch is CancellationError {
                    return
                } catch {
                    model.notice = "Couldn't save the recovery draft: \(error.localizedDescription). Save the project manually."
                }
            }
    }
}

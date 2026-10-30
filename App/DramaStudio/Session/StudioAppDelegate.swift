// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import AppKit
import SwiftUI

@MainActor
final class StudioAppDelegate: NSObject, NSApplicationDelegate {
    weak var model: SessionModel?
    weak var updater: AppUpdater?

    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        guard let model else { return .terminateNow }
        guard FilePanels.confirmReplacement(model, message: "Quit Drama Studio?") else {
            updater?.cancelTermination()
            return .terminateCancel
        }
        guard updater?.completeTermination() ?? true else { return .terminateCancel }
        Task {
            try? await RecoveryStore.shared.update(nil)
            sender.reply(toApplicationShouldTerminate: true)
        }
        return .terminateLater
    }
}

struct ProjectWindowState: NSViewRepresentable {
    let model: SessionModel

    func makeNSView(context: Context) -> NSView { NSView() }

    func updateNSView(_ view: NSView, context: Context) {
        let edited = model.hasUnsavedChanges
        let url = model.projectURL
        DispatchQueue.main.async {
            guard let window = view.window else { return }
            if window.isDocumentEdited != edited { window.isDocumentEdited = edited }
            if window.representedURL != url { window.representedURL = url }
        }
    }
}

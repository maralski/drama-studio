// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import SwiftUI

struct UpdateSettingsView: View {
    @Environment(AppUpdater.self) private var updater
    @Environment(\.openWindow) private var openWindow

    var body: some View {
        @Bindable var updater = updater
        Form {
            Toggle("Automatically check for and download updates", isOn: $updater.automatic)
            Text("Checks at launch. Turning this off stops future automatic checks. A download already in progress will finish. Updates are installed only when you choose Install and Restart.")
                .font(.caption).foregroundStyle(.secondary)
            Text(updater.status).textSelection(.enabled)
            if updater.busy { ProgressView().controlSize(.small) }
            HStack {
                Button("Check for Updates") { Task { await updater.check() } }
                    .disabled(updater.busy || updater.readyVersion != nil)
                if updater.readyVersion != nil {
                    Button("Review Update…") { openWindow(id: "updates"); updater.showsReady = true }
                        .disabled(updater.busy)
                }
            }
        }
        .formStyle(.grouped)
    }
}

struct UpdateConfirmation: ViewModifier {
    @Environment(AppUpdater.self) private var updater

    func body(content: Content) -> some View {
        @Bindable var updater = updater
        content.alert("Drama Studio \(updater.readyVersion ?? "") is ready", isPresented: $updater.showsReady) {
            Button("Later", role: .cancel) { }
            Button("Install and Restart", role: .destructive) { updater.installAndRestart() }
        } message: {
            Text("Save your project and finish any generation or API requests before continuing. Installing will quit Drama Studio and reopen the updated app.")
        }
    }
}

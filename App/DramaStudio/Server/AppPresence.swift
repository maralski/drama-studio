import AppKit
import SwiftUI

@MainActor @Observable
final class AppPresence {
    private(set) var hasWindow = true
    @ObservationIgnored private var observers: [NSObjectProtocol] = []

    func start() {
        let center = NotificationCenter.default
        observers = [NSWindow.didBecomeMainNotification, NSWindow.willCloseNotification].map { name in
            center.addObserver(forName: name, object: nil, queue: .main) { _ in
                // A closing window is still visible during willClose; look once it has gone.
                DispatchQueue.main.async { MainActor.assumeIsolated { self.windowsChanged() } }
            }
        }
    }

    func showMainWindow(_ openWindow: OpenWindowAction) {
        NSApp.setActivationPolicy(.regular)
        openWindow(id: DramaStudioApp.mainWindowID)
        NSApp.activate()
    }

    private func windowsChanged() {
        hasWindow = NSApp.windows.contains { $0.isVisible && $0.canBecomeMain && !($0 is NSPanel) }
        let policy: NSApplication.ActivationPolicy = hasWindow ? .regular : .accessory
        if NSApp.activationPolicy() != policy { NSApp.setActivationPolicy(policy) }
    }
}

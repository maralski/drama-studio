// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import AppKit
import XCTest
@testable import DramaStudio

@MainActor
private final class Flag { var raised = false }

@MainActor
final class FilePanelsTests: XCTestCase {
    /// Spins a nested run loop, as a modal alert does, until other main-actor
    /// work raises the flag or half a second passes.
    private static func modal(waitingFor flag: Flag) -> Bool {
        Task { @MainActor in flag.raised = true }
        let deadline = Date() + 0.5
        while !flag.raised && Date() < deadline { RunLoop.main.run(mode: .modalPanel, before: Date() + 0.01) }
        return flag.raised
    }

    /// Catches an alert shown inside a main-actor task: while it waits, no
    /// other main-actor task runs, and the API server's requests stall.
    func testMainActorWorkRunsWhileAPanelWaits() async {
        // AppKit makes the modal panel mode a common mode, so the main queue drains in it.
        CFRunLoopAddCommonMode(CFRunLoopGetMain(), CFRunLoopMode(RunLoop.Mode.modalPanel.rawValue as CFString))
        let inside = Self.modal(waitingFor: Flag())
        XCTAssertFalse(inside, "the control: inside a main-actor task, other main-actor work waits for the panel")
        let outside = await FilePanels.outsideTask { Self.modal(waitingFor: Flag()) }
        XCTAssertTrue(outside, "other main-actor work ran while the panel waited")
    }
}

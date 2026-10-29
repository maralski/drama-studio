// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import AppKit
import DramaEngine
import SwiftUI
import XCTest
@testable import DramaStudio

@MainActor
final class AppScriptTextViewTests: XCTestCase {
    /// AppKit registers `acceptableDragTypes` when the view joins a window, so
    /// the list itself is what decides whether a Finder drag stops at the editor.
    func testTheEditorAcceptsNoFileDragsSoDropsReachTheImporter() throws {
        let plain = NSTextView.scrollableTextView().documentView as! NSTextView
        let filenames = NSPasteboard.PasteboardType("NSFilenamesPboardType")
        XCTAssertTrue(plain.acceptableDragTypes.contains(filenames), "AppKit's default accepts file drags: \(plain.acceptableDragTypes)")
        XCTAssertTrue(plain.acceptableDragTypes.contains(.URL))

        let editor = try XCTUnwrap(ScriptTextView.makeScrollView().documentView as? ScriptTextView.TextView)
        let accepted = editor.acceptableDragTypes
        XCTAssertFalse(accepted.contains(where: ScriptTextView.TextView.isFileType), "\(accepted)")
        XCTAssertFalse(accepted.contains(filenames))
        XCTAssertFalse(accepted.contains(.URL))
        XCTAssertFalse(accepted.contains(.fileURL))
        XCTAssertTrue(accepted.contains(NSPasteboard.PasteboardType("NSStringPboardType")) || accepted.contains(.string), "text drags still work: \(accepted)")
    }

    func testSelectionConvertsBetweenStringIndicesAndUTF16Ranges() throws {
        let text = "Café 🎉 time."
        let range = NSRange(location: 5, length: 2)
        let selection = try XCTUnwrap(ScriptTextView.textSelection(range, in: text))
        XCTAssertEqual(ScriptTextView.nsRange(of: selection, in: text), range)
        guard case .selection(let indices) = selection.indices else { return XCTFail("a single range") }
        XCTAssertEqual(String(text[indices]), "🎉")
        XCTAssertNil(ScriptTextView.textSelection(NSRange(location: 6, length: 1), in: text), "inside the emoji's surrogate pair is no character boundary")
        XCTAssertNil(ScriptTextView.nsRange(of: nil, in: text))
    }

    func testTransportLabels() {
        XCTAssertEqual(SessionModel.playbackRates.map(TransportBar.speed), ["0.75x", "1x", "1.25x", "1.5x"])
        XCTAssertEqual(TransportBar.sectionSummary(sections: 3, audioSeconds: 12.34, stoppedEarly: false, generationSeconds: 4), "3 sections narrated \u{00B7} 12.3 s of audio \u{00B7} generated in 4.0 s")
        XCTAssertEqual(TransportBar.sectionSummary(sections: 1, audioSeconds: 2, stoppedEarly: true, generationSeconds: nil), "1 section narrated (stopped early) \u{00B7} 2.0 s of audio")
    }
}

import AppKit
import SwiftUI

/// The script editor. SwiftUI's TextEditor cannot mark a range without
/// moving the selection, so this wraps NSTextView and draws the spoken
/// sentence with rendering attributes: no edit, no undo entry.
struct ScriptTextView: NSViewRepresentable {
    @Binding var text: String
    @Binding var selection: TextSelection?
    /// UTF-16 range of the sentence being spoken.
    var highlight: NSRange?

    static let highlightColor = NSColor.systemYellow.withAlphaComponent(0.35)

    /// Declines file drags, so they pass through to the SwiftUI drop
    /// destination around the editor, which imports them. AppKit re-reads
    /// `acceptableDragTypes` whenever it re-registers the view, so an override
    /// holds where a one-time unregister would not.
    final class TextView: NSTextView {
        /// File URLs in every spelling AppKit lists them under, plus file promises.
        static let fileTypes: Set<NSPasteboard.PasteboardType> = [
            .fileURL, .URL, .fileContents, .filePromise,
            NSPasteboard.PasteboardType("NSFilenamesPboardType"),
            NSPasteboard.PasteboardType("Apple URL pasteboard type"),
            NSPasteboard.PasteboardType("WebURLsWithTitlesPboardType"),
            NSPasteboard.PasteboardType("com.apple.NSFilePromiseItemMetaData"),
            NSPasteboard.PasteboardType("com.apple.pasteboard.promised-file-content-type"),
        ]

        static func isFileType(_ type: NSPasteboard.PasteboardType) -> Bool {
            fileTypes.contains(type) || type.rawValue.hasPrefix("dyn.")
        }

        override var acceptableDragTypes: [NSPasteboard.PasteboardType] {
            super.acceptableDragTypes.filter { !Self.isFileType($0) }
        }
    }

    static func makeScrollView() -> NSScrollView {
        TextView.scrollableTextView()
    }

    func makeNSView(context: Context) -> NSScrollView {
        let scrollView = Self.makeScrollView()
        let textView = scrollView.documentView as! NSTextView
        textView.delegate = context.coordinator
        textView.isRichText = false
        textView.importsGraphics = false
        textView.allowsUndo = true
        textView.usesFindBar = true
        textView.isAutomaticQuoteSubstitutionEnabled = false
        textView.isAutomaticDashSubstitutionEnabled = false
        textView.font = .systemFont(ofSize: 15)
        textView.textContainerInset = NSSize(width: 14, height: 12)
        textView.drawsBackground = false
        let paragraph = NSMutableParagraphStyle()
        paragraph.lineSpacing = 3
        textView.defaultParagraphStyle = paragraph
        textView.typingAttributes = [.font: NSFont.systemFont(ofSize: 15), .paragraphStyle: paragraph, .foregroundColor: NSColor.textColor]
        textView.string = text
        scrollView.drawsBackground = false
        return scrollView
    }

    func updateNSView(_ scrollView: NSScrollView, context: Context) {
        let textView = scrollView.documentView as! NSTextView
        let coordinator = context.coordinator
        coordinator.parent = self
        coordinator.isUpdating = true
        defer { coordinator.isUpdating = false }
        if textView.string != text {
            textView.string = text
            coordinator.shown = nil
        }
        if let range = Self.nsRange(of: selection, in: text), textView.selectedRange() != range, range.upperBound <= (text as NSString).length {
            textView.setSelectedRange(range)
            textView.scrollRangeToVisible(range)
        }
        if coordinator.shown != highlight {
            coordinator.show(highlight, in: textView)
        }
    }

    func makeCoordinator() -> Coordinator { Coordinator(parent: self) }

    static func nsRange(of selection: TextSelection?, in text: String) -> NSRange? {
        guard let selection, case .selection(let range) = selection.indices,
              range.lowerBound >= text.startIndex, range.upperBound <= text.endIndex else { return nil }
        let start = text.utf16.distance(from: text.utf16.startIndex, to: range.lowerBound)
        let length = text.utf16.distance(from: range.lowerBound, to: range.upperBound)
        return NSRange(location: start, length: length)
    }

    static func textSelection(_ range: NSRange, in text: String) -> TextSelection? {
        let utf16 = text.utf16
        guard range.upperBound <= utf16.count,
              let start = utf16.index(utf16.startIndex, offsetBy: range.location).samePosition(in: text),
              let end = utf16.index(utf16.startIndex, offsetBy: range.upperBound).samePosition(in: text) else { return nil }
        return TextSelection(range: start..<end)
    }

    final class Coordinator: NSObject, NSTextViewDelegate {
        var parent: ScriptTextView
        var isUpdating = false
        /// The highlight currently drawn, so an unchanged one costs nothing.
        var shown: NSRange??

        init(parent: ScriptTextView) {
            self.parent = parent
        }

        func textDidChange(_ notification: Notification) {
            guard !isUpdating, let textView = notification.object as? NSTextView else { return }
            shown = nil
            parent.text = textView.string
        }

        func textViewDidChangeSelection(_ notification: Notification) {
            guard !isUpdating, let textView = notification.object as? NSTextView else { return }
            let selection = ScriptTextView.textSelection(textView.selectedRange(), in: textView.string)
            if ScriptTextView.nsRange(of: parent.selection, in: textView.string) != textView.selectedRange() {
                parent.selection = selection
            }
        }

        func show(_ range: NSRange?, in textView: NSTextView) {
            let length = (textView.string as NSString).length
            if let layout = textView.textLayoutManager, let content = layout.textContentManager {
                layout.removeRenderingAttribute(.backgroundColor, for: content.documentRange)
                if let range, range.upperBound <= length,
                   let start = content.location(content.documentRange.location, offsetBy: range.location),
                   let end = content.location(start, offsetBy: range.length),
                   let textRange = NSTextRange(location: start, end: end) {
                    layout.addRenderingAttribute(.backgroundColor, value: ScriptTextView.highlightColor, for: textRange)
                }
            } else if let layout = textView.layoutManager {
                layout.removeTemporaryAttribute(.backgroundColor, forCharacterRange: NSRange(location: 0, length: length))
                if let range, range.upperBound <= length {
                    layout.addTemporaryAttribute(.backgroundColor, value: ScriptTextView.highlightColor, forCharacterRange: range)
                }
            }
            if let range, range.upperBound <= length { textView.scrollRangeToVisible(range) }
            shown = .some(range)
        }
    }
}

import CoreText
import Foundation

/// Draws one page per string with Core Text, so PDFKit has real text runs to extract.
enum PDFFixture {
    static func write(pages: [String], title: String?, to url: URL) {
        var mediaBox = CGRect(x: 0, y: 0, width: 612, height: 792)
        var info: [CFString: Any] = [:]
        if let title { info[kCGPDFContextTitle] = title }
        let context = CGContext(url as CFURL, mediaBox: &mediaBox, info as CFDictionary)!
        let font = CTFontCreateWithName("Helvetica" as CFString, 14, nil)
        for text in pages {
            context.beginPDFPage(nil)
            let attributed = NSAttributedString(string: text, attributes: [NSAttributedString.Key(kCTFontAttributeName as String): font])
            let framesetter = CTFramesetterCreateWithAttributedString(attributed)
            let frame = CTFramesetterCreateFrame(framesetter, CFRange(location: 0, length: 0), CGPath(rect: mediaBox.insetBy(dx: 72, dy: 72), transform: nil), nil)
            CTFrameDraw(frame, context)
            context.endPDFPage()
        }
        context.closePDF()
    }
}

import Foundation
@testable import DramaStudio

/// A minimal EPUB: a short cover, a chapter with an `h1`, a chapter titled only
/// by the table of contents (EPUB 3 nav or EPUB 2 NCX), and a non-linear note.
enum EPUBFixture {
    enum TableOfContents { case nav, ncx }

    static let chapterOneBody = "It was a bright cold day in April, and the clocks were striking thirteen. The wind swept dust along the street as she hurried home, past the shuttered shops and the empty square where the market used to stand."
    static let chapterTwoBody = "The path narrowed as the trees closed in overhead. Somewhere ahead a stream ran over stones, and the light turned green and dim. She walked on, counting her steps, listening for anything that was not the wind or the water."

    static func package(toc: TableOfContents = .nav) -> Data {
        var zip = ZipWriter()
        zip.add("mimetype", "application/epub+zip")
        zip.add("META-INF/container.xml", #"<?xml version="1.0"?><container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles></container>"#)
        let tocItem = toc == .nav
            ? #"<item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>"#
            : #"<item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>"#
        zip.add("OEBPS/content.opf", """
        <?xml version="1.0" encoding="UTF-8"?>
        <package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="id">
          <metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>The Long Walk</dc:title></metadata>
          <manifest>
            <item id="cover" href="Text/cover.xhtml" media-type="application/xhtml+xml"/>
            <item id="c1" href="Text/chapter1.xhtml" media-type="application/xhtml+xml"/>
            <item id="c2" href="Text/chapter%202.xhtml" media-type="application/xhtml+xml"/>
            <item id="note" href="Text/note.xhtml" media-type="application/xhtml+xml"/>
            \(tocItem)
          </manifest>
          <spine\(toc == .ncx ? #" toc="ncx""# : "")>
            <itemref idref="cover"/><itemref idref="c1"/><itemref idref="note" linear="no"/><itemref idref="c2"/>
          </spine>
        </package>
        """, method: .deflated)
        zip.add("OEBPS/Text/cover.xhtml", page(title: "Cover", body: #"<div class="cover"><h1>The Long Walk</h1><p>A novel</p></div>"#))
        zip.add("OEBPS/Text/chapter1.xhtml", page(title: "Chapter 1", body: """
        <section><h1>The&nbsp;Beginning</h1>
        <p>\(chapterOneBody)</p>
        <p>She said nothing.<br/>Then &ldquo;Hello.&rdquo;</p>
        <script>var hidden = "do not read";</script>
        </section>
        """), method: .deflated)
        zip.add("OEBPS/Text/chapter 2.xhtml", page(title: "Chapter 2", body: "<p class=\"first\">\(chapterTwoBody)</p>"), method: .deflated)
        zip.add("OEBPS/Text/note.xhtml", page(title: "Note", body: "<p>\(String(repeating: "A translator's note that is not part of the story. ", count: 6))</p>"))
        switch toc {
        case .nav:
            zip.add("OEBPS/nav.xhtml", page(title: "Contents", body: """
            <nav epub:type="toc"><ol>
              <li><a href="Text/chapter1.xhtml">Chapter One</a></li>
              <li><a href="Text/chapter%202.xhtml#top">Into the
                 Woods</a></li>
            </ol></nav>
            """))
        case .ncx:
            zip.add("OEBPS/toc.ncx", """
            <?xml version="1.0"?><ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1"><navMap>
            <navPoint id="n1"><navLabel><text>Chapter One</text></navLabel><content src="Text/chapter1.xhtml"/></navPoint>
            <navPoint id="n2"><navLabel><text>Into the Woods</text></navLabel><content src="Text/chapter%202.xhtml"/></navPoint>
            </navMap></ncx>
            """)
        }
        return zip.finish()
    }

    static func page(title: String, body: String) -> String {
        """
        <?xml version="1.0" encoding="UTF-8"?>
        <html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops"><head><title>\(title)</title><style>p { margin: 0 }</style></head><body>\(body)</body></html>
        """
    }
}

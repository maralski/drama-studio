// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation
@testable import DramaStudio

enum PPTXFixture {
    struct Slide {
        var file: Int
        var title: String?
        var body: [String] = []
        /// Notes paragraphs; nil writes no notes part at all.
        var notes: [[String]]?
    }

    static let ns = #"xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main""#
    static let relNS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"

    static let standard: [Slide] = [
        Slide(file: 1, title: "Welcome", notes: [["Good morning, everyone."], ["Today we cover ", "the numbers."]]),
        Slide(file: 3, title: "Quarterly results", body: ["Revenue up 12%"], notes: [["Revenue grew ", "twelve percent."]]),
        Slide(file: 2, title: "Questions", body: ["Ask anything", "Email us"], notes: nil),
    ]

    static func package(_ slides: [Slide], title: String? = "Quarterly Review", method: ZipArchive.Method = .deflated) -> Data {
        var zip = ZipWriter()
        zip.add("[Content_Types].xml", #"<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="xml" ContentType="application/xml"/></Types>"#)
        zip.add("_rels/.rels", rels([("rId1", "/officeDocument", "ppt/presentation.xml")]))
        if let title {
            zip.add("docProps/core.xml", #"<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>\#(title)</dc:title></cp:coreProperties>"#)
        }
        let ids = slides.enumerated().map { #"<p:sldId id="\#(256 + $0.offset)" r:id="rId\#($0.offset + 10)"/>"# }.joined()
        zip.add("ppt/presentation.xml", #"<p:presentation \#(ns)><p:sldMasterIdLst/><p:sldIdLst>\#(ids)</p:sldIdLst></p:presentation>"#, method: method)
        zip.add("ppt/_rels/presentation.xml.rels", rels([("rId1", "/slideMaster", "slideMasters/slideMaster1.xml")] + slides.enumerated().map { ("rId\($0.offset + 10)", "/slide", "slides/slide\($0.element.file).xml") }))

        for slide in slides {
            var shapes = ""
            if let title = slide.title { shapes += shape(placeholder: #"<p:ph type="title"/>"#, paragraphs: [[title]]) }
            if !slide.body.isEmpty { shapes += shape(placeholder: #"<p:ph idx="1"/>"#, paragraphs: slide.body.map { [$0] }) }
            shapes += shape(placeholder: #"<p:ph type="sldNum" idx="12"/>"#, field: "\(slide.file)")
            shapes += shape(placeholder: #"<p:ph type="ftr" idx="11"/>"#, paragraphs: [["Confidential"]])
            zip.add("ppt/slides/slide\(slide.file).xml", #"<p:sld \#(ns)><p:cSld><p:spTree>\#(shapes)</p:spTree></p:cSld></p:sld>"#, method: method)

            var slideRels = [("rId1", "/slideLayout", "../slideLayouts/slideLayout1.xml")]
            if let notes = slide.notes {
                slideRels.append(("rId2", "/notesSlide", "../notesSlides/notesSlide\(slide.file).xml"))
                let noteShapes = shape(placeholder: #"<p:ph type="sldImg"/>"#, paragraphs: nil)
                    + shape(placeholder: #"<p:ph type="body" idx="1"/>"#, paragraphs: notes)
                    + shape(placeholder: #"<p:ph type="sldNum" idx="5"/>"#, field: "\(slide.file)")
                zip.add("ppt/notesSlides/notesSlide\(slide.file).xml", #"<p:notes \#(ns)><p:cSld><p:spTree>\#(noteShapes)</p:spTree></p:cSld></p:notes>"#, method: method)
            }
            zip.add("ppt/slides/_rels/slide\(slide.file).xml.rels", rels(slideRels))
        }
        return zip.finish()
    }

    private static func rels(_ items: [(String, String, String)]) -> String {
        let body = items.map { #"<Relationship Id="\#($0.0)" Type="\#(relNS)\#($0.1)" Target="\#($0.2)"/>"# }.joined()
        return #"<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\#(body)</Relationships>"#
    }

    private static func shape(placeholder: String, paragraphs: [[String]]?) -> String {
        let nv = #"<p:nvSpPr><p:cNvPr id="2" name="Shape"/><p:cNvSpPr/><p:nvPr>\#(placeholder)</p:nvPr></p:nvSpPr><p:spPr/>"#
        guard let paragraphs else { return "<p:sp>\(nv)</p:sp>" }
        let ps = paragraphs.map { runs in "<a:p>" + runs.map { "<a:r><a:rPr lang=\"en-US\"/><a:t>\(escape($0))</a:t></a:r>" }.joined() + "</a:p>" }.joined()
        return "<p:sp>\(nv)<p:txBody><a:bodyPr/><a:lstStyle/>\(ps)</p:txBody></p:sp>"
    }

    private static func shape(placeholder: String, field: String) -> String {
        let nv = #"<p:nvSpPr><p:cNvPr id="3" name="Slide Number"/><p:cNvSpPr/><p:nvPr>\#(placeholder)</p:nvPr></p:nvSpPr><p:spPr/>"#
        return #"<p:sp>\#(nv)<p:txBody><a:bodyPr/><a:p><a:fld id="{B6F15528}" type="slidenum"><a:t>\#(field)</a:t></a:fld></a:p></p:txBody></p:sp>"#
    }

    static func escape(_ text: String) -> String {
        text.replacingOccurrences(of: "&", with: "&amp;").replacingOccurrences(of: "<", with: "&lt;").replacingOccurrences(of: ">", with: "&gt;")
    }
}

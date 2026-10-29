// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation
import XCTest
@testable import DramaStudio

final class AppZipArchiveTests: XCTestCase {
    private let longText = String(repeating: "The quick brown fox jumps over the lazy dog. ", count: 200)

    func testStoredAndDeflatedEntriesReadBack() throws {
        var writer = ZipWriter()
        writer.add("a.txt", "hello")
        writer.add("dir/b.xml", longText, method: .deflated)
        writer.add("empty.txt", "", method: .deflated)
        let archive = try ZipArchive(data: writer.finish())

        XCTAssertEqual(archive.entries.map(\.path), ["a.txt", "dir/b.xml", "empty.txt"])
        XCTAssertEqual(try archive.data(for: "a.txt"), Data("hello".utf8))
        XCTAssertEqual(try archive.data(for: "dir/b.xml"), Data(longText.utf8))
        XCTAssertEqual(try archive.data(for: "empty.txt"), Data())
        let deflated = try XCTUnwrap(archive.entries.first { $0.path == "dir/b.xml" })
        XCTAssertEqual(deflated.method, 8)
        XCTAssertLessThan(deflated.compressedSize, deflated.uncompressedSize / 4, "the entry really is deflated")
    }

    func testReadsArchiveWrittenByZipTool() throws {
        let dir = try temporaryDirectory()
        try Data(longText.utf8).write(to: dir.appending(path: "story.txt"))
        try FileManager.default.createDirectory(at: dir.appending(path: "sub"), withIntermediateDirectories: true)
        try Data("tiny".utf8).write(to: dir.appending(path: "sub/tiny.txt"))
        try runTool("/usr/bin/zip", ["-q", "-r", "out.zip", "story.txt", "sub"], in: dir)

        let archive = try ZipArchive(url: dir.appending(path: "out.zip"))
        XCTAssertEqual(archive.entries.first { $0.path == "story.txt" }?.method, 8)
        XCTAssertEqual(try archive.data(for: "story.txt"), Data(longText.utf8))
        XCTAssertEqual(try archive.data(for: "sub/tiny.txt"), Data("tiny".utf8))
    }

    func testWriterOutputPassesUnzip() throws {
        let dir = try temporaryDirectory()
        var writer = ZipWriter()
        writer.add("stored.txt", "stored body")
        writer.add("deflated.txt", longText, method: .deflated)
        try writer.finish().write(to: dir.appending(path: "w.zip"))
        XCTAssertEqual(try runTool("/usr/bin/unzip", ["-p", "w.zip", "deflated.txt"], in: dir), Data(longText.utf8))
        XCTAssertEqual(try runTool("/usr/bin/unzip", ["-p", "w.zip", "stored.txt"], in: dir), Data("stored body".utf8))
    }

    func testRejectsNonZipMissingEntryAndBadChecksum() throws {
        XCTAssertThrowsError(try ZipArchive(data: Data(repeating: 7, count: 100))) { XCTAssertEqual($0 as? ZipError, .notAZip) }

        var writer = ZipWriter()
        writer.add("a.txt", "abcdef")
        var bytes = writer.finish()
        let archive = try ZipArchive(data: bytes)
        XCTAssertThrowsError(try archive.data(for: "b.txt")) { XCTAssertEqual($0 as? ZipError, .missingEntry("b.txt")) }

        let payload = try XCTUnwrap(bytes.firstRange(of: Data("abcdef".utf8)))
        bytes[payload.lowerBound] = UInt8(ascii: "X")
        XCTAssertThrowsError(try ZipArchive(data: bytes).data(for: "a.txt")) { XCTAssertEqual($0 as? ZipError, .corrupt("a.txt fails its checksum")) }
    }

    func testZip64ThrowsClearError() throws {
        var writer = ZipWriter()
        writer.add("a.txt", "abc")
        var bytes = writer.finish()
        let eocd = bytes.count - 22
        bytes.replaceSubrange((eocd + 16)..<(eocd + 20), with: [0xFF, 0xFF, 0xFF, 0xFF])
        XCTAssertThrowsError(try ZipArchive(data: bytes)) { error in
            XCTAssertEqual(error as? ZipError, .zip64Unsupported)
            XCTAssertTrue(error.localizedDescription.contains("ZIP64"))
        }
    }

    func testUnsupportedMethodNamesTheEntry() throws {
        var writer = ZipWriter()
        writer.add("a.txt", "abc")
        var bytes = writer.finish()
        let central = try XCTUnwrap(bytes.firstRange(of: Data([0x50, 0x4B, 0x01, 0x02])))
        bytes[central.lowerBound + 10] = 12
        XCTAssertThrowsError(try ZipArchive(data: bytes).data(for: "a.txt")) { XCTAssertEqual($0 as? ZipError, .unsupportedMethod(12, "a.txt")) }
    }

    func testResolvesRelativeTargets() {
        XCTAssertEqual(ZipArchive.resolve("../notesSlides/notesSlide1.xml", relativeTo: "ppt/slides/slide1.xml"), "ppt/notesSlides/notesSlide1.xml")
        XCTAssertEqual(ZipArchive.resolve("slides/slide2.xml", relativeTo: "ppt/presentation.xml"), "ppt/slides/slide2.xml")
        XCTAssertEqual(ZipArchive.resolve("/ppt/slides/slide3.xml", relativeTo: "ppt/presentation.xml"), "ppt/slides/slide3.xml")
        XCTAssertEqual(ZipArchive.resolve("Text/chapter%201.xhtml#start", relativeTo: "OEBPS/content.opf"), "OEBPS/Text/chapter 1.xhtml")
        XCTAssertEqual(ZipArchive.resolve("OEBPS/content.opf", relativeTo: "META-INF/container.xml"), "META-INF/OEBPS/content.opf")
    }

    func testCRC32MatchesKnownValue() {
        XCTAssertEqual(CRC32.checksum(Data("123456789".utf8)), 0xCBF4_3926)
    }
}

func temporaryDirectory() throws -> URL {
    let dir = FileManager.default.temporaryDirectory.appending(path: "DramaStudioTests-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    return dir
}

@discardableResult
func runTool(_ tool: String, _ arguments: [String], in directory: URL) throws -> Data {
    let process = Process()
    process.executableURL = URL(fileURLWithPath: tool)
    process.arguments = arguments
    process.currentDirectoryURL = directory
    let pipe = Pipe()
    process.standardOutput = pipe
    try process.run()
    let output = pipe.fileHandleForReading.readDataToEndOfFile()
    process.waitUntilExit()
    XCTAssertEqual(process.terminationStatus, 0, "\(tool) \(arguments)")
    return output
}

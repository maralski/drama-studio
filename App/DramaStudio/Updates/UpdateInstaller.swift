// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import CryptoKit
import Foundation
import Security

/// All disk image, hashing and process work runs outside the main actor.
enum UpdateInstaller {
    static func run(_ executable: String, _ arguments: [String]) throws {
        let process = Process()
        process.executableURL = URL(filePath: executable)
        process.arguments = arguments
        process.standardOutput = FileHandle.nullDevice
        process.standardError = FileHandle.nullDevice
        try process.run()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else { throw UpdateError.message("Update preparation failed (\(URL(filePath: executable).lastPathComponent)). Please try again.") }
    }

    static func prepare(image: URL, checksum: String, version: String, currentApp: URL, directory: URL,
                        systemVersion: OperatingSystemVersion = ProcessInfo.processInfo.operatingSystemVersion) throws -> URL {
        let handle = try FileHandle(forReadingFrom: image)
        defer { try? handle.close() }
        var hash = SHA256()
        while let bytes = try handle.read(upToCount: 1024 * 1024), !bytes.isEmpty { hash.update(data: bytes) }
        guard hash.finalize().map({ String(format: "%02x", $0) }).joined() == checksum else {
            throw UpdateError.message("The downloaded update failed checksum verification.")
        }
        let mount = directory.appending(path: "mount")
        try FileManager.default.createDirectory(at: mount, withIntermediateDirectories: true)
        try run("/usr/bin/hdiutil", ["attach", image.path, "-readonly", "-nobrowse", "-mountpoint", mount.path])
        defer { try? run("/usr/bin/hdiutil", ["detach", mount.path]) }
        let staged = directory.appending(path: "DramaStudio.app")
        try run("/usr/bin/ditto", [mount.appending(path: "DramaStudio.app").path, staged.path])
        guard let bundle = Bundle(url: staged), bundle.bundleIdentifier == "io.github.maralski.DramaStudio",
              bundle.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String == version,
              FileManager.default.isExecutableFile(atPath: staged.appending(path: "Contents/MacOS/DramaStudio").path) else {
            throw UpdateError.message("The downloaded app does not match this release.")
        }
        guard let minimum = bundle.object(forInfoDictionaryKey: "LSMinimumSystemVersion") as? String else {
            throw UpdateError.message("The update does not declare its minimum macOS version.")
        }
        let parts = minimum.split(separator: ".", omittingEmptySubsequences: false)
        guard (1...3).contains(parts.count),
              let required = AppVersion((parts.map(String.init) + Array(repeating: "0", count: 3 - parts.count)).joined(separator: ".")),
              let installed = AppVersion("\(systemVersion.majorVersion).\(systemVersion.minorVersion).\(systemVersion.patchVersion)") else {
            throw UpdateError.message("The update declares an invalid minimum macOS version.")
        }
        guard installed >= required else {
            throw UpdateError.message("This update requires macOS \(minimum) or later. Your current app has not been changed.")
        }
        try run("/usr/bin/codesign", ["--verify", "--deep", "--strict", staged.path])
        // Signed distributions must retain their designated signing requirement.
        // Ad-hoc distributions are authenticated by the repository's HTTPS checksum.
        var currentCode: SecStaticCode?
        guard SecStaticCodeCreateWithPath(currentApp as CFURL, [], &currentCode) == errSecSuccess, let currentCode else {
            throw UpdateError.message("The running app's signature could not be read.")
        }
        var information: CFDictionary?
        guard SecCodeCopySigningInformation(currentCode, SecCSFlags(rawValue: kSecCSSigningInformation), &information) == errSecSuccess else {
            throw UpdateError.message("The running app's signing information could not be read.")
        }
        if (information as? [String: Any])?[kSecCodeInfoTeamIdentifier as String] != nil {
            var requirement: SecRequirement?
            var candidate: SecStaticCode?
            guard SecCodeCopyDesignatedRequirement(currentCode, [], &requirement) == errSecSuccess,
                  SecStaticCodeCreateWithPath(staged as CFURL, [], &candidate) == errSecSuccess,
                  let candidate, let requirement,
                  SecStaticCodeCheckValidity(candidate, SecCSFlags(rawValue: kSecCSCheckAllArchitectures), requirement) == errSecSuccess else {
                throw UpdateError.message("The update's signing identity does not match this app.")
            }
        }
        return staged
    }

    // Arguments are passed separately, never interpolated into shell source.
    static let script = #"""
    set -eu
    pid="$1"; target="$2"; staged="$3"; work="$4"
    backup="$work/previous.app"
    count=0
    while kill -0 "$pid" 2>/dev/null; do
        count=$((count + 1))
        [ "$count" -lt 60 ] || exit 1
        sleep 1
    done
    if ! /bin/mv "$target" "$backup"; then
        /usr/bin/open "$target"
        exit 1
    fi
    if ! /bin/mv "$staged" "$target"; then
        /bin/mv "$backup" "$target"
        /usr/bin/open "$target"
        exit 1
    fi
    if ! /usr/bin/open "$target"; then
        /bin/mv "$target" "$staged"
        /bin/mv "$backup" "$target"
        /usr/bin/open "$target"
        exit 1
    fi
    /bin/rm -rf "$work"
    """#

    static func launch(staged: URL, target: URL, directory: URL) throws {
        let helper = directory.appending(path: "install.sh")
        try script.write(to: helper, atomically: true, encoding: .utf8)
        let process = Process()
        process.executableURL = URL(filePath: "/bin/sh")
        process.arguments = [helper.path, String(ProcessInfo.processInfo.processIdentifier), target.path, staged.path, directory.path]
        process.standardOutput = FileHandle.nullDevice
        process.standardError = FileHandle.nullDevice
        try process.run()
    }
}

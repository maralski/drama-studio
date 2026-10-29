// swift-tools-version:6.0
// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import PackageDescription

let package = Package(
    name: "SpeakCLI",
    platforms: [.macOS(.v15)],
    dependencies: [.package(name: "DramaStudio", path: "../..")],
    targets: [
        .executableTarget(name: "SpeakCLI", dependencies: [.product(name: "DramaEngine", package: "DramaStudio")]),
    ]
)

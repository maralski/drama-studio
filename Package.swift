// swift-tools-version:6.0
// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import PackageDescription

// tools/make_xcframework.sh sets this to archive DramaEngine as a framework.
let engineLibraryType: Product.Library.LibraryType? = Context.environment["DRAMA_ENGINE_DYNAMIC"] == "1" ? .dynamic : nil

let package = Package(
    name: "DramaStudio",
    platforms: [.macOS(.v15)],
    products: [
        .library(name: "DramaEngine", type: engineLibraryType, targets: ["DramaEngine"]),
        .library(name: "DramaServer", targets: ["DramaServer"]),
        .executable(name: "drama", targets: ["drama"]),
        .executable(name: "DramaStudio", targets: ["DramaStudio"]),
        .executable(name: "drama-mcp", targets: ["drama-mcp"]),
        .executable(name: "drama-say", targets: ["drama-say"]),
        .executable(name: "drama-word-recognizer", targets: ["drama-word-recognizer"]),
    ],
    targets: [
        // Run tools/bootstrap_word_recognition.py before a clean build.
        .binaryTarget(name: "whisper", path: "Vendor/WordRecognition/whisper.xcframework"),
        .target(name: "WordRecognitionProtocol", exclude: ["Assets.json"]),
        .executableTarget(name: "drama-word-recognizer", dependencies: ["WordRecognitionProtocol", "whisper"]),
        .target(
            name: "DramaEngine",
            dependencies: ["WordRecognitionProtocol"],
            path: "Sources/DramaEngine",
            resources: [.copy("Kernels"), .copy("CodecConversion"), .copy("Verification/EnglishSpokenForms.json")]
        ),
        .target(name: "DramaServer", dependencies: ["DramaEngine"], path: "Sources/DramaServer"),
        .target(name: "DramaMCP", path: "Sources/DramaMCP"),
        .executableTarget(name: "drama-mcp", dependencies: ["DramaMCP"], path: "Sources/drama-mcp"),
        .executableTarget(name: "drama-say", dependencies: ["DramaMCP"], path: "Sources/drama-say"),
        .executableTarget(name: "drama", dependencies: ["DramaEngine", "DramaServer"], path: "Sources/drama"),
        .executableTarget(name: "DramaStudio", dependencies: ["DramaEngine", "DramaServer"], path: "App/DramaStudio"),
        .testTarget(name: "DramaEngineTests", dependencies: ["DramaEngine", "WordRecognitionProtocol"], path: "Tests/DramaEngineTests", resources: [.copy("Fixtures")]),
        .testTarget(name: "DramaServerTests", dependencies: ["DramaServer", "DramaEngine"], path: "Tests/DramaServerTests"),
        .testTarget(name: "DramaMCPTests", dependencies: ["DramaMCP"], path: "Tests/DramaMCPTests"),
        .testTarget(name: "DramaStudioTests", dependencies: ["DramaStudio", "DramaEngine", "DramaServer"], path: "Tests/DramaStudioTests"),
    ],
    swiftLanguageVersions: [.v5]
)

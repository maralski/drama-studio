// swift-tools-version:6.0
import PackageDescription

let package = Package(
    name: "SpeakCLI",
    platforms: [.macOS(.v15)],
    dependencies: [.package(name: "DramaStudio", path: "../..")],
    targets: [
        .executableTarget(name: "SpeakCLI", dependencies: [.product(name: "DramaEngine", package: "DramaStudio")]),
    ]
)

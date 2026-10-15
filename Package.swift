// swift-tools-version:6.0
import PackageDescription

let package = Package(
    name: "DramaStudio",
    platforms: [.macOS(.v15)],
    products: [
        .library(name: "DramaEngine", targets: ["DramaEngine"]),
        .executable(name: "drama", targets: ["drama"]),
        .executable(name: "DramaStudio", targets: ["DramaStudio"]),
    ],
    targets: [
        .target(
            name: "DramaEngine",
            path: "Sources/DramaEngine",
            resources: [.copy("Kernels")]
        ),
        .executableTarget(name: "drama", dependencies: ["DramaEngine"], path: "Sources/drama"),
        .executableTarget(name: "DramaStudio", dependencies: ["DramaEngine"], path: "App/DramaStudio"),
        .testTarget(name: "DramaEngineTests", dependencies: ["DramaEngine"], path: "Tests/DramaEngineTests"),
    ],
    swiftLanguageVersions: [.v5]
)

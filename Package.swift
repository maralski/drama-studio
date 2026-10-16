// swift-tools-version:6.0
import PackageDescription

let package = Package(
    name: "DramaStudio",
    platforms: [.macOS(.v15)],
    products: [
        .library(name: "DramaEngine", targets: ["DramaEngine"]),
        .library(name: "DramaServer", targets: ["DramaServer"]),
        .executable(name: "drama", targets: ["drama"]),
        .executable(name: "DramaStudio", targets: ["DramaStudio"]),
    ],
    targets: [
        .target(
            name: "DramaEngine",
            path: "Sources/DramaEngine",
            resources: [.copy("Kernels")]
        ),
        .target(name: "DramaServer", dependencies: ["DramaEngine"], path: "Sources/DramaServer"),
        .executableTarget(name: "drama", dependencies: ["DramaEngine", "DramaServer"], path: "Sources/drama"),
        .executableTarget(name: "DramaStudio", dependencies: ["DramaEngine", "DramaServer"], path: "App/DramaStudio"),
        .testTarget(name: "DramaEngineTests", dependencies: ["DramaEngine"], path: "Tests/DramaEngineTests"),
        .testTarget(name: "DramaServerTests", dependencies: ["DramaServer", "DramaEngine"], path: "Tests/DramaServerTests"),
        .testTarget(name: "DramaStudioTests", dependencies: ["DramaStudio", "DramaEngine"], path: "Tests/DramaStudioTests"),
    ],
    swiftLanguageVersions: [.v5]
)

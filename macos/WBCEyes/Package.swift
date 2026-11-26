// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "WBCEyes",
    platforms: [
        .macOS(.v13)
    ],
    products: [
        .library(name: "WBCEyesCore", targets: ["WBCEyesCore"]),
        .executable(name: "WholeBodyCompute", targets: ["WBCEyes"])
    ],
    targets: [
        .target(
            name: "WBCEyesCore",
            path: "Sources/WBCEyesCore",
            linkerSettings: [
                .linkedFramework("AppKit"),
                .linkedFramework("ApplicationServices"),
                .linkedFramework("AVFoundation"),
                .linkedFramework("CoreGraphics"),
                .linkedFramework("Security"),
                .linkedFramework("Speech"),
                .linkedFramework("Foundation")
            ]
        ),
        .executableTarget(
            name: "WBCEyes",
            dependencies: ["WBCEyesCore"],
            path: "Sources/WBCEyes"
        ),
        .testTarget(
            name: "WBCEyesTests",
            dependencies: ["WBCEyesCore"],
            path: "Tests/WBCEyesTests"
        )
    ]
)

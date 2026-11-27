// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "WholeBodyCompute",
    platforms: [
        .macOS(.v13)
    ],
    products: [
        .library(name: "WholeBodyComputeCore", targets: ["WholeBodyComputeCore"]),
        .executable(name: "WholeBodyCompute", targets: ["WholeBodyCompute"])
    ],
    targets: [
        .target(
            name: "WholeBodyComputeCore",
            path: "Sources/WholeBodyComputeCore",
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
            name: "WholeBodyCompute",
            dependencies: ["WholeBodyComputeCore"],
            path: "Sources/WholeBodyCompute"
        ),
        .testTarget(
            name: "WholeBodyComputeTests",
            dependencies: ["WholeBodyComputeCore"],
            path: "Tests/WholeBodyComputeTests"
        )
    ]
)

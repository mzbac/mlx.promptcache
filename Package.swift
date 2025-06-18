// swift-tools-version: 6.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "mlx.promptcache",
    platforms: [.macOS(.v14), .iOS(.v16)],
    products: [
        .library(
            name: "mlx_promptcache",
            targets: ["mlx_promptcache"])
    ],
    dependencies: [
        .package(
            url: "https://github.com/ml-explore/mlx-swift.git", .upToNextMinor(from: "0.25.4")),

        .package(
            url: "https://github.com/ml-explore/mlx-swift-examples.git",
            .upToNextMinor(from: "2.25.4")),
    ],
    targets: [
        .target(
            name: "mlx_promptcache",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXLLM", package: "mlx-swift-examples"),
                .product(name: "MLXLMCommon", package: "mlx-swift-examples"),
            ]),
        .testTarget(
            name: "mlx.promptcacheTests",
            dependencies: ["mlx_promptcache"]
        ),
    ]
)

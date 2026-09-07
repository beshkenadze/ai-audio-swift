// swift-tools-version:6.2
import PackageDescription

// A standalone package, deliberately NOT a target of the repository's root
// Package.swift.
//
// The root manifest is the one upstream consumers resolve. Adding an omni-bench
// dependency there would force every consumer of mlx-audio-swift to fetch a
// benchmark harness they will never call, and would not survive review on the
// upstream PR. Keeping the host here lets it depend on both packages while the
// library manifest stays clean.
//
// omni-bench is resolved by path because it is not published. Override with
// OMNI_BENCH_PATH if your checkout lives elsewhere.
let omniBenchPath = Context.environment["OMNI_BENCH_PATH"] ?? "../../../omni-bench"

let package = Package(
    name: "OmniBenchVibeVoiceHost",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "OmniBenchVibeVoiceHost", targets: ["OmniBenchVibeVoiceHost"]),
        .executable(name: "vibevoice-omni-bench-run", targets: ["vibevoice-omni-bench-run"]),
    ],
    dependencies: [
        .package(path: "../.."),
        .package(path: omniBenchPath),
    ],
    targets: [
        .target(
            name: "OmniBenchVibeVoiceHost",
            dependencies: [
                .product(name: "MLXAudioSTT", package: "mlx-audio-swift"),
                .product(name: "OmniBench", package: "omni-bench"),
            ]
        ),
        .executableTarget(
            name: "vibevoice-omni-bench-run",
            dependencies: ["OmniBenchVibeVoiceHost"]
        ),
        .testTarget(
            name: "OmniBenchVibeVoiceHostTests",
            dependencies: ["OmniBenchVibeVoiceHost"],
            resources: [.copy("Fixtures")]
        ),
    ]
)

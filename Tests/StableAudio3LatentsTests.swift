import Foundation
import AVFoundation
import MLX
import Testing

@testable import MLXAudioMusic

@Suite(.serialized)
struct StableAudio3LatentsTests {
    @Test func fileRoundTripAndSegmentsPreserveBits() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("\(UUID()).safetensors")
        defer { try? FileManager.default.trashItem(at: url, resultingItemURL: nil) }
        let values = MLXArray((0..<(256 * 7)).map { Float($0) / 1000 }, [1, 256, 7]).asType(.float16)
        let bank = try StableAudio3.LatentAudio(latents: values, variant: .medium, sampleCount: 7 * 4096 - 13)
        try bank.save(to: url)
        let loaded = try StableAudio3.LatentAudio(contentsOf: url)
        #expect(loaded.variant == .medium && loaded.sampleCount == bank.sampleCount)
        #expect(bitwiseEqual(loaded.latents, values))
        #expect(bitwiseEqual(try loaded.segment(length: 3, offset: 2, variant: .medium), values[.ellipsis, 2..<5]))
        #expect(bitwiseEqual(try loaded.segment(length: 2, offset: 5, variant: .medium), values[.ellipsis, 5..<7]))
        #expect(throws: (any Error).self) { try loaded.segment(length: 4, offset: 5, variant: .medium) }
        #expect(throws: (any Error).self) { try loaded.segment(length: 3, offset: 0, variant: .smallMusic) }
        #expect(throws: (any Error).self) { try loaded.segment(length: 3, offset: 7, variant: .medium) }
        #expect(throws: (any Error).self) { try loaded.segment(length: 3, offset: -1, variant: .medium) }
    }

    @Test func malformedBanksAreRejected() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("\(UUID()).safetensors")
        defer { try? FileManager.default.trashItem(at: url, resultingItemURL: nil) }
        try MLX.save(arrays: ["latents": MLXArray.zeros([1, 256, 1]).asType(.float16)], url: url)
        #expect(throws: (any Error).self) { try StableAudio3.LatentAudio(contentsOf: url) }
        #expect(throws: (any Error).self) {
            try StableAudio3.LatentAudio(latents: MLXArray.zeros([1, 256, 2]), variant: .medium, sampleCount: 8192)
        }
        #expect(throws: (any Error).self) {
            try StableAudio3.LatentAudio(latents: MLXArray.zeros([1, 256, 2]).asType(.float16), variant: .medium, sampleCount: 1)
        }
        #expect(throws: (any Error).self) {
            try StableAudio3.LatentAudio(latents: MLXArray.full([1, 256, 1], values: MLXArray(Float.nan)).asType(.float16),
                                        variant: .medium, sampleCount: 4096)
        }
    }

    @Test func secondsOffsetsRoundToTheLatentGrid() throws {
        let position = Double(StableAudio3.samplesPerLatent) / Double(StableAudio3.sampleRate)
        #expect(try StableAudio3.latentOffset(seconds: position * 2) == 2)
        #expect(try StableAudio3.latentOffset(seconds: position * 2.5) == 2)
        #expect(try StableAudio3.latentOffset(seconds: position * 3.5) == 4)
        #expect(throws: (any Error).self) { try StableAudio3.latentOffset(seconds: .nan) }
        #expect(throws: (any Error).self) { try StableAudio3.latentOffset(seconds: -1) }
    }

    @Test func boundedReadsPreserveResamplerPhaseAndEOFPadding() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("\(UUID()).caf")
        defer { try? FileManager.default.trashItem(at: url, resultingItemURL: nil) }
        let format = try #require(AVAudioFormat(standardFormatWithSampleRate: 48_000, channels: 2))
        let pcm = try #require(AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 4800))
        pcm.frameLength = pcm.frameCapacity
        for i in 0..<4800 {
            pcm.floatChannelData![0][i] = Float(sin(Double(i) * 0.07))
            pcm.floatChannelData![1][i] = Float(cos(Double(i) * 0.13))
        }
        do { let file = try AVAudioFile(forWriting: url, settings: format.settings); try file.write(from: pcm) }
        let whole = try StableAudio3.loadInitAudio(url, frames: 10_001)
        #expect(abs(whole[.ellipsis, ..<4400]).max().item(Float.self) > 0.5)
        let reader = try StableAudio3AudioInput(url: url)
        let parts = try [3, 4095, 9, 5894].map { try reader.read(frames: $0) }
        #expect(bitwiseEqual(whole, concatenated(parts, axis: 2)))
        #expect(try reader.read(frames: 17).asArray(Float.self).allSatisfy { $0 == 0 })
    }
}

@Suite(.serialized, .enabled(if: ProcessInfo.processInfo.environment["MLXAUDIO_SA3_LATENTS_SOURCE"] != nil))
struct StableAudio3LatentsParityTests {
    @Test func boundedLongEncodingMatchesWholeEncodingAtTheFinalLookahead() throws {
        let environment = ProcessInfo.processInfo.environment
        let weights = URL(fileURLWithPath: try #require(environment["MLXAUDIO_SA3_WEIGHTS"]))
        let source = URL(fileURLWithPath: try #require(environment["MLXAUDIO_SA3_LATENTS_SOURCE"]))
        let variant = try #require(StableAudio3.Variant(rawValue: environment["MLXAUDIO_SA3_VARIANT"] ?? "sm-music"))
        Memory.cacheLimit = 256 * 1_048_576
        let seconds = Double(1294 * 4096) / 44_100
        let count = StableAudio3.latentLength(seconds: seconds)
        let length = StableAudio3.encoderLatentLength(count, variant: variant)
        let input = try StableAudio3.loadInitAudio(source, frames: length * 4096)
        let expected = try StableAudio3.encodeAudio(StableAudio3.patch(input),
            weights: weights.appendingPathComponent(variant.encoderWeights), variant: variant)[.ellipsis, ..<count].asType(.float16)
        eval(expected)
        Memory.clearCache()
        let actual = try StableAudio3(variant: variant, weightsDirectory: weights).encodeAudioFile(source, seconds: seconds)
        let delta = abs(actual.latents.asType(.float32) - expected.asType(.float32))
        print("SA3 \(variant.rawValue) bounded encoder: max Δ \(delta.max().item(Float.self)), RMS Δ \(sqrt(mean(square(delta))).item(Float.self))")
        #expect(delta.max().item(Float.self) < 1e-3)
        #expect(sqrt(mean(square(delta))).item(Float.self) < 1e-4)
    }

    @Test(arguments: [StableAudio3.DiTPrecision.reference, .float16])
    func encodedFileMatchesDirectAudioWithoutAnEncoderAtGeneration(precision: StableAudio3.DiTPrecision) throws {
        let environment = ProcessInfo.processInfo.environment
        let weights = URL(fileURLWithPath: try #require(environment["MLXAUDIO_SA3_WEIGHTS"]))
        let source = URL(fileURLWithPath: try #require(environment["MLXAUDIO_SA3_LATENTS_SOURCE"]))
        let variant = try #require(StableAudio3.Variant(rawValue: environment["MLXAUDIO_SA3_VARIANT"] ?? "sm-music"))
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("SA3Latents-\(UUID())")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.trashItem(at: directory, resultingItemURL: nil) }
        Memory.cacheLimit = 256 * 1_048_576
        let engine = StableAudio3(variant: variant, weightsDirectory: weights)
        let file = directory.appendingPathComponent("source.safetensors")
        let seconds = 10.0
        let offset = 0.25
        try engine.encodeAudioFile(source, seconds: seconds, offset: offset).save(to: file)
        let request = StableAudio3.Request(prompt: "Warm analog pads, gentle motion.", seconds: seconds, seed: 7,
                                           precision: precision, initAudio: source, initNoiseLevel: 0.7, initAudioOffset: offset)
        let direct = try engine.generate(request)
        for name in variant.requiredFiles {
            try FileManager.default.createSymbolicLink(at: directory.appendingPathComponent(name),
                                                       withDestinationURL: weights.appendingPathComponent(name))
        }
        let cachedEngine = StableAudio3(variant: variant, weightsDirectory: directory)
        var cached = request
        cached.initAudio = nil
        cached.initAudioOffset = 0
        cached.initLatents = file
        var encodedAgain = false
        let fromFile = try cachedEngine.generate(cached) { if case .encodingAudio = $0 { encodedAgain = true } }
        #expect(!encodedAgain)
        #expect(bitwiseEqual(direct, fromFile))
        print("SA3 \(variant.rawValue): init-audio and saved init-latents outputs are bit-identical")
    }
}

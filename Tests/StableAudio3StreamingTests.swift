import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXAudioMusic

/// Same dtype, shape and bytes. `arrayEqual` compares values, so it would let a +0/-0
/// or a dtype difference through.
func bitwiseEqual(_ a: MLXArray, _ b: MLXArray) -> Bool {
    a.dtype == b.dtype && a.shape == b.shape && a.asData(access: .copy).data == b.asData(access: .copy).data
}

/// Streaming blocks out of the archive changes where the weights live, not one bit of
/// what the model computes. Untrained weights: only the plumbing is under test.
struct StableAudio3StreamingTests {
    private func smallConfig(differential: Bool) -> StableAudio3DiTConfig {
        var config = StableAudio3DiTConfig.smallMusic
        config.embedDimension = 128
        config.depth = 3
        config.heads = 2
        config.ropeDimensions = 16
        config.memoryTokens = 4
        config.feedForwardInner = 256
        config.differentialAttention = differential
        return config
    }

    private func velocities(_ model: StableAudio3DiT) -> [MLXArray] {
        let x = MLXRandom.normal([1, 256, 12], key: MLXRandom.key(1)).asType(.float16)
        let t = MLXArray([Float(0.7)])
        let cross = MLXRandom.normal([1, 5, 768], key: MLXRandom.key(2)).asType(.float16)
        let global = MLXRandom.normal([1, 768], key: MLXRandom.key(3)).asType(.float16)
        return [false, true].map { fp16 in
            model.computeInFloat16 = fp16
            return model(x, t: t, crossAttnCond: cross, globalCond: global)
        }
    }

    @Test(arguments: [false, true])
    func streamedTransformerMatchesResident(differential: Bool) throws {
        let config = smallConfig(differential: differential)
        let url = try NPZFixture.write(StableAudio3DiT(config))
        defer { try? FileManager.default.removeItem(at: url) }
        let resident = try StableAudio3.loadDiT(url, config: config)
        let streamed = try StableAudio3.loadDiT(url, config: config, streamsBlocks: true)
        #expect(streamed.transformer.layers.isEmpty)
        for (a, b) in zip(velocities(resident), velocities(streamed)) {
            #expect(bitwiseEqual(a, b))
        }
    }

    @Test func streamedQuantizedTransformerMatchesResident() throws {
        let config = smallConfig(differential: true)
        let source = StableAudio3DiT(config)
        // Like the exported presets: only projections whose input splits into groups.
        quantize(model: source, groupSize: 64, bits: 8, filter: { _, module in
            (module as? Linear).map { $0.weight.dim(1) % 64 == 0 } ?? false
        })
        let url = try NPZFixture.write(source, quantization: (groupSize: 64, bits: 8))
        defer { try? FileManager.default.removeItem(at: url) }
        let resident = try StableAudio3.loadDiT(url, config: config)
        let streamed = try StableAudio3.loadDiT(url, config: config, streamsBlocks: true)
        #expect(streamed.transformer.blocks?.block.selfAttention.toQKV is QuantizedLinear)
        for (a, b) in zip(velocities(resident), velocities(streamed)) {
            #expect(bitwiseEqual(a, b))
        }
    }

    @Test func blockStreamRejectsAnArchiveMissingABlockWeight() throws {
        let config = smallConfig(differential: false)
        let url = try NPZFixture.write(StableAudio3DiT(config), excluding: ["transformer.layers.2.ff.out.bias"])
        defer { try? FileManager.default.removeItem(at: url) }
        #expect(throws: NPZArchiveError.self) {
            _ = try StableAudio3BlockStream(archive: try NPZArchive(url: url), config: config)
        }
    }

    @Test func bitwiseEqualTellsSignedZerosApart() {
        let positive = MLXArray([Float(0)]), negative = MLXArray([-Float(0)])
        #expect(arrayEqual(positive, negative).item(Bool.self))
        #expect(!bitwiseEqual(positive, negative))
        #expect(bitwiseEqual(positive, MLXArray([Float(0)])))
    }

    @Test func textCacheKeyFollowsThePromptAndTheEncoderFile() throws {
        let weights = FileManager.default.temporaryDirectory.appendingPathComponent("\(UUID().uuidString).npz")
        try Data("first".utf8).write(to: weights)
        defer { try? FileManager.default.removeItem(at: weights) }
        let key = try StableAudio3.textCacheKey(prompt: "warm pads", weights: weights)
        #expect(key == (try StableAudio3.textCacheKey(prompt: "warm pads", weights: weights)))
        #expect(key != (try StableAudio3.textCacheKey(prompt: "warm pads ", weights: weights)))
        try Data("second, longer".utf8).write(to: weights)
        #expect(key != (try StableAudio3.textCacheKey(prompt: "warm pads", weights: weights)))
    }
}

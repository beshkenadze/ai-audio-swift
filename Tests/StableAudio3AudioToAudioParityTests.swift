import Foundation
import MLX
import Testing

@testable import MLXAudioMusic

private var audioToAudioEnvironment: [String: String] { ProcessInfo.processInfo.environment }

@Suite(.serialized, .enabled(if: audioToAudioEnvironment["MLXAUDIO_SA3_A2A_REFERENCE"] != nil))
struct StableAudio3AudioToAudioParityTests {
    let weights: URL
    let reference: NPZArchive
    let variant: StableAudio3.Variant

    init() throws {
        Memory.clearCache()
        Memory.cacheLimit = 256 * 1_048_576
        weights = URL(fileURLWithPath: try #require(audioToAudioEnvironment["MLXAUDIO_SA3_WEIGHTS"]))
        reference = try NPZArchive(url: URL(fileURLWithPath: try #require(audioToAudioEnvironment["MLXAUDIO_SA3_A2A_REFERENCE"])))
        variant = try #require(StableAudio3.Variant(rawValue: audioToAudioEnvironment["MLXAUDIO_SA3_VARIANT"] ?? "sm-music"))
    }

    private func ref(_ name: String) throws -> MLXArray { try reference.array(name) }
    private func difference(_ a: MLXArray, _ b: MLXArray) -> Float {
        abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
    }

    @Test func patchesAndEncoderMatch() throws {
        let patches = StableAudio3.patch(try ref("audio_in"))
        #expect(bitwiseEqual(patches, try ref("patches_in")))
        let actual = try StableAudio3.encodeAudio(patches, weights: weights.appendingPathComponent(variant.encoderWeights), variant: variant)
        let expected = try ref("init_latents")
        let delta = difference(actual[.ellipsis, ..<expected.dim(2)], expected)
        print("SA3 a2a encoder max |Δ| \(delta)")
        #expect(delta < 1e-3)
    }

    @Test func mixedNoiseAndScheduleMatch() throws {
        let params = try ref("params").asArray(Float.self)
        let initial = try ref("init_latents").asType(.float16)
        let pure = MLXRandom.normal(initial.shape, dtype: .float16, key: MLXRandom.key(UInt64(params[1])))
        let referencePure = try ref("pure_noise")
        #expect(allClose(pure.asType(.float32), referencePure.asType(.float32), rtol: 1e-3, atol: 1e-6).item(Bool.self))
        // Isolate mixing from the accepted one-step rounding of `normal` across MLX builds.
        let actual = StableAudio3.initialNoise(referencePure, initLatents: initial, sigma: params[3])
        #expect(bitwiseEqual(actual, try ref("noise")))
        #expect(bitwiseEqual(StableAudio3.schedule(steps: Int(params[2]), sigmaMax: params[3]), try ref("sigmas")))
    }

    @Test func transformerAndSamplerMatch() throws {
        let params = try ref("params").asArray(Float.self)
        let model = try StableAudio3.loadDiT(weights.appendingPathComponent(variant.ditWeights),
                                            config: variant.ditConfig, streamsBlocks: true)
        let (noise, sigmas) = (try ref("noise"), try ref("sigmas"))
        let (cross, global) = (try ref("cross_attn"), try ref("global_cond"))
        let v0 = model(noise, t: sigmas[0] * MLXArray.ones([1], dtype: .float16), crossAttnCond: cross, globalCond: global)
        let velocityDelta = difference(v0, try ref("v0"))
        print("SA3 a2a velocity max |Δ| \(velocityDelta)")
        #expect(velocityDelta < 1e-2)
        let latents = StableAudio3.sample(noise: noise, sigmas: sigmas, seed: UInt64(params[1]) + 1) {
            model($0, t: $1, crossAttnCond: cross, globalCond: global)
        }
        let latentDelta = difference(latents, try ref("latents"))
        print("SA3 a2a sampled latents max |Δ| \(latentDelta)")
        #expect(latentDelta < 5e-2)
    }

    @Test func endToEndMatches() throws {
        let params = try ref("params").asArray(Float.self)
        let prompt = String(decoding: try reference.bytes("prompt"), as: UTF8.self)
        let input = URL(fileURLWithPath: String(decoding: try reference.bytes("source"), as: UTF8.self))
        let audio = try StableAudio3(variant: variant, weightsDirectory: weights).generate(
            .init(prompt: prompt, seconds: Double(params[0]), seed: UInt64(params[1]), steps: Int(params[2]),
                  initAudio: input, initNoiseLevel: params[3]))
        let expected = try ref("audio")[0]
        #expect(audio.shape == expected.shape)
        let rms = sqrt(mean(square(audio - expected))).item(Float.self)
        print("SA3 a2a end-to-end RMS Δ \(rms)")
        #expect(rms < 1e-2)
    }
}

@Suite(.serialized, .enabled(if: audioToAudioEnvironment["MLXAUDIO_SA3_ENCODER_WEIGHTS"] != nil))
struct StableAudio3EncoderStreamingTests {
    private var weights: URL {
        URL(fileURLWithPath: audioToAudioEnvironment["MLXAUDIO_SA3_ENCODER_WEIGHTS"]!)
    }

    private func resident(_ patches: MLXArray, chunk: Int?) throws -> MLXArray {
        defer { Memory.clearCache() }
        let (model, _) = try StableAudio3.loadMediumEncoder(weights, streamsBlocks: false)
        let result = model(patches, chunk: chunk)
        eval(result)
        return result
    }

    @Test func streamedEncoderEqualsResidentBitForBit() throws {
        Memory.cacheLimit = 256 * 1_048_576
        let patches = MLXRandom.normal([1, 512, 48], key: MLXRandom.key(1)) * 0.1
        let expected = try resident(patches, chunk: nil)
        let (model, stream) = try StableAudio3.loadMediumEncoder(weights)
        #expect(model.blocks.isEmpty)
        let actual = model(patches, stream: stream, chunk: nil)
        #expect(bitwiseEqual(actual, expected))
    }

    @Test func chunkedEncoderMatchesWholeSequence() throws {
        Memory.cacheLimit = 256 * 1_048_576
        let patches = MLXRandom.normal([1, 512, 640], key: MLXRandom.key(2)) * 0.1
        let (model, stream) = try StableAudio3.loadMediumEncoder(weights)
        let expected = model(patches, stream: stream, chunk: nil)
        eval(expected)
        let actual = model(patches, stream: stream, chunk: 16)
        let delta = abs(actual - expected).max().item(Float.self)
        print("SA3 chunked SAME-L encoder max |Δ| \(delta)")
        #expect(delta < 1e-5)
    }
}

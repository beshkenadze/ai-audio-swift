import Foundation
import MLX
import Testing

@testable import MLXAudioMusic

struct StableAudio3ScheduleTests {
    @Test func scheduleMatchesTheReferenceLogSNRShift() {
        let sigmas = StableAudio3.schedule(steps: 8).asArray(Float.self)
        // Printed by the reference sampler for 8 steps.
        let expected: [Float] = [1.000, 0.994, 0.984, 0.958, 0.891, 0.746, 0.512, 0.274, 0.000]
        #expect(sigmas.count == expected.count)
        for (value, reference) in zip(sigmas, expected) {
            #expect(abs(value - reference) < 5e-4)
        }
    }

    @Test func latentLengthRoundsUp() {
        #expect(StableAudio3.latentLength(seconds: 30) == 323)
        #expect(StableAudio3.latentLength(seconds: 120) == 1292)
        #expect(StableAudio3.latentLength(seconds: 0.01) == 1)
    }

    @Test func sampleCountRoundsHalvesToEvenLikeTheReference() {
        #expect(StableAudio3.sampleCount(seconds: 30) == 1_323_000)
        // 2.005 s is 88 420.5 samples; Python's round() keeps the even one.
        #expect(StableAudio3.sampleCount(seconds: 2.005) == 88_420)
    }

    @Test(arguments: [1, 5, 11, 12, 13, 20])
    func decoderAcceptsEveryLatentLength(length: Int) {
        // Untrained weights: only the length handling is under test.
        let decoder = SAMESDecoder()
        let patches = decoder.decode(MLXRandom.normal([1, SAMESDecoder.latentChannels, length]))
        #expect(patches.shape == [1, SAMESDecoder.outputChannels, length * SAMESDecoder.stride])
    }

    @Test func unpatchInterleavesPatchesPerChannel() {
        // Two channels × patch size 2 × length 3; channel c, sample h of patch l = 100c + 10l + h.
        var values: [Float] = []
        for c in 0..<2 { for h in 0..<2 { for l in 0..<3 { values.append(Float(100 * c + 10 * l + h)) } } }
        let audio = StableAudio3.unpatch(MLXArray(values, [1, 4, 3]), patchSize: 2, channels: 2)
        #expect(audio.shape == [1, 2, 6])
        #expect(audio[0, 0].asArray(Float.self) == [0, 1, 10, 11, 20, 21])
        #expect(audio[0, 1].asArray(Float.self) == [100, 101, 110, 111, 120, 121])
    }
}

private func stableAudio3ParityPaths() -> (weights: URL, reference: URL)? {
    let environment = ProcessInfo.processInfo.environment
    guard let weights = environment["MLXAUDIO_SA3_WEIGHTS"],
          let reference = environment["MLXAUDIO_SA3_REFERENCE"] else { return nil }
    return (URL(fileURLWithPath: weights), URL(fileURLWithPath: reference))
}

/// Stage-by-stage comparison with tensors dumped from the reference MLX/Python
/// pipeline. Needs the published weights and a reference archive:
///
///     MLXAUDIO_SA3_WEIGHTS=<dir with *.npz>  MLXAUDIO_SA3_REFERENCE=<reference.npz>
///
/// (prefix both with `TEST_RUNNER_` under xcodebuild).
@Suite(.serialized, .enabled(if: stableAudio3ParityPaths() != nil))
struct StableAudio3ParityTests {

    let weights: URL
    let reference: NPZArchive
    let prompt: String
    let seconds: Double
    let seed: UInt64

    init() throws {
        let paths = try #require(stableAudio3ParityPaths())
        weights = paths.weights
        reference = try NPZArchive(url: paths.reference)
        prompt = String(decoding: try reference.bytes("prompt"), as: UTF8.self)
        let params = try reference.array("params").asArray(Float.self)
        seconds = Double(params[0])
        seed = UInt64(params[1])
    }

    private func ref(_ name: String) throws -> MLXArray { try reference.array(name) }

    private func maxDifference(_ a: MLXArray, _ b: MLXArray) -> Float {
        abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
    }

    /// Within one float16 rounding step of the reference. The reference wheel and
    /// mlx-swift build the same MLX version from different sources, and their
    /// float16 roundings of `normal` and of small float32 matmuls differ by one step.
    private func withinHalfPrecisionStep(_ a: MLXArray, _ b: MLXArray) -> Bool {
        allClose(a.asType(.float32), b.asType(.float32), rtol: 1e-3, atol: 1e-6).item(Bool.self)
    }

    @Test func textEncoderMatches() throws {
        let encoder = try StableAudio3TextEncoder(npz: weights.appendingPathComponent("t5gemma_f16.npz"))
        let (ids, mask) = encoder.tokenize(prompt)
        #expect(ids.asArray(Int32.self) == (try ref("ids")).asArray(Int32.self))
        #expect(mask.asArray(Int32.self) == (try ref("mask")).asArray(Int32.self))
        let (embeddings, _) = encoder.encode(prompt)
        let difference = maxDifference(embeddings, try ref("embeds"))
        print("SA3 parity text encoder: max |Δ| \(difference)")
        #expect(difference < 1e-2)
    }

    @Test func conditioningMatches() throws {
        let conditioning = try StableAudio3.Conditioning(
            npz: weights.appendingPathComponent("dit_sm-music_f16.npz"))
        let (crossAttn, global) = conditioning(
            embeddings: try ref("embeds"), mask: try ref("mask"), seconds: seconds)
        #expect(maxDifference(crossAttn, try ref("cross_attn")) == 0)
        #expect(maxDifference(global, try ref("global_cond")) == 0)
    }

    @Test func noiseAndScheduleMatch() throws {
        let latentLength = StableAudio3.latentLength(seconds: seconds)
        let noise = MLXRandom.normal([1, 256, latentLength], dtype: .float16, key: MLXRandom.key(seed))
        #expect(withinHalfPrecisionStep(noise, try ref("noise")))
        #expect(maxDifference(StableAudio3.schedule(steps: 8), try ref("sigmas")) == 0)
    }

    @Test func transformerAndSamplerMatch() throws {
        let model = try StableAudio3.loadDiT(weights.appendingPathComponent("dit_sm-music_f16.npz"))
        let (noise, crossAttn, global, sigmas) = (try ref("noise"), try ref("cross_attn"),
                                                  try ref("global_cond"), try ref("sigmas"))
        let v0 = model(noise, t: sigmas[0] * MLXArray.ones([1], dtype: .float16),
                       crossAttnCond: crossAttn, globalCond: global)
        let firstStep = maxDifference(v0, try ref("v0"))
        print("SA3 parity DiT first velocity: max |Δ| \(firstStep)")
        #expect(firstStep < 1e-2)

        let latents = StableAudio3.sample(noise: noise, sigmas: sigmas, seed: seed + 1) {
            model($0, t: $1, crossAttnCond: crossAttn, globalCond: global)
        }
        let final = maxDifference(latents, try ref("latents"))
        print("SA3 parity sampled latents: max |Δ| \(final)")
        #expect(final < 5e-2)
    }

    @Test func decoderMatches() throws {
        let decoder = try StableAudio3.loadDecoder(weights.appendingPathComponent("same_s_decoder_f32.npz"))
        let patches = decoder.decode(try ref("latents").asType(.float32))
        let difference = maxDifference(patches, try ref("patches"))
        print("SA3 parity decoder: max |Δ| \(difference)")
        #expect(difference < 1e-3)
    }

    @Test func endToEndAudioMatches() throws {
        let model = StableAudio3(variant: .smallMusic, weightsDirectory: weights)
        let audio = try model.generate(.init(prompt: prompt, seconds: seconds, seed: seed))
        let reference = try ref("audio")[0]
        #expect(audio.shape == reference.shape)
        let difference = maxDifference(audio, reference)
        let rms = sqrt(mean(square(audio - reference))).item(Float.self)
        print("SA3 parity end to end: max |Δ| \(difference), rms Δ \(rms)")
        #expect(rms < 1e-2)
    }
}

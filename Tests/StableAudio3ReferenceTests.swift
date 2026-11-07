import Foundation
import MLX
import Testing

@testable import MLXAudioMusic

/// The reference is pinned at the input of every velocity evaluation, not only the first.
struct StableAudio3ReferenceTests {
    private let channels = 4
    private let length = 12
    private let head = 3

    private func reference() -> StableAudio3.Reference {
        .init(latents: MLXRandom.normal([1, channels, head], key: MLXRandom.key(10)).asType(.float16), noiseSeed: 99)
    }

    private func pin() throws -> StableAudio3.Pin {
        try .init(reference(), channels: channels, sequenceLength: length)
    }

    @Test func pinBlendsFromNoiseToReference() throws {
        let pin = try pin()
        let x = MLXRandom.normal([1, channels, length], key: MLXRandom.key(1)).asType(.float16)

        let noisy = pin.apply(to: x, sigma: MLXArray(Float(1)))
        let clean = pin.apply(to: x, sigma: MLXArray(Float(0)))
        let half = pin.apply(to: x, sigma: MLXArray(Float(0.5)))

        #expect(bitwiseEqual(noisy[0..., 0..., ..<head], pin.noise))
        #expect(bitwiseEqual(clean[0..., 0..., ..<head], pin.latents))
        #expect(allClose(half[0..., 0..., ..<head], 0.5 * pin.noise + 0.5 * pin.latents, atol: 1e-3).item(Bool.self))
        for pinned in [noisy, clean, half] {
            #expect(bitwiseEqual(pinned[0..., 0..., head...], x[0..., 0..., head...]))
        }
    }

    @Test func samplerPinsEveryEvaluationAndEndsOnTheReference() throws {
        let pin = try pin()
        let sigmas = StableAudio3.schedule(steps: 4)
        var seen: [(sigma: Float, head: MLXArray)] = []
        let noise = MLXRandom.normal([1, channels, length], dtype: .float16, key: MLXRandom.key(2))

        let result = StableAudio3.sample(noise: noise, sigmas: sigmas, seed: 3, pin: pin) { x, t in
            seen.append((t[0].item(Float.self), x[0..., 0..., ..<head]))
            return MLXArray.zeros(like: x)
        }

        #expect(seen.count == 4)
        for (sigma, observed) in seen {
            let expected = pin.apply(to: noise, sigma: MLXArray(sigma))[0..., 0..., ..<head]
            #expect(bitwiseEqual(observed, expected))
        }
        #expect(bitwiseEqual(result[0..., 0..., ..<head], pin.latents))
        #expect(!bitwiseEqual(result[0..., 0..., head...], noise[0..., 0..., head...]))
    }

    @Test func withoutAReferenceTheSamplerIsTheOneBeforePinning() {
        let noise = MLXRandom.normal([1, channels, length], dtype: .float16, key: MLXRandom.key(4))
        let sigmas = StableAudio3.schedule(steps: 8)
        let velocity: (MLXArray, MLXArray) -> MLXArray = { x, t in
            x * 0.5 + t.reshaped(-1, 1, 1).asType(x.dtype) * 0.25
        }

        #expect(bitwiseEqual(
            StableAudio3.sample(noise: noise, sigmas: sigmas, seed: 5, velocity: velocity),
            Self.samplerBeforePinning(noise: noise, sigmas: sigmas, seed: 5, velocity: velocity)))
    }

    @Test func referenceKeepsItsPrecision() throws {
        let latents = MLXArray([Float](repeating: 1.0001, count: channels * head)).reshaped(1, channels, head)
        let pin = try StableAudio3.Pin(.init(latents: latents, noiseSeed: 1), channels: channels, sequenceLength: length)
        let x = MLXArray.zeros([1, channels, length], dtype: .float32)

        #expect(bitwiseEqual(pin.apply(to: x, sigma: MLXArray(Float(0)))[0..., 0..., ..<head], latents))
    }

    @Test func referenceMustFitInsideTheSequence() {
        let latents = MLXArray.zeros([1, channels, length], dtype: .float16)
        #expect(throws: StableAudio3Error.self) {
            try StableAudio3.Pin(.init(latents: latents, noiseSeed: 1), channels: channels, sequenceLength: length)
        }
        #expect(throws: StableAudio3Error.self) {
            try StableAudio3.Pin(
                .init(latents: MLXArray.zeros([1, channels + 1, head], dtype: .float16), noiseSeed: 1),
                channels: channels, sequenceLength: length)
        }
    }

    @Test func aRequestWithAnUnusableReferenceFailsBeforeLoadingAnyWeights() throws {
        let engine = StableAudio3(variant: .smallMusic, weightsDirectory: FileManager.default.temporaryDirectory
            .appendingPathComponent("no-weights-\(UUID())", isDirectory: true))
        let tooLong = StableAudio3.Reference(latents: MLXArray.zeros([1, 256, 40], dtype: .float16), noiseSeed: 1)
        #expect(throws: StableAudio3Error.self) {
            try engine.generate(.init(prompt: "x", seconds: 1, seed: 1, reference: tooLong))
        }
        let fits = StableAudio3.Reference(latents: MLXArray.zeros([1, 256, 4], dtype: .float16), noiseSeed: 1)
        #expect(throws: CocoaError.self) {
            try engine.generate(.init(prompt: "x", seconds: 10, seed: 1,
                                      initLatents: URL(fileURLWithPath: "/dev/null"), reference: fits))
        }
    }

    @Test func savedLatentsGiveTheReferenceBeforeTheEndPosition() throws {
        let values = MLXArray((0..<(256 * 40)).map { Float($0 % 977) / 100 }, [1, 256, 40]).asType(.float16)
        let saved = try StableAudio3.LatentAudio(latents: values, variant: .medium, sampleCount: 40 * 4096 - 5)
        let position = Double(StableAudio3.samplesPerLatent) / Double(StableAudio3.sampleRate)

        let tail = try saved.reference(seconds: position * 8, noiseSeed: 7)
        #expect(bitwiseEqual(tail.latents, values[.ellipsis, 32..<40]) && tail.noiseSeed == 7)

        let body = try saved.reference(seconds: position * 8, end: position * 30, noiseSeed: 7)
        #expect(bitwiseEqual(body.latents, values[.ellipsis, 22..<30]))

        let pastTheEnd = try saved.reference(seconds: position * 8, end: position * 99, noiseSeed: 7)
        #expect(bitwiseEqual(pastTheEnd.latents, values[.ellipsis, 32..<40]))

        #expect(throws: StableAudio3Error.self) { try saved.reference(seconds: position * 8, end: position * 5, noiseSeed: 7) }
        #expect(throws: (any Error).self) { try saved.reference(seconds: 0, noiseSeed: 7) }
    }

    @Test func latentsSampledForALongRequestFitTheSavedFormat() throws {
        let seconds = 120.0
        let count = StableAudio3.latentLength(seconds: seconds)
        let sampled = MLXRandom.normal([1, 256, count], dtype: .float32, key: MLXRandom.key(8))
        let bank = try StableAudio3.LatentAudio(latents: sampled.asType(.float16), variant: .medium,
                                                sampleCount: StableAudio3.sampleCount(seconds: seconds))
        #expect(count == 1292 && bank.latents.shape == [1, 256, 1292])
        #expect(try StableAudio3.latentOffset(seconds: 10) == 108)
    }

    /// The loop as it was before the reference existed.
    private static func samplerBeforePinning(
        noise: MLXArray, sigmas: MLXArray, seed: UInt64, velocity: (MLXArray, MLXArray) -> MLXArray
    ) -> MLXArray {
        var x = noise
        var key = MLXRandom.key(seed)
        let steps = sigmas.dim(0) - 1
        for i in 0..<steps {
            let current = sigmas[i]
            let next = sigmas[i + 1]
            let t = current * MLXArray.ones([x.dim(0)], dtype: x.dtype)
            let denoised = x - current.asType(x.dtype) * velocity(x, t)
            if i < steps - 1, next.item(Float.self) > 0 {
                let (nextKey, sub) = MLXRandom.split(key: key)
                key = nextKey
                let fresh = MLXRandom.normal(x.shape, dtype: x.dtype, key: sub)
                x = (1 - next).asType(x.dtype) * denoised + next.asType(x.dtype) * fresh
            } else {
                x = denoised
            }
            eval(x)
        }
        return x
    }
}

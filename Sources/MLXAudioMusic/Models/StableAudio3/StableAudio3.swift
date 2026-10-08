import Foundation
import MLX
import MLXNN

/// Text-to-music with Stable Audio 3, on the MLX weights Stability AI publishes in
/// `stabilityai/stable-audio-3-optimized` (`MLX/*.npz`).
///
/// Generation runs in three stages — text encoder, diffusion transformer, decoder —
/// and each model is loaded for its stage and released after it, so the peak is
/// the largest single model rather than all three.
public struct StableAudio3 {
    public enum Variant: String, CaseIterable, Sendable {
        case smallMusic = "sm-music"

        var ditWeights: String { "dit_\(rawValue)_f16.npz" }
        var decoderWeights: String { "same_s_decoder_f32.npz" }
        static let textEncoderWeights = "t5gemma_f16.npz"

        /// Every file `generate` reads, relative to the weights directory.
        public var requiredFiles: [String] { [Self.textEncoderWeights, ditWeights, decoderWeights] }
    }

    public static let sampleRate = 44_100
    /// Audio samples per latent position: 256-sample patches, 16 per latent.
    public static let samplesPerLatent = 4096

    public struct Request: Sendable {
        public var prompt: String
        public var seconds: Double
        public var seed: UInt64
        public var steps: Int

        public init(prompt: String, seconds: Double, seed: UInt64, steps: Int = 8) {
            self.prompt = prompt
            self.seconds = seconds
            self.seed = seed
            self.steps = steps
        }
    }

    public enum Stage: Sendable {
        case encodingText, sampling(step: Int, of: Int), decoding
    }

    public let variant: Variant
    public let weightsDirectory: URL

    public init(variant: Variant, weightsDirectory: URL) {
        self.variant = variant
        self.weightsDirectory = weightsDirectory
    }

    /// Stereo audio `[2, samples]` at 44.1 kHz, float32, trimmed to `request.seconds`.
    public func generate(
        _ request: Request, progress: (Stage) -> Void = { _ in }
    ) throws -> MLXArray {
        let latentLength = Self.latentLength(seconds: request.seconds)

        progress(.encodingText)
        let (embeddings, mask) = try autoreleasedStage {
            let encoder = try StableAudio3TextEncoder(npz: url(StableAudio3.Variant.textEncoderWeights))
            let (embeddings, mask) = encoder.encode(request.prompt)
            eval(embeddings, mask)
            return (embeddings, mask)
        }

        let latents = try autoreleasedStage {
            let dit = try url(variant.ditWeights)
            let conditioning = try Conditioning(npz: dit)
                .callAsFunction(embeddings: embeddings, mask: mask, seconds: request.seconds)
            let model = try Self.loadDiT(dit)
            let noise = MLXRandom.normal(
                [1, StableAudio3DiTConfig.smallMusic.ioChannels, latentLength],
                dtype: .float16, key: MLXRandom.key(request.seed))
            let latents = Self.sample(
                noise: noise, sigmas: Self.schedule(steps: request.steps), seed: request.seed + 1,
                velocity: { model($0, t: $1, crossAttnCond: conditioning.crossAttn,
                                  globalCond: conditioning.global) },
                onStep: { progress(.sampling(step: $0, of: request.steps)) })
            eval(latents)
            return latents
        }

        progress(.decoding)
        return try autoreleasedStage {
            let decoder = try Self.loadDecoder(url(variant.decoderWeights))
            let patches = decoder.decode(latents.asType(.float32))
            let audio = Self.unpatch(patches)[0, 0..., ..<Self.sampleCount(seconds: request.seconds)]
            eval(audio)
            return audio
        }
    }

    public static func sampleCount(seconds: Double) -> Int {
        // Ties to even, as Python's round().
        Int((seconds * Double(sampleRate)).rounded(.toNearestOrEven))
    }

    public static func latentLength(seconds: Double) -> Int {
        max(1, Int((seconds * Double(sampleRate) / Double(samplesPerLatent)).rounded(.up)))
    }

    // MARK: - Stages

    private func url(_ name: String) throws -> URL {
        let url = weightsDirectory.appendingPathComponent(name)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw NPZArchiveError.missingEntry(url.path)
        }
        return url
    }

    /// Runs a stage and returns its unified-memory cache to the system before the next.
    private func autoreleasedStage<T>(_ body: () throws -> T) rethrows -> T {
        defer { Memory.clearCache() }
        return try body()
    }

    static func loadDiT(_ url: URL) throws -> StableAudio3DiT {
        let model = StableAudio3DiT(.smallMusic)
        let weights = StableAudio3DiT.sanitize(try NPZArchive(url: url).arrays { !$0.hasPrefix("cond.") })
            .mapValues { $0.asType(.float16) }
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        eval(model.parameters())
        return model
    }

    static func loadDecoder(_ url: URL) throws -> SAMESDecoder {
        let model = SAMESDecoder()
        let weights = SAMESDecoder.sanitize(try NPZArchive(url: url).arrays()).mapValues { $0.asType(.float32) }
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        eval(model.parameters())
        return model
    }

    // MARK: - Conditioning

    /// Prompt tokens with padding replaced by a learned embedding, followed by an
    /// embedding of the requested duration, which also serves as the global condition.
    struct Conditioning {
        let paddingEmbedding: MLXArray
        let secondsWeight: MLXArray
        let secondsBias: MLXArray

        init(npz url: URL) throws {
            let archive = try NPZArchive(url: url)
            paddingEmbedding = try archive.array("cond.padding_embedding").asType(.float32)
            secondsWeight = try archive.array("cond.seconds_total_weight").asType(.float32)
            secondsBias = try archive.array("cond.seconds_total_bias").asType(.float32)
        }

        func callAsFunction(
            embeddings: MLXArray, mask: MLXArray, seconds: Double
        ) -> (crossAttn: MLXArray, global: MLXArray) {
            let embeddings = embeddings.asType(.float16)
            let keep = mask.asType(embeddings.dtype)[0..., 0..., .newAxis]
            let padding = paddingEmbedding.asType(embeddings.dtype).reshaped(1, 1, -1)
            let padded = embeddings * keep + padding * (1 - keep)
            let secondsEmbedding = embed(seconds: seconds).asType(.float16)
            let crossAttn = concatenated([padded, secondsEmbedding], axis: 1)
            let global = secondsEmbedding[0..., 0, 0...]
            eval(crossAttn, global)
            return (crossAttn, global)
        }

        /// `NumberConditioner(min 0, max 384)` with exponential Fourier features → `[1, 1, 768]`.
        func embed(seconds: Double) -> MLXArray {
            let (minimum, maximum): (Float, Float) = (0, 384)
            let s = clip(MLXArray([Float(seconds)]), min: minimum, max: maximum)
            let normalised = (s - minimum) / (maximum - minimum)
            let half = 128
            let ramp = MLXArray(0..<half).asType(.float32) / Float(max(half - 1, 1))
            let (low, high) = (Foundation.log(0.5), Foundation.log(10_000.0))
            let frequencies = exp(ramp * Float(high - low) + Float(low))
            let arguments = normalised.reshaped(-1, 1) * frequencies * 2 * nearestPi
            let features = concatenated([cos(arguments), sin(arguments)], axis: -1)
            return (matmul(features, secondsWeight.T) + secondsBias)[0..., .newAxis, 0...]
        }
    }

    // MARK: - Sampling

    /// Linear noise levels from 1 to 0, warped through log-SNR (anchor −6.2, end 2.0),
    /// with the first level pinned to 1.
    static func schedule(steps: Int, sigmaMax: Float = 1) -> MLXArray {
        let t = linspace(sigmaMax, Float(0), count: steps + 1)
        let logSNR = 2.0 - t * Float(2.0 - -6.2)
        var shifted = sigmoid(-logSNR)
        shifted = which(t .<= 0, MLXArray.zeros(like: shifted), shifted)
        shifted = which(t .>= 1, MLXArray.ones(like: shifted), shifted)
        return concatenated([MLXArray([sigmaMax]), shifted[1...]], axis: 0)
    }

    /// Ping-pong sampler for a rectified-flow velocity model: denoise fully, then
    /// re-noise to the next level with fresh noise.
    static func sample(
        noise: MLXArray, sigmas: MLXArray, seed: UInt64,
        velocity: (MLXArray, MLXArray) -> MLXArray,
        onStep: (Int) -> Void = { _ in }
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
            onStep(i + 1)
        }
        return x
    }

    /// `[B, 2 * 256, L]` patches → `[B, 2, 256 L]` samples.
    static func unpatch(_ patches: MLXArray, patchSize: Int = 256, channels: Int = 2) -> MLXArray {
        let (b, length) = (patches.dim(0), patches.dim(2))
        return patches.reshaped(b, channels, patchSize, length)
            .transposed(0, 1, 3, 2)
            .reshaped(b, channels, length * patchSize)
    }
}

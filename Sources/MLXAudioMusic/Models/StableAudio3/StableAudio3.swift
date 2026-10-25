import CryptoKit
import Foundation
import MLX
import MLXNN

/// Text-to-music with Stable Audio 3, on the MLX weights Stability AI publishes in
/// `stabilityai/stable-audio-3-optimized` (`MLX/*.npz`).
///
/// Generation runs in three stages — text encoder, diffusion transformer, decoder —
/// and each model is loaded for its stage and released after it. The transformer and
/// the Medium decoder also stream their blocks from the archive, so a stage holds
/// about one block of weights rather than its whole model.
public struct StableAudio3 {
    public enum Variant: String, CaseIterable, Sendable {
        case smallMusic = "sm-music"
        case medium

        var ditWeights: String { "dit_\(rawValue)_f16.npz" }
        var decoderWeights: String { self == .medium ? "same_l_decoder_f32.npz" : "same_s_decoder_f32.npz" }
        public var encoderWeights: String { self == .medium ? "same_l_encoder_f32.npz" : "same_s_encoder_f32.npz" }
        var ditConfig: StableAudio3DiTConfig { self == .medium ? .medium : .smallMusic }
        static let textEncoderWeights = "t5gemma_f16.npz"

        /// Every file `generate` reads, relative to the weights directory.
        public var requiredFiles: [String] { [Self.textEncoderWeights, ditWeights, decoderWeights] }
    }

    public static let sampleRate = 44_100
    /// Audio samples per latent position: 256-sample patches, 16 per latent.
    public static let samplesPerLatent = 4096

    public enum DiTPrecision: String, Sendable {
        /// Preserve the reference's FP32 promotion through timestep/local conditioning.
        case reference
        /// FP32 Fourier features, then FP16 projections and DiT activations.
        case float16 = "fp16"
    }

    public struct Request: Sendable {
        public var prompt: String
        public var seconds: Double
        public var seed: UInt64
        public var steps: Int
        public var precision: DiTPrecision
        public var initAudio: URL?
        public var initNoiseLevel: Float
        public var initAudioOffset: Double
        public var initLatents: URL?
        public var initLatentOffset: Int

        public init(prompt: String, seconds: Double, seed: UInt64, steps: Int = 8,
                    precision: DiTPrecision = .reference, initAudio: URL? = nil,
                    initNoiseLevel: Float = 1, initAudioOffset: Double = 0,
                    initLatents: URL? = nil, initLatentOffset: Int = 0) {
            self.prompt = prompt
            self.seconds = seconds
            self.seed = seed
            self.steps = steps
            self.precision = precision
            self.initAudio = initAudio
            self.initNoiseLevel = initNoiseLevel
            self.initAudioOffset = initAudioOffset
            self.initLatents = initLatents
            self.initLatentOffset = initLatentOffset
        }
    }

    public enum Stage: Sendable {
        /// `sampling(step: 0, …)` marks the models loaded and the first step starting.
        case encodingText, encodingAudio, sampling(step: Int, of: Int), decoding
    }

    public let variant: Variant
    public let weightsDirectory: URL
    /// Where prompt embeddings are kept between runs; nil encodes every prompt.
    public let textCacheDirectory: URL?

    public init(variant: Variant, weightsDirectory: URL, textCacheDirectory: URL? = nil) {
        self.variant = variant
        self.weightsDirectory = weightsDirectory
        self.textCacheDirectory = textCacheDirectory
    }

    /// Stereo audio `[2, samples]` at 44.1 kHz, float32, trimmed to `request.seconds`.
    ///
    /// `onAudio` receives the audio in order as the decoder produces it, so a caller can
    /// write or play the start before decoding ends; the pieces concatenate to the result.
    public func generate(
        _ request: Request, progress: (Stage) -> Void = { _ in }, onAudio: (MLXArray) -> Void = { _ in }
    ) throws -> MLXArray {
        guard request.seconds.isFinite, request.seconds > 0, request.steps > 0,
              request.initNoiseLevel.isFinite, (0.01...1).contains(request.initNoiseLevel),
              request.initAudioOffset.isFinite, request.initAudioOffset >= 0,
              request.initLatentOffset >= 0, request.initAudio == nil || request.initLatents == nil else {
            throw CocoaError(.validationMissingMandatoryProperty)
        }
        let latentLength = Self.latentLength(seconds: request.seconds)

        progress(.encodingText)
        let (embeddings, mask) = try autoreleasedStage { try encodeText(request.prompt) }

        let initLatents: MLXArray?
        if let file = request.initLatents {
            initLatents = try autoreleasedStage {
                let encoded = try LatentAudio(contentsOf: file)
                    .segment(length: latentLength, offset: request.initLatentOffset, variant: variant)
                eval(encoded)
                return encoded
            }
        } else if let audio = request.initAudio {
            initLatents = try encodeAudioFile(audio, seconds: request.seconds, offset: request.initAudioOffset,
                                              progress: progress).latents
        } else {
            initLatents = nil
        }

        let latents = try autoreleasedStage {
            let dit = try url(variant.ditWeights)
            let model = try Self.loadDiT(dit, config: variant.ditConfig, streamsBlocks: true)
            let conditioning = try Conditioning(npz: dit)
                .callAsFunction(embeddings: embeddings, mask: mask, seconds: request.seconds)
            model.computeInFloat16 = request.precision == .float16
            let pureNoise = MLXRandom.normal(
                [1, variant.ditConfig.ioChannels, latentLength],
                dtype: .float16, key: MLXRandom.key(request.seed))
            let noise = Self.initialNoise(pureNoise, initLatents: initLatents, sigma: request.initNoiseLevel)
            progress(.sampling(step: 0, of: request.steps))
            let latents = Self.sample(
                noise: noise, sigmas: Self.schedule(steps: request.steps, sigmaMax: request.initNoiseLevel), seed: request.seed + 1,
                velocity: { model($0, t: $1, crossAttnCond: conditioning.crossAttn,
                                  globalCond: conditioning.global) },
                onStep: { progress(.sampling(step: $0, of: request.steps)) })
            eval(latents)
            return latents
        }

        progress(.decoding)
        return try autoreleasedStage {
            let weights = try url(variant.decoderWeights)
            let latents = latents.asType(.float32)
            let total = Self.sampleCount(seconds: request.seconds)
            var pieces: [MLXArray] = []
            var written = 0
            func emit(_ patches: MLXArray) {
                let audio = Self.unpatch(patches)[0]
                let count = min(audio.dim(1), total - written)
                guard count > 0 else { return }
                let piece = audio[0..., ..<count]
                eval(piece)
                pieces.append(piece)
                written += count
                onAudio(piece)
            }
            switch variant {
            case .smallMusic: _ = try Self.loadDecoder(weights).decode(latents, onPiece: emit)
            case .medium:
                let (decoder, blocks) = try Self.loadStreamedMediumDecoder(weights)
                _ = decoder.decode(latents, blocks: blocks, onPiece: emit)
            }
            let audio = concatenated(pieces, axis: 1)
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

    /// With `streamsBlocks`, only the weights outside the transformer blocks are loaded
    /// here; the blocks are read from the archive as they run.
    static func loadDiT(
        _ url: URL, config: StableAudio3DiTConfig = .smallMusic, streamsBlocks: Bool = false
    ) throws -> StableAudio3DiT {
        let model = StableAudio3DiT(config, streamsBlocks: streamsBlocks)
        let archive = try NPZArchive(url: url)
        if streamsBlocks {
            model.transformer.blocks = try StableAudio3BlockStream(archive: archive, config: config)
        }
        let weights = StableAudio3Quantization.cast(StableAudio3DiT.sanitize(try archive.arrays {
            !$0.hasPrefix("cond.") && $0 != "QUANTIZATION"
                && !(streamsBlocks && $0.hasPrefix("transformer.layers."))
        }), to: .float16)
        try StableAudio3Quantization.prepare(model, archive: archive, weights: weights)
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        eval(model.parameters())
        return model
    }

    static func loadDecoder(_ url: URL) throws -> SAMESDecoder {
        let model = SAMESDecoder()
        let archive = try NPZArchive(url: url)
        let weights = StableAudio3Quantization.cast(SAMESDecoder.sanitize(
            try archive.arrays { $0 != "QUANTIZATION" }), to: .float32)
        try StableAudio3Quantization.prepare(model, archive: archive, weights: weights)
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        eval(model.parameters())
        return model
    }

    static func loadMediumDecoder(_ url: URL) throws -> SAMELDecoder {
        let model = SAMELDecoder()
        let archive = try NPZArchive(url: url)
        let weights = StableAudio3Quantization.cast(SAMELDecoder.sanitize(
            try archive.arrays { $0 != "QUANTIZATION" }), to: .float32)
        try StableAudio3Quantization.prepare(model, archive: archive, weights: weights)
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        eval(model.parameters())
        return model
    }

    static func loadStreamedMediumDecoder(_ url: URL) throws -> (SAMELDecoder, SAMELBlockStream) {
        let model = SAMELDecoder(streamsBlocks: true)
        let archive = try NPZArchive(url: url)
        let blocks = try SAMELBlockStream(archive: archive)
        let weights = StableAudio3Quantization.cast(SAMELDecoder.sanitize(
            try archive.arrays { $0 != "QUANTIZATION" && !$0.hasPrefix("blocks.") }), to: .float32)
        try StableAudio3Quantization.prepare(model, archive: archive, weights: weights)
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        eval(model.parameters())
        return (model, blocks)
    }

    // MARK: - Text

    /// The encoder's output depends only on the prompt and the encoder weights, and the
    /// app renders the same few prompts again and again: a hit skips loading 0.57 GB.
    private func encodeText(_ prompt: String) throws -> (embeddings: MLXArray, mask: MLXArray) {
        let weights = try url(Variant.textEncoderWeights)
        let cached = try textCacheDirectory.map {
            $0.appendingPathComponent(try Self.textCacheKey(prompt: prompt, weights: weights) + ".safetensors")
        }
        if let cached, let arrays = try? MLX.loadArrays(url: cached),
           let embeddings = arrays["embeddings"], let mask = arrays["mask"],
           embeddings.ndim == 3, embeddings.shape.prefix(2) == [1, StableAudio3TextEncoder.maxLength],
           embeddings.dtype == .float16, mask.shape == [1, StableAudio3TextEncoder.maxLength], mask.dtype == .int32 {
            eval(embeddings, mask)
            return (embeddings, mask)
        }
        let encoder = try StableAudio3TextEncoder(npz: weights)
        let (embeddings, mask) = encoder.encode(prompt)
        eval(embeddings, mask)
        if let cached {
            // Written aside and renamed, so a reader never sees half a file.
            let directory = cached.deletingLastPathComponent()
            let partial = directory.appendingPathComponent("\(UUID().uuidString).safetensors")
            do {
                try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
                try MLX.save(arrays: ["embeddings": embeddings, "mask": mask], url: partial)
                // rename(2) replaces an existing entry in one step; there is no moment without one.
                guard rename(partial.path, cached.path) == 0 else { throw CocoaError(.fileWriteUnknown) }
            } catch {
                try? FileManager.default.removeItem(at: partial)
            }
        }
        return (embeddings, mask)
    }

    /// The prompt and the identity of the encoder file it was encoded with.
    static func textCacheKey(prompt: String, weights: URL) throws -> String {
        let file = weights.resolvingSymlinksInPath()
        let values = try file.resourceValues(forKeys: [.fileSizeKey, .contentModificationDateKey])
        let identity = [
            "stable-audio-3-text-v1", file.path, String(values.fileSize ?? 0),
            String(values.contentModificationDate?.timeIntervalSince1970 ?? 0), prompt,
        ].joined(separator: "\u{0}")
        return SHA256.hash(data: Data(identity.utf8)).map { String(format: "%02x", $0) }.joined()
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

    static func encoderLatentLength(_ length: Int, variant: Variant) -> Int {
        variant == .smallMusic ? (length + 1) / 2 * 2 : length
    }

    static func initialNoise(_ noise: MLXArray, initLatents: MLXArray?, sigma: Float) -> MLXArray {
        guard let initLatents, sigma != 1 else { return noise }
        return initLatents * (1 - sigma) + noise * sigma
    }

    /// Linear noise levels from `sigmaMax` to 0, warped through log-SNR (−6.2 to 2.0),
    /// with the first level pinned to `sigmaMax`, as in the MLX reference.
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

    static func patch(_ audio: MLXArray, patchSize: Int = 256) -> MLXArray {
        let (b, c, samples) = (audio.dim(0), audio.dim(1), audio.dim(2))
        precondition(samples % patchSize == 0)
        let length = samples / patchSize
        return audio.reshaped(b, c, length, patchSize).transposed(0, 1, 3, 2)
            .reshaped(b, c * patchSize, length)
    }
}

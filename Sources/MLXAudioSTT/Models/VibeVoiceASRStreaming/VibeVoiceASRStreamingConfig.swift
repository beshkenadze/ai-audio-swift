//
//  VibeVoiceASRStreamingConfig.swift
//  MLXAudioSTT
//
//  Config structs mirroring `microsoft/VibeVoice-ASR-Streaming-{1.5B,7B}`'s
//  config.json / preprocessor_config.json. Field names/defaults verified
//  directly against the HF checkpoints (see PR description / research notes)
//  and cross-checked against the Python MLX reference in
//  `mlx_audio/stt/models/vibevoice_asr/config.py`.
//

import Foundation
import MLXLMCommon

// MARK: - Acoustic / Semantic Tokenizer Config (shared shape, different vae_dim)

public struct VibeVoiceTokenizerConfig: Codable, Sendable {
    public var causal: Bool
    public var channels: Int
    public var convBias: Bool
    public var convNorm: String
    public var disableLastNorm: Bool
    /// e.g. "3-3-3-3-3-3-8" -> [3,3,3,3,3,3,8]
    public var encoderDepths: String
    public var encoderNFilters: Int
    public var encoderRatios: [Int]
    public var fixStd: Float
    public var layerScaleInitValue: Float
    public var layernorm: String
    public var layernormElementwiseAffine: Bool
    public var layernormEps: Float
    public var mixerLayer: String
    public var padMode: String
    public var stdDistType: String
    public var vaeDim: Int
    public var weightInitValue: Float

    enum CodingKeys: String, CodingKey {
        case causal
        case channels
        case convBias = "conv_bias"
        case convNorm = "conv_norm"
        case disableLastNorm = "disable_last_norm"
        case encoderDepths = "encoder_depths"
        case encoderNFilters = "encoder_n_filters"
        case encoderRatios = "encoder_ratios"
        case fixStd = "fix_std"
        case layerScaleInitValue = "layer_scale_init_value"
        case layernorm
        case layernormElementwiseAffine = "layernorm_elementwise_affine"
        case layernormEps = "layernorm_eps"
        case mixerLayer = "mixer_layer"
        case padMode = "pad_mode"
        case stdDistType = "std_dist_type"
        case vaeDim = "vae_dim"
        case weightInitValue = "weight_init_value"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        causal = try c.decodeIfPresent(Bool.self, forKey: .causal) ?? true
        channels = try c.decodeIfPresent(Int.self, forKey: .channels) ?? 1
        convBias = try c.decodeIfPresent(Bool.self, forKey: .convBias) ?? true
        convNorm = try c.decodeIfPresent(String.self, forKey: .convNorm) ?? "none"
        disableLastNorm = try c.decodeIfPresent(Bool.self, forKey: .disableLastNorm) ?? true
        encoderDepths = try c.decodeIfPresent(String.self, forKey: .encoderDepths) ?? "3-3-3-3-3-3-8"
        encoderNFilters = try c.decodeIfPresent(Int.self, forKey: .encoderNFilters) ?? 32
        encoderRatios = try c.decodeIfPresent([Int].self, forKey: .encoderRatios) ?? [8, 5, 5, 4, 2, 2]
        fixStd = try c.decodeIfPresent(Float.self, forKey: .fixStd) ?? 0.5
        layerScaleInitValue = try c.decodeIfPresent(Float.self, forKey: .layerScaleInitValue) ?? 1e-6
        layernorm = try c.decodeIfPresent(String.self, forKey: .layernorm) ?? "RMSNorm"
        layernormElementwiseAffine = try c.decodeIfPresent(Bool.self, forKey: .layernormElementwiseAffine) ?? true
        layernormEps = try c.decodeIfPresent(Float.self, forKey: .layernormEps) ?? 1e-5
        mixerLayer = try c.decodeIfPresent(String.self, forKey: .mixerLayer) ?? "depthwise_conv"
        padMode = try c.decodeIfPresent(String.self, forKey: .padMode) ?? "constant"
        stdDistType = try c.decodeIfPresent(String.self, forKey: .stdDistType) ?? "gaussian"
        vaeDim = try c.decodeIfPresent(Int.self, forKey: .vaeDim) ?? 64
        weightInitValue = try c.decodeIfPresent(Float.self, forKey: .weightInitValue) ?? 0.01
    }

    /// "3-3-3-3-3-3-8" -> [3, 3, 3, 3, 3, 3, 8]
    public var parsedEncoderDepths: [Int] {
        encoderDepths.split(separator: "-").compactMap { Int($0) }
    }

    /// Product of encoder_ratios; samples-per-latent-frame (3200 for both
    /// acoustic and semantic tokenizers in the released checkpoints).
    public var hopLength: Int {
        encoderRatios.reduce(1, *)
    }
}

// MARK: - Decoder Config (Qwen2)

public struct VibeVoiceDecoderConfig: Codable, Sendable {
    public var attentionDropout: Float
    public var hiddenAct: String
    public var hiddenSize: Int
    public var initializerRange: Float
    public var intermediateSize: Int
    public var maxPositionEmbeddings: Int
    public var maxWindowLayers: Int
    public var modelType: String
    public var numAttentionHeads: Int
    public var numHiddenLayers: Int
    public var numKeyValueHeads: Int
    public var rmsNormEps: Float
    public var ropeTheta: Float
    public var slidingWindow: Int?
    public var tieWordEmbeddings: Bool
    public var useCache: Bool
    public var useSlidingWindow: Bool
    public var vocabSize: Int

    enum CodingKeys: String, CodingKey {
        case attentionDropout = "attention_dropout"
        case hiddenAct = "hidden_act"
        case hiddenSize = "hidden_size"
        case initializerRange = "initializer_range"
        case intermediateSize = "intermediate_size"
        case maxPositionEmbeddings = "max_position_embeddings"
        case maxWindowLayers = "max_window_layers"
        case modelType = "model_type"
        case numAttentionHeads = "num_attention_heads"
        case numHiddenLayers = "num_hidden_layers"
        case numKeyValueHeads = "num_key_value_heads"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case slidingWindow = "sliding_window"
        case tieWordEmbeddings = "tie_word_embeddings"
        case useCache = "use_cache"
        case useSlidingWindow = "use_sliding_window"
        case vocabSize = "vocab_size"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        attentionDropout = try c.decodeIfPresent(Float.self, forKey: .attentionDropout) ?? 0.0
        hiddenAct = try c.decodeIfPresent(String.self, forKey: .hiddenAct) ?? "silu"
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 3584
        initializerRange = try c.decodeIfPresent(Float.self, forKey: .initializerRange) ?? 0.02
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 18944
        maxPositionEmbeddings = try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 131_072
        maxWindowLayers = try c.decodeIfPresent(Int.self, forKey: .maxWindowLayers) ?? 28
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "qwen2"
        numAttentionHeads = try c.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 28
        numHiddenLayers = try c.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 28
        numKeyValueHeads = try c.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? 4
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1_000_000.0
        slidingWindow = try c.decodeIfPresent(Int.self, forKey: .slidingWindow)
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        useCache = try c.decodeIfPresent(Bool.self, forKey: .useCache) ?? true
        useSlidingWindow = try c.decodeIfPresent(Bool.self, forKey: .useSlidingWindow) ?? false
        vocabSize = try c.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 152_064
    }

    public var headDim: Int { hiddenSize / numAttentionHeads }
}

// MARK: - Diffusion Head Config (unused at ASR inference time; decoded only
// so weight-loading can skip its keys without a decode error).

public struct VibeVoiceDiffusionHeadConfig: Codable, Sendable {
    public var hiddenSize: Int
    public var latentSize: Int

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case latentSize = "latent_size"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 3584
        latentSize = try c.decodeIfPresent(Int.self, forKey: .latentSize) ?? 64
    }
}

// MARK: - Top-Level Model Config (config.json root)

public struct VibeVoiceASRStreamingModelConfig: Codable, Sendable {
    public var modelType: String
    public var acousticTokenizerConfig: VibeVoiceTokenizerConfig
    public var semanticTokenizerConfig: VibeVoiceTokenizerConfig
    public var decoderConfig: VibeVoiceDecoderConfig
    public var diffusionHeadConfig: VibeVoiceDiffusionHeadConfig?
    public var acousticVaeDim: Int
    public var semanticVaeDim: Int
    public var useSemanticFeature: Bool
    /// Samples-per-token for the combined acoustic/semantic pipeline (3200
    /// for every released checkpoint so far == tokenizer hopLength).
    public var speechTokCompressRatio: Int
    public var targetSampleRate: Int
    public var perLayerQuantization: BaseConfiguration.PerLayerQuantization?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case acousticTokenizerConfig = "acoustic_tokenizer_config"
        case semanticTokenizerConfig = "semantic_tokenizer_config"
        case decoderConfig = "decoder_config"
        case diffusionHeadConfig = "diffusion_head_config"
        case acousticVaeDim = "acoustic_vae_dim"
        case semanticVaeDim = "semantic_vae_dim"
        case useSemanticFeature = "use_semantic_feature"
        case speechTokCompressRatio = "speech_tok_compress_ratio"
        case targetSampleRate = "target_sample_rate"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "vibevoice"
        acousticTokenizerConfig = try c.decode(
            VibeVoiceTokenizerConfig.self, forKey: .acousticTokenizerConfig)
        semanticTokenizerConfig = try c.decode(
            VibeVoiceTokenizerConfig.self, forKey: .semanticTokenizerConfig)
        decoderConfig = try c.decode(VibeVoiceDecoderConfig.self, forKey: .decoderConfig)
        diffusionHeadConfig = try c.decodeIfPresent(
            VibeVoiceDiffusionHeadConfig.self, forKey: .diffusionHeadConfig)
        acousticVaeDim = try c.decodeIfPresent(Int.self, forKey: .acousticVaeDim) ?? 64
        semanticVaeDim = try c.decodeIfPresent(Int.self, forKey: .semanticVaeDim) ?? 128
        useSemanticFeature = try c.decodeIfPresent(Bool.self, forKey: .useSemanticFeature) ?? true
        speechTokCompressRatio = try c.decodeIfPresent(Int.self, forKey: .speechTokCompressRatio) ?? 3200
        targetSampleRate = try c.decodeIfPresent(Int.self, forKey: .targetSampleRate) ?? 24000

        let baseConfig = try? BaseConfiguration(from: decoder)
        perLayerQuantization = baseConfig?.perLayerQuantization
    }
}

// MARK: - Preprocessor Config (preprocessor_config.json)
//
// Governs the official streaming semantics (verified against
// `vibevoice/modular/modeling_vibevoice_asr.py::streaming_generate`,
// `split_then_encode` path, in the upstream microsoft/VibeVoice repo):
// each streamed window covers `chunk_frames + lookahead_frames` *tokenizer*
// frames of raw audio (1 frame == `speechTokCompressRatio` samples), the
// window is independently re-encoded through the acoustic/semantic
// tokenizers every step (no persistent conv state carried across chunks),
// and only advances by `chunk_frames` each step -- the trailing
// `lookahead_frames` worth of raw audio is read again as context for the
// next window. Only the Qwen2 KV-cache persists across chunks.

public struct VibeVoiceASRPreprocessorConfig: Codable, Sendable {
    public var chunkFrames: Int
    public var lookaheadFrames: Int
    public var targetSampleRate: Int
    public var normalizeAudio: Bool
    public var processorClass: String

    enum CodingKeys: String, CodingKey {
        case chunkFrames = "chunk_frames"
        case lookaheadFrames = "lookahead_frames"
        case targetSampleRate = "target_sample_rate"
        case normalizeAudio = "normalize_audio"
        case processorClass = "processor_class"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        chunkFrames = try c.decodeIfPresent(Int.self, forKey: .chunkFrames) ?? 22
        lookaheadFrames = try c.decodeIfPresent(Int.self, forKey: .lookaheadFrames) ?? 4
        targetSampleRate = try c.decodeIfPresent(Int.self, forKey: .targetSampleRate) ?? 24000
        normalizeAudio = try c.decodeIfPresent(Bool.self, forKey: .normalizeAudio) ?? false
        processorClass = try c.decodeIfPresent(String.self, forKey: .processorClass) ?? "VibeVoiceASRProcessor"
    }

    /// Samples per tokenizer frame, e.g. 3200 for the released checkpoints.
    public func frameSamples(hopLength: Int) -> Int { hopLength }

    /// Chunk advance, in raw-audio samples.
    public func chunkSamples(hopLength: Int) -> Int { chunkFrames * hopLength }

    /// Lookahead tail, in raw-audio samples.
    public func lookaheadSamples(hopLength: Int) -> Int { lookaheadFrames * hopLength }

    /// Full window size (chunk + lookahead), in raw-audio samples.
    public func windowSamples(hopLength: Int) -> Int {
        (chunkFrames + lookaheadFrames) * hopLength
    }
}

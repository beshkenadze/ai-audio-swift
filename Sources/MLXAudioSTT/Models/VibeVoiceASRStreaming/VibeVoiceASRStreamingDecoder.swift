//
//  VibeVoiceASRStreamingDecoder.swift
//  MLXAudioSTT
//
//  Plain Qwen2 decoder (GQA attention + RoPE + SwiGLU MLP + RMSNorm,
//  pre-norm residual blocks) -- the Python reference wraps
//  `mlx_lm.models.qwen2.Qwen2Model` directly rather than defining its own
//  attention/MLP. This is the vanilla Qwen2 architecture (unlike Qwen3, it
//  has NO q_norm/k_norm on the attention projections), mirrored here rather
//  than depending on mlx-lm so weight keys line up 1:1 with the checkpoint.
//

import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN

// MARK: - Attention

final class VibeVoiceDecoderAttention: Module {
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    let rope: RoPE

    init(_ config: VibeVoiceDecoderConfig) {
        self.numHeads = config.numAttentionHeads
        self.numKVHeads = config.numKeyValueHeads
        self.headDim = config.headDim
        self.scale = pow(Float(headDim), -0.5)

        // Qwen2 keeps bias=true on qkv projections (bias=false on o_proj),
        // unlike bias-free Qwen3/Llama-style attention.
        self._qProj.wrappedValue = Linear(config.hiddenSize, numHeads * headDim, bias: true)
        self._kProj.wrappedValue = Linear(config.hiddenSize, numKVHeads * headDim, bias: true)
        self._vProj.wrappedValue = Linear(config.hiddenSize, numKVHeads * headDim, bias: true)
        self._oProj.wrappedValue = Linear(numHeads * headDim, config.hiddenSize, bias: false)

        self.rope = RoPE(dimensions: headDim, traditional: false, base: config.ropeTheta)
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let B = hiddenStates.dim(0)
        let L = hiddenStates.dim(1)

        var queries = qProj(hiddenStates)
        var keys = kProj(hiddenStates)
        var values = vProj(hiddenStates)

        queries = queries.reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)
        keys = keys.reshaped(B, L, numKVHeads, headDim).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, numKVHeads, headDim).transposed(0, 2, 1, 3)

        if let cache {
            queries = rope(queries, offset: cache.offset)
            keys = rope(keys, offset: cache.offset)
        } else {
            queries = rope(queries)
            keys = rope(keys)
        }

        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask
        ).transposed(0, 2, 1, 3).reshaped(B, L, -1)

        return oProj(output)
    }
}

// MARK: - MLP (SwiGLU)

final class VibeVoiceDecoderMLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ config: VibeVoiceDecoderConfig) {
        self._gateProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        self._upProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        self._downProj.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

// MARK: - Decoder Layer

final class VibeVoiceDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: VibeVoiceDecoderAttention
    @ModuleInfo(key: "mlp") var mlp: VibeVoiceDecoderMLP
    @ModuleInfo(key: "input_layernorm") var inputLayernorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayernorm: RMSNorm

    init(_ config: VibeVoiceDecoderConfig) {
        self._selfAttn.wrappedValue = VibeVoiceDecoderAttention(config)
        self._mlp.wrappedValue = VibeVoiceDecoderMLP(config)
        self._inputLayernorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayernorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        var residual = hiddenStates
        var h = inputLayernorm(hiddenStates)
        h = selfAttn(h, mask: mask, cache: cache)
        h = residual + h

        residual = h
        h = postAttentionLayernorm(h)
        h = mlp(h)
        h = residual + h

        return h
    }
}

// MARK: - Qwen2 Text Model (embed_tokens + N decoder layers + final norm)

final class VibeVoiceQwen2Model: Module {
    let config: VibeVoiceDecoderConfig

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [VibeVoiceDecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(_ config: VibeVoiceDecoderConfig) {
        self.config = config
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        self._layers.wrappedValue = (0..<config.numHiddenLayers).map { _ in
            VibeVoiceDecoderLayer(config)
        }
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(
        inputIds: MLXArray? = nil,
        inputsEmbeds: MLXArray? = nil,
        cache: [KVCache]? = nil
    ) -> MLXArray {
        var h: MLXArray
        if let embeds = inputsEmbeds {
            h = embeds
        } else if let ids = inputIds {
            h = embedTokens(ids)
        } else {
            fatalError("Either inputIds or inputsEmbeds must be provided")
        }

        let mask = createAttentionMask(h: h, cache: cache?.first)

        let caches = cache ?? [KVCache?](repeating: nil, count: layers.count)
        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: caches[i])
        }

        return norm(h)
    }
}

// MARK: - LanguageModel wrapper (adds lm_head; mirrors Python `LanguageModel`)

final class VibeVoiceLanguageModel: Module {
    let config: VibeVoiceDecoderConfig

    @ModuleInfo(key: "model") var model: VibeVoiceQwen2Model
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    init(_ config: VibeVoiceDecoderConfig) {
        self.config = config
        self._model.wrappedValue = VibeVoiceQwen2Model(config)
        if config.tieWordEmbeddings {
            self._lmHead.wrappedValue = nil
        } else {
            self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabSize, bias: false)
        }
    }

    func callAsFunction(
        inputIds: MLXArray? = nil,
        inputsEmbeds: MLXArray? = nil,
        cache: [KVCache]? = nil
    ) -> MLXArray {
        let hiddenStates = model(inputIds: inputIds, inputsEmbeds: inputsEmbeds, cache: cache)
        if let lmHead {
            return lmHead(hiddenStates)
        }
        return model.embedTokens.asLinear(hiddenStates)
    }

    func makeCache() -> [KVCache] {
        (0..<config.numHiddenLayers).map { _ in KVCacheSimple() }
    }
}

import Foundation
import MLX
import MLXAudioCore
import MLXNN

/// T5Gemma (google/t5gemma-b-b-ul2) encoder configuration, read from the `META` blob of
/// `t5gemma_f16.npz`.
struct T5GemmaEncoderConfig: Codable, Sendable {
    var hiddenSize = 768
    var numHiddenLayers = 12
    var numAttentionHeads = 12
    var numKeyValueHeads = 12
    var headDim = 64
    var intermediateSize = 2048
    var vocabSize = 256_000
    var rmsNormEps: Float = 1e-6
    var attnLogitSoftcapping: Float? = 50
    var queryPreAttnScalar = 64
    var padTokenId = 0

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case intermediateSize = "intermediate_size"
        case vocabSize = "vocab_size"
        case rmsNormEps = "rms_norm_eps"
        case attnLogitSoftcapping = "attn_logit_softcapping"
        case queryPreAttnScalar = "query_pre_attn_scalar"
        case padTokenId = "pad_token_id"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let d = Self.defaults
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? d.hiddenSize
        numHiddenLayers = try c.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? d.numHiddenLayers
        numAttentionHeads = try c.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? d.numAttentionHeads
        numKeyValueHeads = try c.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? d.numKeyValueHeads
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? d.headDim
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? d.intermediateSize
        vocabSize = try c.decodeIfPresent(Int.self, forKey: .vocabSize) ?? d.vocabSize
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? d.rmsNormEps
        attnLogitSoftcapping = try c.decodeIfPresent(Float.self, forKey: .attnLogitSoftcapping)
        queryPreAttnScalar = try c.decodeIfPresent(Int.self, forKey: .queryPreAttnScalar) ?? d.queryPreAttnScalar
        padTokenId = try c.decodeIfPresent(Int.self, forKey: .padTokenId) ?? d.padTokenId
    }

    private init() {}
    private static let defaults = T5GemmaEncoderConfig()
}

/// Gemma RMSNorm: normalise in float32, scale by `1 + weight`, cast back.
final class T5GemmaRMSNorm: Module {
    let weight: MLXArray
    let eps: Float

    init(dimensions: Int, eps: Float) {
        weight = MLXArray.zeros([dimensions])
        self.eps = eps
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let x32 = x.asType(.float32)
        let normed = x32 * rsqrt((x32 * x32).mean(axis: -1, keepDims: true) + eps)
        return (normed * (1 + weight.asType(.float32))).asType(x.dtype)
    }
}

final class T5GemmaAttention: Module {
    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let scaling: Float
    let softcap: Float?

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    init(_ config: T5GemmaEncoderConfig) {
        heads = config.numAttentionHeads
        kvHeads = config.numKeyValueHeads
        headDim = config.headDim
        scaling = pow(Float(config.queryPreAttnScalar), -0.5)
        softcap = config.attnLogitSoftcapping
        _qProj.wrappedValue = Linear(config.hiddenSize, heads * headDim, bias: false)
        _kProj.wrappedValue = Linear(config.hiddenSize, kvHeads * headDim, bias: false)
        _vProj.wrappedValue = Linear(config.hiddenSize, kvHeads * headDim, bias: false)
        _oProj.wrappedValue = Linear(heads * headDim, config.hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray, mask: MLXArray?) -> MLXArray {
        let (b, s) = (x.dim(0), x.dim(1))
        func split(_ t: MLXArray, _ n: Int) -> MLXArray {
            t.reshaped(b, s, n, headDim).transposed(0, 2, 1, 3)
        }
        var q = split(qProj(x), heads)
        var k = split(kProj(x), kvHeads)
        let v = split(vProj(x), kvHeads)
        q = q * cos.asType(q.dtype) + rotateHalf(q) * sin.asType(q.dtype)
        k = k * cos.asType(k.dtype) + rotateHalf(k) * sin.asType(k.dtype)

        var scores = matmul(q, k.transposed(0, 1, 3, 2)) * scaling
        if let softcap {
            scores = tanh(scores / softcap) * softcap
        }
        if let mask {
            scores = scores + mask
        }
        let probs = softmax(scores.asType(.float32), axis: -1).asType(v.dtype)
        let out = matmul(probs, v).transposed(0, 2, 1, 3).reshaped(b, s, heads * headDim)
        return oProj(out)
    }

    private func rotateHalf(_ x: MLXArray) -> MLXArray {
        let half = x.dim(-1) / 2
        return concatenated([-x[.ellipsis, half...], x[.ellipsis, ..<half]], axis: -1)
    }
}

final class T5GemmaMLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ config: T5GemmaEncoderConfig) {
        _gateProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _upProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _downProj.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(geluApproximate(gateProj(x)) * upProj(x))
    }
}

final class T5GemmaEncoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttention: T5GemmaAttention
    @ModuleInfo(key: "mlp") var mlp: T5GemmaMLP
    @ModuleInfo(key: "pre_self_attn_layernorm") var preAttentionNorm: T5GemmaRMSNorm
    @ModuleInfo(key: "post_self_attn_layernorm") var postAttentionNorm: T5GemmaRMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedForwardNorm: T5GemmaRMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedForwardNorm: T5GemmaRMSNorm

    init(_ config: T5GemmaEncoderConfig) {
        _selfAttention.wrappedValue = T5GemmaAttention(config)
        _mlp.wrappedValue = T5GemmaMLP(config)
        let norm = { T5GemmaRMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps) }
        _preAttentionNorm.wrappedValue = norm()
        _postAttentionNorm.wrappedValue = norm()
        _preFeedForwardNorm.wrappedValue = norm()
        _postFeedForwardNorm.wrappedValue = norm()
    }

    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray, mask: MLXArray?) -> MLXArray {
        var h = postAttentionNorm(selfAttention(preAttentionNorm(x), cos: cos, sin: sin, mask: mask))
        let x = x + h
        h = postFeedForwardNorm(mlp(preFeedForwardNorm(x)))
        return x + h
    }
}

/// The T5Gemma encoder that turns a prompt into Stable Audio 3's cross-attention tokens.
final class T5GemmaEncoder: Module {
    let config: T5GemmaEncoderConfig

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [T5GemmaEncoderLayer]
    @ModuleInfo(key: "norm") var norm: T5GemmaRMSNorm
    @ParameterInfo(key: "rope_inv_freq") var ropeInverseFrequencies: MLXArray

    init(_ config: T5GemmaEncoderConfig) {
        self.config = config
        _embedTokens.wrappedValue = Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        _layers.wrappedValue = (0..<config.numHiddenLayers).map { _ in T5GemmaEncoderLayer(config) }
        _norm.wrappedValue = T5GemmaRMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _ropeInverseFrequencies.wrappedValue = MLXArray.zeros([config.headDim / 2])
    }

    /// - Parameters:
    ///   - ids: `[B, S]` int32 token ids.
    ///   - mask: `[B, S]` int32, 1 for real tokens.
    /// - Returns: `[B, S, hidden]` float16 hidden states.
    func callAsFunction(_ ids: MLXArray, mask: MLXArray) -> MLXArray {
        var x = embedTokens(ids).asType(.float16)
        x = x * MLXArray(Float(config.hiddenSize).squareRoot()).asType(.float16)

        let positions = MLXArray(0..<x.dim(1)).asType(.float32)
        let frequencies = outer(positions, ropeInverseFrequencies.asType(.float32))
        let angles = concatenated([frequencies, frequencies], axis: -1)
        let cos = MLX.cos(angles)[.newAxis, .newAxis]
        let sin = MLX.sin(angles)[.newAxis, .newAxis]

        let keep = mask.asType(.float32)
        let additive = ((1 - keep) * -1e9)[0..., .newAxis, .newAxis, 0...].asType(x.dtype)

        for layer in layers {
            x = layer(x, cos: cos, sin: sin, mask: additive)
        }
        return norm(x)
    }
}

/// Tokenizer and encoder loaded from `t5gemma_f16.npz`, which carries its config and
/// SentencePiece model as byte blobs next to the weights.
public final class StableAudio3TextEncoder {
    public static let maxLength = 256

    let encoder: T5GemmaEncoder
    let tokenizer: SentencePieceTokenizer

    public init(npz url: URL) throws {
        let archive = try NPZArchive(url: url)
        let config = try JSONDecoder().decode(T5GemmaEncoderConfig.self, from: archive.bytes("META"))
        tokenizer = try SentencePieceTokenizer(sentencePieceModelData: archive.bytes("TOKENIZER_MODEL"))
        encoder = T5GemmaEncoder(config)
        let weights = try archive.arrays { $0 != "META" && $0 != "TOKENIZER_MODEL" }
        try encoder.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        eval(encoder.parameters())
    }

    /// Token ids and mask padded to `maxLength`, as `[1, maxLength]` int32 arrays.
    public func tokenize(_ prompt: String) -> (ids: MLXArray, mask: MLXArray) {
        let tokens = Array(tokenizer.encodeWithByteFallback(prompt).prefix(Self.maxLength))
        let pad = encoder.config.padTokenId
        let ids = tokens.map(Int32.init) + Array(repeating: Int32(pad), count: Self.maxLength - tokens.count)
        let mask = Array(repeating: Int32(1), count: tokens.count)
            + Array(repeating: Int32(0), count: Self.maxLength - tokens.count)
        return (MLXArray(ids, [1, Self.maxLength]), MLXArray(mask, [1, Self.maxLength]))
    }

    /// Encoder hidden states `[1, maxLength, 768]` (float16) and the mask they were made with.
    public func encode(_ prompt: String) -> (embeddings: MLXArray, mask: MLXArray) {
        let (ids, mask) = tokenize(prompt)
        // An empty prompt would leave softmax with no visible key; let the forward pass
        // see one position, and report the true all-zero mask so padding replaces it.
        let visible = mask.sum().item(Int32.self) == 0
            ? concatenated([MLXArray([Int32(1)], [1, 1]), mask[0..., 1...]], axis: 1)
            : mask
        return (encoder(ids, mask: visible), mask)
    }
}

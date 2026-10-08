import Foundation
import MLX
import MLXNN

/// Shape of a Stable Audio 3 diffusion transformer.
struct StableAudio3DiTConfig: Sendable {
    var ioChannels = 256
    var embedDimension = 1024
    var depth = 20
    var heads = 16
    var ropeDimensions = 32
    var condTokenDimension = 768
    var globalCondDimension = 768
    var localAddCondDimension = 257
    var memoryTokens = 64
    var feedForwardInner = 4096
    var timestepFeatures = 256
    var normEps: Float = 1e-5
    var qkNormEps: Float = 1e-6
    /// Medium attends twice and subtracts: `SDPA(q, k, v) - SDPA(q', k', v)`.
    var differentialAttention = false

    var headDim: Int { embedDimension / heads }

    static let smallMusic = StableAudio3DiTConfig()
    static let medium = StableAudio3DiTConfig(
        embedDimension: 1536, depth: 24, heads: 24, feedForwardInner: 6144, differentialAttention: true)
}

/// `SDPA(q, k, v)`, or with `diff` the differential form `SDPA(q, k, v) - SDPA(q', k', v)`.
func attend(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray,
            diff: (q: MLXArray, k: MLXArray)?, scale: Float) -> MLXArray {
    let main = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: nil)
    guard let diff else { return main }
    return main - MLXFast.scaledDotProductAttention(
        queries: diff.q, keys: diff.k, values: v, scale: scale, mask: nil)
}

/// `Linear → SiLU → Linear`, stored upstream as a list with the activation at index 1.
final class SiLUProjection: Module {
    @ModuleInfo(key: "input") var input: Linear
    @ModuleInfo(key: "output") var output: Linear

    init(_ inDimension: Int, _ hidden: Int, _ outDimension: Int, bias: Bool) {
        _input.wrappedValue = Linear(inDimension, hidden, bias: bias)
        _output.wrappedValue = Linear(hidden, outDimension, bias: bias)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        output(silu(input(x)))
    }
}

final class StableAudio3SelfAttention: Module {
    let config: StableAudio3DiTConfig

    @ModuleInfo(key: "to_qkv") var toQKV: Linear
    @ModuleInfo(key: "to_out") var toOut: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    init(_ config: StableAudio3DiTConfig) {
        self.config = config
        let d = config.embedDimension
        _toQKV.wrappedValue = Linear(d, (config.differentialAttention ? 5 : 3) * d, bias: false)
        _toOut.wrappedValue = Linear(d, d, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: config.headDim, eps: config.qkNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: config.headDim, eps: config.qkNormEps)
    }

    /// Packed as q, k, v and, for differential attention, q', k'.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (b, t) = (x.dim(0), x.dim(1))
        let parts = toQKV(x).split(parts: config.differentialAttention ? 5 : 3, axis: -1).map {
            $0.reshaped(b, t, config.heads, config.headDim).transposed(0, 2, 1, 3)
        }
        let queries = { rope(self.qNorm($0), self.config.ropeDimensions) }
        let keys = { rope(self.kNorm($0), self.config.ropeDimensions) }
        let out = attend(
            queries(parts[0]), keys(parts[1]), parts[2],
            diff: config.differentialAttention ? (queries(parts[3]), keys(parts[4])) : nil,
            scale: pow(Float(config.headDim), -0.5))
        return toOut(out.transposed(0, 2, 1, 3).reshaped(b, t, config.embedDimension))
    }
}

final class StableAudio3CrossAttention: Module {
    let config: StableAudio3DiTConfig

    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_kv") var toKV: Linear
    @ModuleInfo(key: "to_out") var toOut: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    init(_ config: StableAudio3DiTConfig) {
        self.config = config
        let d = config.embedDimension
        let differential = config.differentialAttention
        _toQ.wrappedValue = Linear(d, (differential ? 2 : 1) * d, bias: false)
        _toKV.wrappedValue = Linear(d, (differential ? 3 : 2) * d, bias: false)
        _toOut.wrappedValue = Linear(d, d, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: config.headDim, eps: config.qkNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: config.headDim, eps: config.qkNormEps)
    }

    /// No rotary embedding. Differential packing: `to_q` → q, q'; `to_kv` → k, k', v.
    func callAsFunction(_ x: MLXArray, context: MLXArray) -> MLXArray {
        let b = x.dim(0)
        func heads(_ y: MLXArray) -> MLXArray {
            y.reshaped(b, y.dim(1), config.heads, config.headDim).transposed(0, 2, 1, 3)
        }
        let differential = config.differentialAttention
        let q = toQ(x).split(parts: differential ? 2 : 1, axis: -1).map { qNorm(heads($0)) }
        let kv = toKV(context).split(parts: differential ? 3 : 2, axis: -1).map(heads)
        let out = attend(
            q[0], kNorm(kv[0]), kv[differential ? 2 : 1],
            diff: differential ? (q[1], kNorm(kv[1])) : nil,
            scale: pow(Float(config.headDim), -0.5))
        return toOut(out.transposed(0, 2, 1, 3).reshaped(b, x.dim(1), config.embedDimension))
    }
}

/// SwiGLU feed-forward: upstream `ff.ff.0.proj` and `ff.ff.2`.
final class StableAudio3FeedForward: Module {
    @ModuleInfo(key: "glu") var glu: Linear
    @ModuleInfo(key: "out") var out: Linear

    init(_ config: StableAudio3DiTConfig) {
        _glu.wrappedValue = Linear(config.embedDimension, 2 * config.feedForwardInner, bias: true)
        _out.wrappedValue = Linear(config.feedForwardInner, config.embedDimension, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let parts = glu(x).split(parts: 2, axis: -1)
        return out(parts[0] * silu(parts[1]))
    }
}

final class StableAudio3DiTBlock: Module {
    @ModuleInfo(key: "pre_norm") var preNorm: RMSNorm
    @ModuleInfo(key: "self_attn") var selfAttention: StableAudio3SelfAttention
    @ModuleInfo(key: "cross_attend_norm") var crossAttendNorm: RMSNorm
    @ModuleInfo(key: "cross_attn") var crossAttention: StableAudio3CrossAttention
    @ModuleInfo(key: "ff_norm") var ffNorm: RMSNorm
    @ModuleInfo(key: "ff") var ff: StableAudio3FeedForward
    @ModuleInfo(key: "to_local_embed") var toLocalEmbed: SiLUProjection
    @ParameterInfo(key: "to_scale_shift_gate") var toScaleShiftGate: MLXArray

    init(_ config: StableAudio3DiTConfig) {
        let d = config.embedDimension
        _preNorm.wrappedValue = RMSNorm(dimensions: d, eps: config.normEps)
        _selfAttention.wrappedValue = StableAudio3SelfAttention(config)
        _crossAttendNorm.wrappedValue = RMSNorm(dimensions: d, eps: config.normEps)
        _crossAttention.wrappedValue = StableAudio3CrossAttention(config)
        _ffNorm.wrappedValue = RMSNorm(dimensions: d, eps: config.normEps)
        _ff.wrappedValue = StableAudio3FeedForward(config)
        _toLocalEmbed.wrappedValue = SiLUProjection(config.localAddCondDimension, d, d, bias: true)
        _toScaleShiftGate.wrappedValue = MLXArray.zeros([6 * d])
    }

    func callAsFunction(
        _ x: MLXArray, context: MLXArray, globalCond: MLXArray, localEmbedding: MLXArray
    ) -> MLXArray {
        let modulation = (toScaleShiftGate + globalCond)[0..., .newAxis, 0...].split(parts: 6, axis: -1)
        let (scaleSelf, shiftSelf, gateSelf) = (modulation[0], modulation[1], modulation[2])
        let (scaleFF, shiftFF, gateFF) = (modulation[3], modulation[4], modulation[5])

        var h = preNorm(x) * (1 + scaleSelf) + shiftSelf
        h = selfAttention(h) * sigmoid(1 - gateSelf)
        var x = h + x

        x = x + crossAttention(crossAttendNorm(x), context: context)
        x = x + localEmbedding

        h = ffNorm(x) * (1 + scaleFF) + shiftFF
        h = ff(h) * sigmoid(1 - gateFF)
        return h + x
    }
}

final class StableAudio3ContinuousTransformer: Module {
    let config: StableAudio3DiTConfig

    @ModuleInfo(key: "project_in") var projectIn: Linear
    @ModuleInfo(key: "project_out") var projectOut: Linear
    @ParameterInfo(key: "memory_tokens") var memoryTokens: MLXArray
    @ModuleInfo(key: "global_cond_embedder") var globalCondEmbedder: SiLUProjection
    @ModuleInfo(key: "layers") var layers: [StableAudio3DiTBlock]

    init(_ config: StableAudio3DiTConfig) {
        self.config = config
        let d = config.embedDimension
        _projectIn.wrappedValue = Linear(config.ioChannels, d, bias: false)
        _projectOut.wrappedValue = Linear(d, config.ioChannels, bias: false)
        _memoryTokens.wrappedValue = MLXArray.zeros([config.memoryTokens, d])
        _globalCondEmbedder.wrappedValue = SiLUProjection(d, d, 6 * d, bias: true)
        _layers.wrappedValue = (0..<config.depth).map { _ in StableAudio3DiTBlock(config) }
    }

    func callAsFunction(
        _ x: MLXArray, context: MLXArray, globalEmbed: MLXArray, localAddCond: MLXArray
    ) -> MLXArray {
        let b = x.dim(0)
        let memory = broadcast(memoryTokens[.newAxis], to: [b, config.memoryTokens, config.embedDimension])
        var x = concatenated([memory, projectIn(x)], axis: 1)
        let g = globalCondEmbedder(globalEmbed)
        for layer in layers {
            let local = layer.toLocalEmbed(localAddCond)
            let padding = MLXArray.zeros([b, config.memoryTokens, config.embedDimension], dtype: local.dtype)
            x = layer(x, context: context, globalCond: g,
                      localEmbedding: concatenated([padding, local], axis: 1))
            // One block's intermediates at a time instead of the whole graph's.
            eval(x)
        }
        return projectOut(x[0..., config.memoryTokens..., 0...])
    }
}

/// Velocity-predicting diffusion transformer of Stable Audio 3.
final class StableAudio3DiT: Module {
    let config: StableAudio3DiTConfig

    @ModuleInfo(key: "preprocess_conv") var preprocessConv: Conv1d
    @ModuleInfo(key: "postprocess_conv") var postprocessConv: Conv1d
    @ModuleInfo(key: "to_cond_embed") var toCondEmbed: SiLUProjection
    @ModuleInfo(key: "to_global_embed") var toGlobalEmbed: SiLUProjection
    @ModuleInfo(key: "to_timestep_embed") var toTimestepEmbed: SiLUProjection
    @ModuleInfo(key: "transformer") var transformer: StableAudio3ContinuousTransformer

    init(_ config: StableAudio3DiTConfig) {
        self.config = config
        let d = config.embedDimension
        _preprocessConv.wrappedValue = Conv1d(
            inputChannels: config.ioChannels, outputChannels: config.ioChannels, kernelSize: 1, bias: false)
        _postprocessConv.wrappedValue = Conv1d(
            inputChannels: config.ioChannels, outputChannels: config.ioChannels, kernelSize: 1, bias: false)
        _toCondEmbed.wrappedValue = SiLUProjection(config.condTokenDimension, d, d, bias: false)
        _toGlobalEmbed.wrappedValue = SiLUProjection(config.globalCondDimension, d, d, bias: false)
        _toTimestepEmbed.wrappedValue = SiLUProjection(config.timestepFeatures, d, d, bias: true)
        _transformer.wrappedValue = StableAudio3ContinuousTransformer(config)
    }

    /// Fourier frequencies of the timestep features, scaled by 2π. Computed rather than
    /// stored: a stored array would be taken for a weight.
    private var timestepFrequencies: MLXArray {
        let ramp = linspace(Float(0), Float(1), count: config.timestepFeatures / 2)
        // Constants in double, rounded once, as the reference computes them.
        let (low, high) = (Foundation.log(0.5), Foundation.log(10_000.0))
        return exp(ramp * Float(high - low) + Float(low)) * 2 * nearestPi
    }

    /// - Parameters:
    ///   - x: noisy latents `[B, ioChannels, T]`.
    ///   - t: noise level `[B]`.
    ///   - crossAttnCond: `[B, tokens, condTokenDimension]`.
    ///   - globalCond: `[B, globalCondDimension]`.
    /// - Returns: velocity `[B, ioChannels, T]`.
    func callAsFunction(
        _ x: MLXArray, t: MLXArray, crossAttnCond: MLXArray, globalCond: MLXArray
    ) -> MLXArray {
        let context = toCondEmbed(crossAttnCond)
        let arguments = t[0..., .newAxis] * timestepFrequencies
        let features = concatenated([cos(arguments), sin(arguments)], axis: -1)
        let globalEmbed = toGlobalEmbed(globalCond) + toTimestepEmbed(features)

        let channelsLast = x.transposed(0, 2, 1)
        let input = preprocessConv(channelsLast) + channelsLast
        // Text-to-audio: no inpainting mask or masked input.
        let local = MLXArray.zeros([x.dim(0), x.dim(-1), config.localAddCondDimension])
        let h = transformer(input, context: context, globalEmbed: globalEmbed, localAddCond: local)
        return (postprocessConv(h) + h).transposed(0, 2, 1)
    }

    /// Maps upstream weight names onto this module tree. Index-addressed lists with
    /// an activation in the middle become named projections; the conditioner lives
    /// in the same archive under `cond.` and is loaded separately.
    static func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var result: [String: MLXArray] = [:]
        for (key, value) in weights {
            if key.hasPrefix("cond.") || key.hasSuffix("rotary_pos_emb.inv_freq") { continue }
            var name = key
            for list in ["to_cond_embed", "to_global_embed", "to_timestep_embed",
                         "global_cond_embedder", "to_local_embed.seq"] {
                name = name.replacingOccurrences(of: "\(list).0.", with: "\(list).input.")
                name = name.replacingOccurrences(of: "\(list).2.", with: "\(list).output.")
            }
            name = name.replacingOccurrences(of: "to_local_embed.seq.", with: "to_local_embed.")
            name = name.replacingOccurrences(of: ".ff.ff.0.proj.", with: ".ff.glu.")
            name = name.replacingOccurrences(of: ".ff.ff.2.", with: ".ff.out.")
            result[name] = value
        }
        return result
    }
}

/// π rounded to the nearest float, as the reference gets it from `math.pi`.
/// `Float.pi` is rounded toward zero, one step lower, which moves Fourier features
/// at arguments of thousands of radians.
let nearestPi = Float(Double.pi)

func rope(_ x: MLXArray, _ dimensions: Int) -> MLXArray {
    MLXFast.RoPE(x, dimensions: dimensions, traditional: false, base: 10_000, scale: 1, offset: 0)
}

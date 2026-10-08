import Foundation
import MLX
import MLXNN

/// Dynamic tanh: `gamma * tanh(alpha * x) + beta`.
final class DyT: Module {
    let alpha: MLXArray
    let gamma: MLXArray
    let beta: MLXArray

    init(dimensions: Int) {
        alpha = MLXArray.ones([1])
        gamma = MLXArray.ones([dimensions])
        beta = MLXArray.zeros([dimensions])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        gamma * tanh(alpha * x) + beta
    }
}

/// `SDPA(q, k, v) - SDPA(q_diff, k_diff, v)`; `to_qkv` packs q, k, v, q_diff, k_diff.
final class SAMEDifferentialAttention: Module {
    static let heads = 12
    static let headDim = 64

    @ModuleInfo(key: "to_qkv") var toQKV: Linear
    @ModuleInfo(key: "to_out") var toOut: Linear
    @ModuleInfo(key: "q_norm") var qNorm: DyT
    @ModuleInfo(key: "k_norm") var kNorm: DyT

    init(dimensions: Int) {
        _toQKV.wrappedValue = Linear(dimensions, 5 * dimensions, bias: false)
        _toOut.wrappedValue = Linear(dimensions, dimensions, bias: false)
        _qNorm.wrappedValue = DyT(dimensions: Self.headDim)
        _kNorm.wrappedValue = DyT(dimensions: Self.headDim)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (b, t, d) = (x.dim(0), x.dim(1), x.dim(2))
        let parts = toQKV(x).split(parts: 5, axis: -1).map {
            $0.reshaped(b, t, Self.heads, Self.headDim).transposed(0, 2, 1, 3)
        }
        let q = rope(qNorm(parts[0]), 32)
        let k = rope(kNorm(parts[1]), 32)
        let qDiff = rope(qNorm(parts[3]), 32)
        let kDiff = rope(kNorm(parts[4]), 32)
        let scale = pow(Float(Self.headDim), -0.5)
        let main = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: parts[2], scale: scale, mask: nil)
        let diff = MLXFast.scaledDotProductAttention(
            queries: qDiff, keys: kDiff, values: parts[2], scale: scale, mask: nil)
        return toOut((main - diff).transposed(0, 2, 1, 3).reshaped(b, t, d))
    }
}

final class SAMEFeedForward: Module {
    @ModuleInfo(key: "glu_proj") var gluProj: Linear
    @ModuleInfo(key: "proj_out") var projOut: Linear

    init(dimensions: Int, inner: Int) {
        _gluProj.wrappedValue = Linear(dimensions, 2 * inner, bias: true)
        _projOut.wrappedValue = Linear(inner, dimensions, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let parts = gluProj(x).split(parts: 2, axis: -1)
        return projOut(parts[0] * silu(parts[1]))
    }
}

final class SAMEBlock: Module {
    @ModuleInfo(key: "pre_norm") var preNorm: DyT
    @ModuleInfo(key: "attn") var attention: SAMEDifferentialAttention
    @ModuleInfo(key: "ff_norm") var ffNorm: DyT
    @ModuleInfo(key: "ff") var ff: SAMEFeedForward

    init(dimensions: Int, feedForwardInner: Int) {
        _preNorm.wrappedValue = DyT(dimensions: dimensions)
        _attention.wrappedValue = SAMEDifferentialAttention(dimensions: dimensions)
        _ffNorm.wrappedValue = DyT(dimensions: dimensions)
        _ff.wrappedValue = SAMEFeedForward(dimensions: dimensions, inner: feedForwardInner)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let x = x + attention(preNorm(x))
        return x + ff(ffNorm(x))
    }
}

/// SAME-S decoder of Stable Audio 3 Small: latents `[B, 256, T]` to audio patches
/// `[B, 512, 16T]`.
///
/// Each latent expands into itself plus 16 learned tokens. The first three blocks
/// attend within windows of 34 positions, the last three within windows shifted by
/// half a window, so information crosses window edges.
final class SAMESDecoder: Module, SAMEDecoding {
    static let latentChannels = 256
    static let dimensions = 768
    static let outputChannels = 512
    static let stride = 16
    static let window = (chunk: 8, overlap: 2)
    private static let subChunk = stride + 1
    private static let attentionWindow = 32 + 32 / stride
    private static let shift = attentionWindow / 2

    @ParameterInfo(key: "running_std") var runningStd: MLXArray
    @ModuleInfo(key: "project_in") var projectIn: Linear
    @ParameterInfo(key: "new_tokens") var newTokens: MLXArray
    @ModuleInfo(key: "blocks") var blocks: [SAMEBlock]
    @ModuleInfo(key: "mapping") var mapping: Conv1d

    override init() {
        let d = Self.dimensions
        _runningStd.wrappedValue = MLXArray.ones([1])
        _projectIn.wrappedValue = Linear(Self.latentChannels, d, bias: true)
        _newTokens.wrappedValue = MLXArray.zeros([1, 1, d])
        _blocks.wrappedValue = (0..<6).map { _ in SAMEBlock(dimensions: d, feedForwardInner: 2304) }
        _mapping.wrappedValue = Conv1d(
            inputChannels: d, outputChannels: Self.outputChannels, kernelSize: 3, padding: 1, bias: true)
    }

    /// Requires an even `T`: windows of 34 must tile `17 T` positions.
    func callAsFunction(_ latents: MLXArray) -> MLXArray {
        let (b, t) = (latents.dim(0), latents.dim(2))
        let d = Self.dimensions
        var x = projectIn((latents * runningStd).transposed(0, 2, 1))

        let learned = broadcast(newTokens[.newAxis], to: [b, t, Self.stride, d])
        x = concatenated([x[0..., 0..., .newAxis, 0...], learned], axis: 2)
        let length = t * Self.subChunk
        x = x.reshaped(b * length / Self.attentionWindow, Self.attentionWindow, d)
        for block in blocks[0..<3] { x = block(x) }
        x = x.reshaped(b, length, d)

        x = concatenated([x[0..., ..<Self.shift, 0...], x, x[0..., (length - Self.shift)..., 0...]], axis: 1)
        x = x.reshaped(b * (length + Self.attentionWindow) / Self.attentionWindow, Self.attentionWindow, d)
        for block in blocks[3...] { x = block(x) }
        x = x.reshaped(b, length + Self.attentionWindow, d)[0..., Self.shift..<(Self.shift + length), 0...]

        x = x.reshaped(b * t, Self.subChunk, d)[0..., 1..., 0...].reshaped(b, t * Self.stride, d)
        return mapping(x).transposed(0, 2, 1)
    }

    /// The decoder ships its output convolution in PyTorch layout `[out, in, k]`.
    static func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var weights = weights
        if let w = weights["mapping.weight"], w.ndim == 3, w.dim(1) == dimensions, w.dim(2) == 3 {
            weights["mapping.weight"] = w.transposed(0, 2, 1)
        }
        return weights
    }
}

import Foundation
import MLX
import MLXNN

/// Differential attention over sliding windows: each group of 17 positions (one
/// latent and its 16 learned tokens) attends to its own group and both neighbours.
final class SAMELDifferentialAttention: Module {
    static let heads = 24
    static let headDim = 64
    static let group = SAMELDecoder.stride + 1
    private static let window = 3 * group

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

    /// Within a window of three groups, query `i` of the middle group sees keys `i ... i + 34`.
    static func band(dtype: DType) -> MLXArray {
        let q = MLXArray(0..<group)[0..., .newAxis]
        let kv = MLXArray(0..<window)[.newAxis, 0...]
        return which(logicalAnd(kv .>= q, kv .<= q + 2 * group), MLXArray(Float(0)), MLXArray(Float(-1e9)))
            .asType(dtype)
    }

    func callAsFunction(_ x: MLXArray, band: MLXArray, offset: Int = 0) -> MLXArray {
        let (b, t, d) = (x.dim(0), x.dim(1), x.dim(2))
        let parts = toQKV(x).split(parts: 5, axis: -1).map {
            $0.reshaped(b, t, Self.heads, Self.headDim).transposed(0, 2, 1, 3)
        }
        let q1 = rope(qNorm(parts[0]), 32, offset: offset)
        let k1 = rope(kNorm(parts[1]), 32, offset: offset)
        let q2 = rope(qNorm(parts[3]), 32, offset: offset)
        let k2 = rope(kNorm(parts[4]), 32, offset: offset)
        let out = t <= Self.group
            ? differential(q1, k1, parts[2], q2, k2, mask: nil)
            : slidingWindow(q1, k1, parts[2], q2, k2, band: band)
        return toOut(out.transposed(0, 2, 1, 3).reshaped(b, t, d))
    }

    /// Both attentions in one call, the second set stacked as extra heads.
    private func differential(
        _ q1: MLXArray, _ k1: MLXArray, _ v: MLXArray, _ q2: MLXArray, _ k2: MLXArray, mask: MLXArray?
    ) -> MLXArray {
        let out = MLXFast.scaledDotProductAttention(
            queries: concatenated([q1, q2], axis: 1), keys: concatenated([k1, k2], axis: 1),
            values: concatenated([v, v], axis: 1), scale: pow(Float(Self.headDim), -0.5), mask: mask)
        let halves = out.split(parts: 2, axis: 1)
        return halves[0] - halves[1]
    }

    private func slidingWindow(
        _ q1: MLXArray, _ k1: MLXArray, _ v: MLXArray, _ q2: MLXArray, _ k2: MLXArray, band: MLXArray
    ) -> MLXArray {
        let (b, h, t, d) = (q1.dim(0), q1.dim(1), q1.dim(2), q1.dim(3))
        let (g, w, groups) = (Self.group, Self.window, t / Self.group)
        let length = t + 2 * g

        // Each group's window: three consecutive groups of the sequence padded by one group per side.
        func windows(_ y: MLXArray) -> MLXArray {
            let y = padded(y, widths: [0, 0, IntOrPair([g, g]), 0])
            return asStrided(y, [b, h, groups, w, d], strides: [h * length * d, length * d, g * d, d, 1])
                .transposed(0, 2, 1, 3, 4).reshaped(b * groups, h, w, d)
        }
        func grouped(_ y: MLXArray) -> MLXArray {
            y.reshaped(b, h, groups, g, d).transposed(0, 2, 1, 3, 4).reshaped(b * groups, h, g, d)
        }

        // Padding outside the sequence is masked off, on top of the band.
        let position = MLXArray(0..<groups)[0..., .newAxis] * g + MLXArray(0..<w)[.newAxis, 0...]
        let inside = logicalAnd(position .>= g, position .< t + g)
        let boundary = which(inside, MLXArray(Float(0)), MLXArray(Float(-1e9))).asType(q1.dtype)
        let mask = broadcast((band + boundary[0..., .newAxis, 0...])[.newAxis], to: [b, groups, g, w])
            .reshaped(b * groups, 1, g, w)

        let out = differential(grouped(q1), windows(k1), windows(v), grouped(q2), windows(k2), mask: mask)
        return out.reshaped(b, groups, h, g, d).transposed(0, 2, 1, 3, 4).reshaped(b, h, t, d)
    }
}

final class SAMELFeedForward: Module {
    /// Not a weight: a streamed block module is reused for both kinds of block.
    var sineGate: Bool

    @ModuleInfo(key: "glu_proj") var gluProj: Linear
    @ModuleInfo(key: "proj_out") var projOut: Linear

    init(dimensions: Int, inner: Int, sineGate: Bool) {
        self.sineGate = sineGate
        _gluProj.wrappedValue = Linear(dimensions, 2 * inner, bias: true)
        _projOut.wrappedValue = Linear(inner, dimensions, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let parts = gluProj(x).split(parts: 2, axis: -1)
        let gate = sineGate ? sin(parts[1] * nearestPi) : silu(parts[1])
        return projOut(parts[0] * gate)
    }
}

final class SAMELBlock: Module {
    @ModuleInfo(key: "pre_norm") var preNorm: DyT
    @ModuleInfo(key: "attn") var attention: SAMELDifferentialAttention
    @ModuleInfo(key: "ff_norm") var ffNorm: DyT
    @ModuleInfo(key: "ff") var ff: SAMELFeedForward

    init(dimensions: Int, sineGate: Bool) {
        _preNorm.wrappedValue = DyT(dimensions: dimensions)
        _attention.wrappedValue = SAMELDifferentialAttention(dimensions: dimensions)
        _ffNorm.wrappedValue = DyT(dimensions: dimensions)
        _ff.wrappedValue = SAMELFeedForward(dimensions: dimensions, inner: 4608, sineGate: sineGate)
    }

    func callAsFunction(_ x: MLXArray, band: MLXArray, offset: Int = 0) -> MLXArray {
        let x = x + attention(preNorm(x), band: band, offset: offset)
        // Frees the attention's windowed copies before the feed-forward allocates.
        eval(x)
        return x + ff(ffNorm(x))
    }
}

/// SAME-L decoder of Stable Audio 3 Medium: latents `[B, 256, T]` to audio patches
/// `[B, 512, 16T]`. Twelve blocks of sliding-window differential attention; blocks
/// from the sixth on gate their feed-forward with `sin(πx)`.
final class SAMELDecoder: Module, SAMEDecoding {
    static let latentChannels = 256
    static let dimensions = 1536
    static let outputChannels = 512
    static let stride = 16
    static let window = (chunk: 128, overlap: 8)

    @ParameterInfo(key: "running_std") var runningStd: MLXArray
    @ModuleInfo(key: "project_in") var projectIn: Linear
    @ParameterInfo(key: "new_tokens") var newTokens: MLXArray
    @ModuleInfo(key: "blocks") var blocks: [SAMELBlock]
    @ModuleInfo(key: "mapping") var mapping: Linear

    static let depth = 12
    static func sineGate(block: Int) -> Bool { block >= 5 }

    /// `streamsBlocks` leaves `blocks` empty; decode with ``decode(_:blocks:onPiece:)``.
    init(streamsBlocks: Bool = false) {
        let d = Self.dimensions
        _runningStd.wrappedValue = MLXArray.ones([1])
        _projectIn.wrappedValue = Linear(Self.latentChannels, d, bias: true)
        _newTokens.wrappedValue = MLXArray.zeros([1, 1, d])
        _blocks.wrappedValue = streamsBlocks
            ? [] : (0..<Self.depth).map { SAMELBlock(dimensions: d, sineGate: Self.sineGate(block: $0)) }
        _mapping.wrappedValue = Linear(d, Self.outputChannels, bias: true)
    }

    func callAsFunction(_ latents: MLXArray) -> MLXArray {
        precondition(blocks.count == Self.depth, "a streaming decoder decodes with its block stream")
        var x = embed(latents)
        let band = SAMELDifferentialAttention.band(dtype: x.dtype)
        for block in blocks {
            x = block(x, band: band)
            eval(x)
        }
        return project(x)
    }

    /// Each latent, then its 16 learned tokens: `[B, 17 T, d]`.
    private func embed(_ latents: MLXArray) -> MLXArray {
        let (b, t) = (latents.dim(0), latents.dim(2))
        let d = Self.dimensions
        let x = projectIn((latents * runningStd).transposed(0, 2, 1))
        let learned = broadcast(newTokens[.newAxis], to: [b, t, Self.stride, d])
        return concatenated([x[0..., 0..., .newAxis, 0...], learned], axis: 2)
            .reshaped(b, t * (Self.stride + 1), d)
    }

    private func project(_ x: MLXArray) -> MLXArray {
        let (b, d) = (x.dim(0), Self.dimensions)
        let t = x.dim(1) / (Self.stride + 1)
        let x = x.reshaped(b, t, Self.stride + 1, d)[0..., 0..., 1..., 0...].reshaped(b, t * Self.stride, d)
        return mapping(x).transposed(0, 2, 1)
    }

    /// Decodes the same windows as ``decode(_:onPiece:)``, with the blocks read one at a
    /// time. The first window goes through every block on its own, so its audio is out
    /// as early as before; the rest then go through each block together, so each block
    /// is read twice per track instead of once per window.
    func decode(_ latents: MLXArray, blocks: SAMELBlockStream, onPiece: (MLXArray) -> Void = { _ in })
        -> MLXArray
    {
        let windows = Self.windows(latents)
        var pieces: [MLXArray] = []
        func finish(_ x: MLXArray, _ window: SAMEWindow) {
            let piece = project(x)[.ellipsis, window.keep]
            eval(piece)
            pieces.append(piece)
            onPiece(piece)
        }
        var first = embed(windows[0].latents)
        let band = SAMELDifferentialAttention.band(dtype: first.dtype)
        for index in 0..<Self.depth {
            first = blocks.load(index)(first, band: band)
            eval(first)
        }
        finish(first, windows[0])

        var rest = windows.dropFirst().map { embed($0.latents) }
        if !rest.isEmpty {
            for index in 0..<Self.depth {
                let block = blocks.load(index)
                for i in rest.indices {
                    rest[i] = block(rest[i], band: band)
                    eval(rest[i])
                }
            }
        }
        for (x, window) in zip(rest, windows.dropFirst()) { finish(x, window) }
        return pieces.count == 1 ? pieces[0] : concatenated(pieces, axis: -1)
    }

    /// The output projection ships as a width-1 PyTorch convolution `[out, in, 1]`.
    static func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var weights = weights
        if let w = weights["mapping.weight"], w.ndim == 3, w.dim(2) == 1 {
            weights["mapping.weight"] = w.reshaped(w.dim(0), w.dim(1))
        }
        return weights
    }
}

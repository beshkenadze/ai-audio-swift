import Foundation
import MLX
import MLXNN

final class SAMESEncoder: Module {
    @ModuleInfo(key: "mapping") var mapping: Linear
    @ParameterInfo(key: "new_tokens") var newTokens: MLXArray
    @ModuleInfo(key: "blocks") var blocks: [SAMEBlock]
    @ModuleInfo(key: "project_out") var projectOut: Linear
    @ParameterInfo(key: "scaling_factor") var scalingFactor: MLXArray
    let bias: MLXArray
    @ParameterInfo(key: "running_std") var runningStd: MLXArray

    override init() {
        _mapping.wrappedValue = Linear(512, 768)
        _newTokens.wrappedValue = MLXArray.zeros([1, 1, 768])
        _blocks.wrappedValue = (0..<6).map { _ in SAMEBlock(dimensions: 768, feedForwardInner: 2304) }
        _projectOut.wrappedValue = Linear(768, 256)
        _scalingFactor.wrappedValue = MLXArray.ones([1, 256, 1])
        bias = MLXArray.zeros([1, 256, 1])
        _runningStd.wrappedValue = MLXArray.ones([1])
    }

    func callAsFunction(_ patches: MLXArray) -> MLXArray {
        precondition(patches.dim(1) == 512 && patches.dim(2) % 32 == 0)
        let (b, t, d) = (patches.dim(0), patches.dim(2) / 16, 768)
        let x = mapping(patches.transposed(0, 2, 1)).reshaped(b * t, 16, d)
        let learned = broadcast(newTokens, to: [b * t, 1, d])
        let tokens = concatenated([x, learned], axis: 1).reshaped(b, t * 17, d)
        let encoded = sameSWindowedBlocks(tokens, blocks: blocks)
            .reshaped(b * t, 17, d)[0..., 16, 0...].reshaped(b, t, d)
        return (projectOut(encoded).transposed(0, 2, 1) * scalingFactor + bias) / runningStd
    }

    static func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var weights = weights
        if let mapping = weights["mapping.weight"], mapping.ndim == 3 {
            weights["mapping.weight"] = mapping.reshaped(mapping.dim(0), -1)
        }
        return weights
    }
}

/// SAME-S's shared unshifted/shifted window pass, in encoder and decoder order.
func sameSWindowedBlocks(_ tokens: MLXArray, blocks: [SAMEBlock]) -> MLXArray {
    let (b, length, d) = (tokens.dim(0), tokens.dim(1), tokens.dim(2))
    var x = tokens.reshaped(b * length / 34, 34, d)
    for block in blocks[0..<3] { x = block(x) }
    x = x.reshaped(b, length, d)
    x = concatenated([x[0..., ..<17, 0...], x, x[0..., (length - 17)..., 0...]], axis: 1)
    x = x.reshaped(b * (length + 34) / 34, 34, d)
    for block in blocks[3...] { x = block(x) }
    return x.reshaped(b, length + 34, d)[0..., 17..<(17 + length), 0...]
}

final class SAMELEncoder: Module {
    @ModuleInfo(key: "mapping") var mapping: Linear
    @ParameterInfo(key: "new_tokens") var newTokens: MLXArray
    @ModuleInfo(key: "blocks") var blocks: [SAMELBlock]
    @ModuleInfo(key: "project_out") var projectOut: Linear
    @ParameterInfo(key: "scaling_factor") var scalingFactor: MLXArray
    let bias: MLXArray
    @ParameterInfo(key: "running_std") var runningStd: MLXArray

    init(streamsBlocks: Bool = false) {
        _mapping.wrappedValue = Linear(512, 1536)
        _newTokens.wrappedValue = MLXArray.zeros([1, 1, 1536])
        _blocks.wrappedValue = streamsBlocks ? [] : (0..<12).map { _ in SAMELBlock(dimensions: 1536, sineGate: false) }
        _projectOut.wrappedValue = Linear(1536, 256)
        _scalingFactor.wrappedValue = MLXArray.ones([1, 256, 1])
        bias = MLXArray.zeros([1, 256, 1])
        _runningStd.wrappedValue = MLXArray.ones([1])
    }

    func callAsFunction(_ patches: MLXArray, stream: SAMELBlockStream? = nil, chunk: Int? = 128) -> MLXArray {
        precondition(patches.dim(1) == 512 && patches.dim(2) % 16 == 0)
        precondition(stream != nil || blocks.count == 12)
        let (b, t, d) = (patches.dim(0), patches.dim(2) / 16, 1536)
        let chunk = chunk ?? t
        precondition(chunk > 0)
        // Twelve local-attention blocks reach at most twelve groups either side.
        let windows = stride(from: 0, to: t, by: chunk).map { start in
            (start: start, end: min(start + chunk, t), left: max(0, start - 16), right: min(t, start + chunk + 16))
        }
        var hidden = windows.map { window in
            let count = window.right - window.left
            let x = mapping(patches[.ellipsis, (window.left * 16)..<(window.right * 16)]
                .transposed(0, 2, 1)).reshaped(b * count, 16, d)
            let tokens = concatenated([x, broadcast(newTokens, to: [b * count, 1, d])], axis: 1)
                .reshaped(b, count * 17, d)
            eval(tokens)
            return tokens
        }
        let band = SAMELDifferentialAttention.band(dtype: .float32)
        for index in 0..<12 {
            let block = stream?.load(index) ?? blocks[index]
            for i in hidden.indices {
                hidden[i] = block(hidden[i], band: band, offset: windows[i].left * 17)
                eval(hidden[i])
            }
        }
        let pieces = zip(hidden, windows).map { x, window in
            let encoded = x.reshaped(b, window.right - window.left, 17, d)[
                0..., (window.start - window.left)..<(window.end - window.left), 16, 0...]
            let latents = (projectOut(encoded).transposed(0, 2, 1) * scalingFactor + bias) / runningStd
            eval(latents)
            return latents
        }
        return concatenated(pieces, axis: -1)
    }
}

extension StableAudio3 {
    static func loadSmallEncoder(_ url: URL) throws -> SAMESEncoder {
        let model = SAMESEncoder()
        let weights = SAMESEncoder.sanitize(try NPZArchive(url: url).arrays())
            .mapValues { $0.asType(.float32) }
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        eval(model.parameters())
        return model
    }

    static func loadMediumEncoder(_ url: URL, streamsBlocks: Bool = true) throws -> (SAMELEncoder, SAMELBlockStream?) {
        let model = SAMELEncoder(streamsBlocks: streamsBlocks)
        let archive = try NPZArchive(url: url)
        let stream = streamsBlocks ? try SAMELBlockStream(archive: archive, sineGate: { _ in false }) : nil
        let weights = SAMESEncoder.sanitize(try archive.arrays { !streamsBlocks || !$0.hasPrefix("blocks.") })
            .mapValues { $0.asType(.float32) }
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        eval(model.parameters())
        return (model, stream)
    }

    static func encodeAudio(_ patches: MLXArray, weights: URL, variant: Variant) throws -> MLXArray {
        let result: MLXArray
        switch variant {
        case .smallMusic: result = try loadSmallEncoder(weights)(patches)
        case .medium:
            let (encoder, blocks) = try loadMediumEncoder(weights)
            result = encoder(patches, stream: blocks)
        }
        eval(result)
        return result
    }
}

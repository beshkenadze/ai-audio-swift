import Foundation
import MLX

extension StableAudio3 {
    /// Latents of earlier audio held at the start of the sequence while the rest is
    /// generated, so the new part continues them. Training-free inpainting: before every
    /// velocity evaluation the head is rewritten to `noise * t + latents * (1 - t)`.
    public struct Reference: @unchecked Sendable {
        /// `[1, channels, length]`, as the sampler returns them.
        public var latents: MLXArray
        /// Seeds the noise the head starts from, apart from the request's own seed.
        public var noiseSeed: UInt64

        public init(latents: MLXArray, noiseSeed: UInt64) {
            self.latents = latents
            self.noiseSeed = noiseSeed
        }
    }

    /// The reference at the head of the sequence, at whatever noise level the sampler is at.
    struct Pin {
        let latents: MLXArray
        let noise: MLXArray

        init(_ reference: Reference, channels: Int, sequenceLength: Int) throws {
            let shape = reference.latents.shape
            guard shape.count == 3, shape[0] == 1, shape[1] == channels,
                  shape[2] > 0, shape[2] < sequenceLength else {
                throw StableAudio3Error.invalidReference(shape: shape, channels: channels, sequenceLength: sequenceLength)
            }
            latents = reference.latents
            noise = MLXRandom.normal(shape, dtype: .float16, key: MLXRandom.key(reference.noiseSeed))
        }

        var length: Int { latents.dim(2) }

        /// `sigma` 1 is pure noise, 0 the clean reference.
        func apply(to x: MLXArray, sigma: MLXArray) -> MLXArray {
            let level = sigma.asType(x.dtype)
            let head = level * noise.asType(x.dtype) + (1 - level) * latents.asType(x.dtype)
            return concatenated([head, x[0..., 0..., length...]], axis: 2)
        }
    }
}

extension StableAudio3.LatentAudio {
    /// The `seconds` of these latents that end at `end` seconds (their own end by default).
    /// Both snap to the nearest latent position, so the reference ends on a position boundary.
    public func reference(seconds: Double, end: Double? = nil, noiseSeed: UInt64) throws -> StableAudio3.Reference {
        let length = try StableAudio3.latentOffset(seconds: seconds)
        let stop = try end.map { min(latents.dim(2), try StableAudio3.latentOffset(seconds: $0)) } ?? latents.dim(2)
        guard length > 0, stop >= length else {
            throw StableAudio3Error.referenceOutOfRange(seconds: seconds, end: stop, available: latents.dim(2))
        }
        return .init(latents: try segment(length: length, offset: stop - length, variant: variant), noiseSeed: noiseSeed)
    }
}

public enum StableAudio3Error: LocalizedError {
    case invalidReference(shape: [Int], channels: Int, sequenceLength: Int)
    case referenceOutOfRange(seconds: Double, end: Int, available: Int)

    public var errorDescription: String? {
        switch self {
        case .invalidReference(let shape, let channels, let sequenceLength):
            "Reference latents \(shape) must be [1, \(channels), n] with 0 < n < \(sequenceLength) positions of the sequence"
        case .referenceOutOfRange(let seconds, let end, let available):
            "A \(seconds) s reference does not fit before latent \(end) of \(available)"
        }
    }
}

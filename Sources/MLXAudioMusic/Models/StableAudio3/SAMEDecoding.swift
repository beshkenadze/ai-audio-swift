import Foundation
import MLX
import MLXNN

/// What the SAME-S and SAME-L decoders share: latents `[B, 256, T]` in, audio patches
/// `[B, 512, stride * T]` out, and the way the reference pipeline feeds them latents of
/// any length.
protocol SAMEDecoding: Module {
    /// Audio patches per latent.
    static var stride: Int { get }
    /// The reference pipeline's window: latents kept per call, and context on each side.
    static var window: (chunk: Int, overlap: Int) { get }

    func callAsFunction(_ latents: MLXArray) -> MLXArray
}

extension SAMEDecoding {
    /// Dispatched as the reference pipeline does. SAME-S only accepts even lengths, and
    /// both decoders go through the same rules so their output matches the reference.
    func decode(_ latents: MLXArray) -> MLXArray {
        let t = latents.dim(2)
        let (chunk, overlap) = Self.window
        if t > chunk + 2 * overlap { return decodeChunked(latents, chunk: chunk, overlap: overlap) }
        if t % 2 == 0 { return self(latents) }
        if t > 6 { return decodeChunked(latents, chunk: 2, overlap: 2) }
        // Too short and odd for any even window: repeat the last latent, then trim.
        let even = concatenated([latents, latents[.ellipsis, (t - 1)...]], axis: -1)
        return self(even)[.ellipsis, ..<(t * Self.stride)]
    }

    /// Decodes window by window, so memory stays flat with length.
    ///
    /// Every call sees `chunk + 2 * overlap` real latents and keeps the middle
    /// `chunk`; the first and last calls keep their outer edge as well.
    func decodeChunked(_ latents: MLXArray, chunk: Int, overlap: Int) -> MLXArray {
        let t = latents.dim(2)
        let kernel = chunk + 2 * overlap
        precondition(kernel % 2 == 0 && t > kernel, "windows must be even and shorter than the input")
        let s = Self.stride
        var pieces = [self(latents[.ellipsis, 0..<kernel])[.ellipsis, ..<((chunk + overlap) * s)]]
        var i = chunk + overlap
        while i + chunk + overlap <= t {
            let out = self(latents[.ellipsis, (i - overlap)..<(i + chunk + overlap)])
            pieces.append(out[.ellipsis, (overlap * s)..<((overlap + chunk) * s)])
            eval(pieces.last!)
            i += chunk
        }
        let remaining = t - i
        if remaining > 0 {
            pieces.append(self(latents[.ellipsis, (t - kernel)..<t])[.ellipsis, ((kernel - remaining) * s)...])
        }
        return concatenated(pieces, axis: -1)
    }
}

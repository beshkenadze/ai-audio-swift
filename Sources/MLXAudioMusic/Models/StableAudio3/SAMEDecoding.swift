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
    ///
    /// `onPiece` receives the output in order as each window is decoded; the pieces
    /// concatenate to the returned patches.
    func decode(_ latents: MLXArray, onPiece: (MLXArray) -> Void = { _ in }) -> MLXArray {
        let t = latents.dim(2)
        let (chunk, overlap) = Self.window
        if t > chunk + 2 * overlap {
            return decodeChunked(latents, chunk: chunk, overlap: overlap, onPiece: onPiece)
        }
        if t % 2 != 0, t > 6 { return decodeChunked(latents, chunk: 2, overlap: 2, onPiece: onPiece) }
        let whole: MLXArray
        if t % 2 == 0 {
            whole = self(latents)
        } else {
            // Too short and odd for any even window: repeat the last latent, then trim.
            let even = concatenated([latents, latents[.ellipsis, (t - 1)...]], axis: -1)
            whole = self(even)[.ellipsis, ..<(t * Self.stride)]
        }
        onPiece(whole)
        return whole
    }

    /// Decodes window by window, so memory stays flat with length.
    ///
    /// Every call sees `chunk + 2 * overlap` real latents and keeps the middle
    /// `chunk`; the first and last calls keep their outer edge as well.
    func decodeChunked(
        _ latents: MLXArray, chunk: Int, overlap: Int, onPiece: (MLXArray) -> Void = { _ in }
    ) -> MLXArray {
        let t = latents.dim(2)
        let kernel = chunk + 2 * overlap
        precondition(kernel % 2 == 0 && t > kernel, "windows must be even and shorter than the input")
        let s = Self.stride
        var pieces: [MLXArray] = []
        func keep(_ piece: MLXArray) {
            eval(piece)
            pieces.append(piece)
            onPiece(piece)
        }
        keep(self(latents[.ellipsis, 0..<kernel])[.ellipsis, ..<((chunk + overlap) * s)])
        var i = chunk + overlap
        while i + chunk + overlap <= t {
            let out = self(latents[.ellipsis, (i - overlap)..<(i + chunk + overlap)])
            keep(out[.ellipsis, (overlap * s)..<((overlap + chunk) * s)])
            i += chunk
        }
        let remaining = t - i
        if remaining > 0 {
            keep(self(latents[.ellipsis, (t - kernel)..<t])[.ellipsis, ((kernel - remaining) * s)...])
        }
        return concatenated(pieces, axis: -1)
    }
}

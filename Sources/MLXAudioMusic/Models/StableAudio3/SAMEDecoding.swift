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

/// One decoder call: the latents it sees and the patches of its output that are kept.
struct SAMEWindow {
    let latents: MLXArray
    let keep: Range<Int>
}

extension SAMEDecoding {
    /// Dispatched as the reference pipeline does. SAME-S only accepts even lengths, and
    /// both decoders go through the same rules so their output matches the reference.
    ///
    /// `onPiece` receives the output in order as each window is decoded; the pieces
    /// concatenate to the returned patches.
    func decode(_ latents: MLXArray, onPiece: (MLXArray) -> Void = { _ in }) -> MLXArray {
        run(Self.windows(latents), onPiece: onPiece)
    }

    /// Decodes window by window, so memory stays flat with length.
    func decodeChunked(
        _ latents: MLXArray, chunk: Int, overlap: Int, onPiece: (MLXArray) -> Void = { _ in }
    ) -> MLXArray {
        run(Self.chunkedWindows(latents, chunk: chunk, overlap: overlap), onPiece: onPiece)
    }

    private func run(_ windows: [SAMEWindow], onPiece: (MLXArray) -> Void) -> MLXArray {
        var pieces: [MLXArray] = []
        for window in windows {
            let piece = self(window.latents)[.ellipsis, window.keep]
            eval(piece)
            pieces.append(piece)
            onPiece(piece)
        }
        return pieces.count == 1 ? pieces[0] : concatenated(pieces, axis: -1)
    }

    static func windows(_ latents: MLXArray) -> [SAMEWindow] {
        let t = latents.dim(2)
        let (chunk, overlap) = window
        if t > chunk + 2 * overlap {
            return chunkedWindows(latents, chunk: chunk, overlap: overlap)
        }
        if t % 2 != 0, t > 6 { return chunkedWindows(latents, chunk: 2, overlap: 2) }
        if t % 2 == 0 { return [SAMEWindow(latents: latents, keep: 0..<(t * stride))] }
        // Too short and odd for any even window: repeat the last latent, then trim.
        let even = concatenated([latents, latents[.ellipsis, (t - 1)...]], axis: -1)
        return [SAMEWindow(latents: even, keep: 0..<(t * stride))]
    }

    /// Every call sees `chunk + 2 * overlap` real latents and keeps the middle
    /// `chunk`; the first and last calls keep their outer edge as well.
    static func chunkedWindows(_ latents: MLXArray, chunk: Int, overlap: Int) -> [SAMEWindow] {
        let t = latents.dim(2)
        let kernel = chunk + 2 * overlap
        precondition(kernel % 2 == 0 && t > kernel, "windows must be even and shorter than the input")
        let s = stride
        var windows = [SAMEWindow(latents: latents[.ellipsis, 0..<kernel], keep: 0..<((chunk + overlap) * s))]
        var i = chunk + overlap
        while i + chunk + overlap <= t {
            windows.append(SAMEWindow(latents: latents[.ellipsis, (i - overlap)..<(i + chunk + overlap)],
                                      keep: (overlap * s)..<((overlap + chunk) * s)))
            i += chunk
        }
        let remaining = t - i
        if remaining > 0 {
            windows.append(SAMEWindow(latents: latents[.ellipsis, (t - kernel)..<t],
                                      keep: ((kernel - remaining) * s)..<(kernel * s)))
        }
        return windows
    }
}

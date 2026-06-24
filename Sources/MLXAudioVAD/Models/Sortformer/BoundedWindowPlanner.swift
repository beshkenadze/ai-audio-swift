import Foundation

/// Frame-accounting result for one bounded streaming step.
///
/// Gives the `generateStreamBounded` caller everything needed to (a) buffer the exact
/// absolute raw-sample range, and (b) slice the per-window `preEncode` embeddings into
/// `[discard left halo | chunk | right context | discard rest]`.
///
/// All counts are pure `Int`; no MLX is involved.
public struct WindowSpec: Equatable, Sendable {
    /// Absolute first raw sample to feed `extractMelFeatures` for this step (hop-aligned).
    public let rawStart: Int
    /// Absolute end (exclusive) of the raw sample range, clamped to known samples.
    public let rawEnd: Int
    /// Global mel-frame index of the window's first frame (= `g0 - leftHalo`).
    public let melWindowStart: Int
    /// Number of mel frames the window spans (`leftHalo + new + rightHalo`).
    public let melWindowFrameCount: Int
    /// Left-halo embedding frames to discard from the front of the window's `preEncode` output.
    public let discardLeftEmb: Int
    /// Chunk embedding frames (the `streamingStepFromEmbeddings` `chunkEmbs`), `<= chunkLenEmb`.
    public let chunkEmbCount: Int
    /// Right-context embedding frames to pass as lookahead (`min(chunkRightContext, available)`).
    public let rcEmbCount: Int
    /// NEW mel frames consumed this step — advance `g0` by this amount.
    public let newMelFrames: Int
}

/// Pure-`Int` window planner for the memory-bounded Sortformer streaming path.
///
/// The bounded path processes the stream in fixed steps of `C = chunkLenEmb * subsamplingFactor`
/// NEW mel frames (fewer in the final EOF step). Each step is computed over a window padded with
/// left/right halos so the mel + `preEncode` output matches the full-file forward exactly:
///
/// - STFT center-pads each window with `nFft/2` zeros, so a sub-window whose raw start is
///   hop-aligned at `a*hop` has sub-window mel frame `j` ≡ global mel frame `a + j`. The window
///   start is therefore `melWindowStart*hop` (hop-aligned by construction); the STFT's own
///   center-pad supplies the left `nFft/2` margin, so the raw start does NOT subtract it.
/// - Halos are multiples of `subsamplingFactor` so the embedding grid stays aligned:
///   `discardLeftEmb = leftHalo / subsamplingFactor` for interior steps, `0` for the first step.
/// - `chunkEmbCount` is the number of embedding frames produced by the new mel region
///   (`embFrames(leftHalo + new) - discardLeftEmb`); summed across contiguous steps it telescopes
///   to the full-file embedding count, so there is no per-step ±1 drift.
public struct BoundedWindowPlanner: Equatable, Sendable {
    public let hop: Int
    public let nFft: Int
    public let winLength: Int
    public let subsamplingFactor: Int
    /// Embedding frames per reference step (`chunkLen`, default 188).
    public let chunkLenEmb: Int
    /// Left halo in mel frames (multiple of `subsamplingFactor`; covers conv radius + STFT margin).
    public let haloLeftMel: Int
    /// Right halo / lookahead in mel frames (multiple of `subsamplingFactor`).
    public let haloRightMel: Int
    /// Right-context embedding frames requested per step (`chunkRightContext`, default 1).
    public let chunkRightContext: Int

    public init(
        hop: Int,
        nFft: Int,
        winLength: Int,
        subsamplingFactor: Int,
        chunkLenEmb: Int = 188,
        haloLeftMel: Int = 16,
        haloRightMel: Int = 16,
        chunkRightContext: Int = 1
    ) {
        self.hop = hop
        self.nFft = nFft
        self.winLength = winLength
        self.subsamplingFactor = subsamplingFactor
        self.chunkLenEmb = chunkLenEmb
        self.haloLeftMel = haloLeftMel
        self.haloRightMel = haloRightMel
        self.chunkRightContext = chunkRightContext
    }

    /// NEW mel frames consumed per interior step (`C = chunkLenEmb * subsamplingFactor`).
    public var stepMelFrames: Int { chunkLenEmb * subsamplingFactor }

    /// `ConvSubsampling` length map: `floor((L-1)/2)+1` applied once per stride-2 stage (3 stages).
    private func embFrames(_ length: Int) -> Int {
        var d = length
        for _ in 0..<3 {
            d = (d - 1) / 2 + 1
        }
        return d
    }

    /// Mel frames a full-file `extractMelFeatures` produces for `samples` samples.
    ///
    /// STFT constant-pads `nFft/2` both sides, so `paddedLen = samples + nFft` and
    /// `numFrames = 1 + (paddedLen - nFft)/hop = 1 + samples/hop` (integer floor).
    public func totalMelFrames(forSamples samples: Int) -> Int {
        guard samples > 0 else { return 0 }
        return 1 + samples / hop
    }

    /// Plan the window for the step whose first NEW (chunk) mel frame is global index `g0`.
    ///
    /// - Parameters:
    ///   - g0: Global mel-frame index of the first new frame for this step.
    ///   - totalKnownSamples: Total raw samples known so far (caps the window at EOF).
    ///   - eof: Whether `totalKnownSamples` is the final sample count (no more audio will arrive).
    ///          When `false`, the right halo is still shrunk to what is currently available; the
    ///          caller is responsible for only stepping once enough samples (or EOF) are buffered.
    public func plan(g0: Int, totalKnownSamples: Int, eof: Bool) -> WindowSpec {
        _ = eof // halo shrink is driven purely by available frames; flag documents intent
        let totalMel = totalMelFrames(forSamples: totalKnownSamples)

        // Halos shrink at the boundaries (left at the start, right at EOF / when little remains).
        let leftHalo = min(haloLeftMel, g0)
        let newMelFrames = max(0, min(stepMelFrames, totalMel - g0))
        let futureMel = totalMel - (g0 + newMelFrames)
        let rightHalo = min(haloRightMel, max(0, futureMel))

        let melWindowStart = g0 - leftHalo
        let melWindowFrameCount = leftHalo + newMelFrames + rightHalo

        // Embedding accounting (window-local; matches the slicing the caller does on preEncode).
        let discardLeftEmb = leftHalo / subsamplingFactor
        let chunkEmbCount = embFrames(leftHalo + newMelFrames) - discardLeftEmb
        let totalWinEmb = embFrames(melWindowFrameCount)
        let availableFutureEmb = max(0, totalWinEmb - discardLeftEmb - chunkEmbCount)
        let rcEmbCount = min(chunkRightContext, availableFutureEmb)

        // Raw sample range. Start is hop-aligned at the window's first mel-frame center; the
        // STFT's own center-pad supplies the left nFft/2 margin. The right end pulls nFft/2 extra
        // so the last kept frame's STFT support is real samples (clamped to known samples at EOF).
        let rawStart = max(0, melWindowStart * hop)
        let rawEndUnclamped = (melWindowStart + melWindowFrameCount) * hop + nFft / 2
        let rawEnd = min(rawEndUnclamped, totalKnownSamples)

        return WindowSpec(
            rawStart: rawStart,
            rawEnd: rawEnd,
            melWindowStart: melWindowStart,
            melWindowFrameCount: melWindowFrameCount,
            discardLeftEmb: discardLeftEmb,
            chunkEmbCount: chunkEmbCount,
            rcEmbCount: rcEmbCount,
            newMelFrames: newMelFrames
        )
    }
}

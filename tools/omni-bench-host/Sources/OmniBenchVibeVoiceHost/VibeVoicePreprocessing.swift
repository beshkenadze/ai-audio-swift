//
//  VibeVoicePreprocessing.swift
//
//  The half of the host contract that must agree byte-for-byte with the CUDA
//  arm in `tools/omni_bench_host_vibevoice/vibevoice_host.py`.
//
//  omni-bench compares two independently-run arms and attributes the delta to
//  the thing under test -- here, the MLX port versus the PyTorch reference.
//  That attribution is only valid if everything *around* the model is
//  identical. Two places would otherwise silently differ:
//
//    * Resampling. Prepared audio is canonically 16 kHz; VibeVoice consumes
//      24 kHz. `MLXAudioCore.resampleAudio` uses AVAudioConverter, whose
//      algorithm is undocumented and certainly not librosa's, so using the
//      convenient thing on each side would feed the two arms different
//      waveforms and the resulting WER delta would be blamed on the port.
//      `VibeVoiceResampler` is therefore written from an explicit formula that
//      the Python side reimplements verbatim, and the agreement is *proved*
//      against a golden vector rather than assumed.
//
//    * Hypothesis cleanup. The streaming prompt asks the model to transcribe
//      "with these keys: speaker, content", so chunks arrive framed as
//      `" \n Speaker 0:text"`. Stripping that framing on one side only would
//      look like a large quality difference.
//

import Foundation

public enum VibeVoicePreprocessing {

    /// Sample rate the VibeVoice tokenizers are trained on.
    public static let modelSampleRate = 24_000

    /// Samples per acoustic latent frame at 24 kHz.
    public static let tokenizerHop = 3200
}

// MARK: - Resampling

/// Bandlimited sinc interpolation to 24 kHz.
///
/// Mirrors `resample_16k_to_24k` in the Python host. 16 kHz -> 24 kHz is
/// upsampling, so the kernel needs no anti-alias narrowing: it keeps the full
/// input band (cutoff at input Nyquist, 0.5 cycles per input sample) and
/// interpolates. Output sample `n` reads input position
/// `t = n * sourceRate / 24000` and convolves the `zeros` input samples either
/// side with `sinc(t - k) * blackman(...)`.
public enum VibeVoiceResampler {

    /// Zero crossings each side of the kernel. Larger means a sharper
    /// transition band and more work; 32 puts reconstruction error far below
    /// the bf16 checkpoint's own noise floor, so it cannot be what a parity
    /// delta is measuring.
    public static let zeros = 32

    public enum Error: Swift.Error, CustomStringConvertible {
        case downsamplingUnsupported(from: Int, to: Int)

        public var description: String {
            switch self {
            case let .downsamplingUnsupported(from, to):
                return "downsampling \(from) -> \(to) is not part of the shared "
                    + "contract; prepared omni-bench audio is 16 kHz"
            }
        }
    }

    /// Blackman window over `u` in [0, 1]. Written out rather than taken from a
    /// library so both arms derive the same coefficients from the same formula.
    @inline(__always)
    private static func blackman(_ u: Double) -> Double {
        0.42 - 0.5 * Foundation.cos(2.0 * Double.pi * u)
            + 0.08 * Foundation.cos(4.0 * Double.pi * u)
    }

    @inline(__always)
    private static func sinc(_ x: Double) -> Double {
        if x == 0 { return 1.0 }
        let px = Double.pi * x
        return Foundation.sin(px) / px
    }

    public static func resample(_ samples: [Float], from sourceRate: Int) throws -> [Float] {
        let target = VibeVoicePreprocessing.modelSampleRate
        if sourceRate == target { return samples }
        if sourceRate > target {
            throw Error.downsamplingUnsupported(from: sourceRate, to: target)
        }
        if samples.isEmpty { return [] }

        let step = Double(sourceRate) / Double(target)  // input samples per output sample
        let inCount = samples.count
        let outCount = Int((Double(inCount) / step).rounded(.down))
        if outCount <= 0 { return [] }
        let half = zeros

        var output = [Float](repeating: 0, count: outCount)
        // Accumulate in Double: the taps span three orders of magnitude and a
        // Float accumulator would make the result depend on summation order,
        // which is exactly the kind of difference this file exists to remove.
        samples.withUnsafeBufferPointer { x in
            for n in 0..<outCount {
                let t = Double(n) * step
                let base = Int(t.rounded(.down))
                var acc = 0.0
                for offset in (-half + 1)...half {
                    let idx = base + offset
                    if idx < 0 || idx >= inCount { continue }
                    let dist = t - Double(idx)
                    if abs(dist) >= Double(half) { continue }
                    let weight = sinc(dist) * blackman((dist + Double(half)) / (2.0 * Double(half)))
                    acc += Double(x[idx]) * weight
                }
                output[n] = Float(acc)
            }
        }
        return output
    }

    /// Incremental resampler whose concatenated output is *identical* to
    /// resampling the whole signal at once.
    ///
    /// Resampling each paced 100 ms chunk independently would put a kernel
    /// discontinuity at every chunk boundary, so the streaming seam would feed
    /// the model a subtly different waveform than the batch seam -- and the two
    /// arms would disagree for a reason that has nothing to do with either
    /// model. This keeps the `zeros` samples of context the kernel needs and
    /// emits an output sample only once every tap it depends on has arrived.
    public final class Stream {
        private let sourceRate: Int
        private let step: Double
        private var pending: [Float] = []
        /// Number of source samples already dropped off the front of `pending`;
        /// tap indices are global, so this maps them back into the buffer.
        private var droppedInput = 0
        private var nextOutput = 0
        private var finished = false

        public init(sourceRate: Int) throws {
            let target = VibeVoicePreprocessing.modelSampleRate
            if sourceRate > target {
                throw Error.downsamplingUnsupported(from: sourceRate, to: target)
            }
            self.sourceRate = sourceRate
            self.step = Double(sourceRate) / Double(target)
        }

        public var isPassthrough: Bool { sourceRate == VibeVoicePreprocessing.modelSampleRate }

        /// Accepts source samples and returns every output sample that is now
        /// fully determined.
        public func push(_ samples: [Float]) -> [Float] {
            if isPassthrough { return samples }
            pending.append(contentsOf: samples)
            return drain(allowIncomplete: false)
        }

        /// Flushes the tail, treating input past the end as zero exactly as the
        /// whole-signal path does.
        public func finish() -> [Float] {
            if isPassthrough { return [] }
            finished = true
            let out = drain(allowIncomplete: true)
            pending.removeAll()
            return out
        }

        private func drain(allowIncomplete: Bool) -> [Float] {
            let half = VibeVoiceResampler.zeros
            let available = droppedInput + pending.count
            // The whole-signal path produces floor(totalInput / step) samples;
            // without knowing totalInput yet, cap at what the input so far
            // supports and let finish() emit the remainder.
            let limit = allowIncomplete
                ? Int((Double(available) / step).rounded(.down))
                : Int.max

            var out: [Float] = []
            while nextOutput < limit {
                let t = Double(nextOutput) * step
                let base = Int(t.rounded(.down))
                if !allowIncomplete && base + half >= available { break }

                var acc = 0.0
                for offset in (-half + 1)...half {
                    let idx = base + offset
                    if idx < 0 || idx >= available { continue }
                    let local = idx - droppedInput
                    if local < 0 || local >= pending.count { continue }
                    let dist = t - Double(idx)
                    if abs(dist) >= Double(half) { continue }
                    let weight = VibeVoiceResampler.sinc(dist)
                        * VibeVoiceResampler.blackman(
                            (dist + Double(half)) / (2.0 * Double(half)))
                    acc += Double(pending[local]) * weight
                }
                out.append(Float(acc))
                nextOutput += 1
            }

            if !finished {
                // Everything below the leftmost tap of the next output is dead.
                let nextBase = Int((Double(nextOutput) * step).rounded(.down))
                let keepFrom = max(0, nextBase - half + 1)
                let drop = min(max(0, keepFrom - droppedInput), pending.count)
                if drop > 0 {
                    pending.removeFirst(drop)
                    droppedInput += drop
                }
            }
            return out
        }
    }
}

// MARK: - Hypothesis cleanup

public enum VibeVoiceHypothesis {

    /// `" \n Speaker 0:text"` -> `"text"`. Anchored at a line start so a literal
    /// mention of a speaker inside the transcription is not eaten.
    private static let speakerLabel = try! NSRegularExpression(
        pattern: #"^\s*Speaker\s+\d+\s*:\s*"#,
        options: [.anchorsMatchLines])

    /// Removes the diarization scaffolding the streaming prompt asks for.
    ///
    /// The model is instructed to transcribe "with these keys: speaker,
    /// content", so every chunk is framed as `Speaker N:`. That framing is
    /// protocol, not hypothesis -- leaving it in would inflate WER against raw
    /// orthographic references for a reason unrelated to recognition quality.
    /// Bracketed events like `[Laughter]` are the model's transcription choice
    /// and are left alone; the Scoring Profile's normalizer owns those.
    public static func stripSpeakerLabels(_ text: String) -> String {
        let range = NSRange(text.startIndex..<text.endIndex, in: text)
        let stripped = speakerLabel.stringByReplacingMatches(
            in: text, options: [], range: range, withTemplate: " ")
        return stripped.split(whereSeparator: { $0.isWhitespace }).joined(separator: " ")
    }
}

// MARK: - Windowing

/// Chunk/lookahead sizes in samples at 24 kHz.
///
/// The lookahead is snapped to a whole tokenizer frame exactly as the reference
/// does (`round(delay / frameDuration) * frameDuration`): the encoder only
/// produces latents on frame boundaries, so a partial frame of lookahead would
/// round somewhere else on the other arm and desync the two.
public struct VibeVoiceWindowPlan: Equatable {
    public let chunkSamples: Int
    public let lookaheadSamples: Int

    public var windowSamples: Int { chunkSamples + lookaheadSamples }

    public init(chunkDuration: Double, textAudioDelay: Double) {
        let frameDuration =
            Double(VibeVoicePreprocessing.tokenizerHop)
            / Double(VibeVoicePreprocessing.modelSampleRate)
        // Python's round() is round-half-to-even; Swift's bare .rounded() is
        // half-away-from-zero. They agree for the 0.5 s default (3.75 -> 4) but
        // would split on an exact .5, so match the reference explicitly.
        let lookaheadSeconds = (textAudioDelay / frameDuration).rounded(.toNearestOrEven) * frameDuration
        self.chunkSamples = Int(chunkDuration * Double(VibeVoicePreprocessing.modelSampleRate))
        self.lookaheadSamples = Int(lookaheadSeconds * Double(VibeVoicePreprocessing.modelSampleRate))
    }
}

import Foundation
import MLX

// Two-tier streaming ASR. A fast monotonic lane (Nemotron at minimal latency) emits
// instant *partial* text; an accurate lane (Voxtral native streaming, 480 ms delay)
// emits *confirmed* text that overwrites the partials cross-model. Result: Nemotron's
// latency with Voxtral's accuracy — the DeepGram interim→final UX, WITHOUT the
// streaming-LA penalty.
//
// Why not Local Agreement over Voxtral? Measured: Voxtral's *native* streaming is more
// accurate (EN 2.24 % vs LA 4.48 %; on noised RU LA hallucinates/drops/switches
// language), because Voxtral is streaming-trained and re-decoding short windows throws
// that away. So the accurate lane is native streaming; the "revision" UX comes from the
// fast→accurate cross-model replacement, not from self-agreement.
//
// Merge: both lanes transcribe the same audio in order, and Voxtral lags, so its text
// is the accurate prefix while Nemotron's words beyond Voxtral's reach are the volatile
// tail. The junction is count-based; transient glitches there self-heal as Voxtral
// advances. (Upgrade path: align by [STREAMING_WORD] word-times once Nemotron exposes
// token timestamps.)

public final class TwoTierSession {
    private let fast: NemotronASRStreamSession      // instant partials, monotonic
    private let accurate: VoxtralRealtimeStreamSession  // accurate finals, lags

    public init(nemotron: NemotronASRModel, voxtral: VoxtralRealtimeModel,
                language: String? = nil, fastChunkMs: Int = 80) {
        self.fast = nemotron.makeStreamSession(language: language, chunkMs: fastChunkMs)
        self.accurate = voxtral.makeStreamSession()
    }

    /// Accurate (Voxtral) text covered so far — not revised once Voxtral commits it.
    public var confirmed: String { accurate.text }

    /// Instant (Nemotron) tail beyond Voxtral's coverage — provisional, to be replaced.
    public var partial: String {
        let conf = Self.words(accurate.text)
        let fastW = Self.words(fast.text)
        return fastW.count > conf.count ? fastW[conf.count...].joined(separator: " ") : ""
    }

    /// Full live view: confirmed prefix + provisional tail.
    public var text: String { let p = partial; return p.isEmpty ? confirmed : confirmed + " " + p }

    /// Ingest 16 kHz mono samples into both lanes; returns the current split.
    private let debug = ProcessInfo.processInfo.environment["TWOTIER_DEBUG"] != nil
    private var steps = 0

    @discardableResult
    public func step(_ samples: [Float]) -> (confirmed: String, partial: String) {
        _ = fast.step(samples)
        _ = accurate.step(samples)
        let (c, p) = (confirmed, partial)
        if debug { steps += 1; if steps % 12 == 0 {
            FileHandle.standardError.write(Data("[2TIER] conf=\(Self.words(c).count)w  partial=⟨\(p)⟩\n".utf8))
        } }
        return (c, p)
    }

    /// End of stream: flush both lanes. Voxtral is the authority — its full text is the
    /// final transcript (the partial tail is subsumed once Voxtral catches up).
    @discardableResult
    public func finish() -> (confirmed: String, partial: String) {
        _ = fast.finish()
        _ = accurate.finish()
        return (accurate.text, "")
    }

    private static func words(_ s: String) -> [String] {
        s.split(whereSeparator: { $0 == " " || $0 == "\n" }).map(String.init)
    }
}

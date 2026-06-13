import Foundation
import MLX

// Local-Agreement-2 streaming over Voxtral — the faithful DeepGram interim→final
// pattern, on-device, vendor-independent. ONE model (Voxtral) "agrees with itself":
// every tick we re-decode the unconfirmed audio window with the offline decoder (full
// bidirectional attention, so early words may revise as right context arrives), and
// commit the longest word-prefix that two consecutive decodes agree on. The agreed
// prefix is `confirmed` (stable, never revised); the rest is `volatile` (may change).
//
// Word boundaries come from the [STREAMING_WORD] token (id 33 in tekken.json); since
// the decoder emits one token per 80 ms frame, a word's terminating [STREAMING_WORD]
// index is its end-time (offset by the transcription delay). Those end-times let us
// TRIM committed audio out of the window, so each re-decode is bounded (≈ the
// unconfirmed tail) instead of O(n²) over the whole stream.
//
// Nemotron is NOT part of this agreement — it is a separate instant-partial lane.

public final class VoxtralLocalAgreementSession {
    public struct Word: Sendable { public let text: String; public let endS: Double }

    private let model: VoxtralRealtimeModel
    private let tickSamples: Int
    private let streamingWordTokenId: Int   // [STREAMING_WORD] = 33 (tekken.json)
    private let framesPerSec = 12.5         // VoxtralRealtimeConstants.frameRate
    private let delayS: Double
    private let leftContextS = 0.5          // audio kept before the cut for decode quality
    private let maxWindowS = 30.0           // safety cap if nothing commits

    private var buffer: [Float] = []        // unconfirmed audio, starting at bufferStartS
    private var bufferStartS = 0.0
    private var sinceTick = 0
    private var prevNorm: [String] = []     // previous decode's words (normalized) for LA-2
    private var confirmed: [Word] = []      // committed words (absolute end-times)
    private var lastVolatile: [Word] = []   // current unconfirmed tail (for display)

    private let debug = ProcessInfo.processInfo.environment["VOXLA_DEBUG"] != nil
    private var redecodes = 0

    public init(model: VoxtralRealtimeModel, tickMs: Int = 600, streamingWordTokenId: Int = 33) {
        self.model = model
        self.tickSamples = max(1, 16000 * tickMs / 1000)
        self.streamingWordTokenId = streamingWordTokenId
        self.delayS = Double(model.config.transcriptionDelayMs) / 1000.0
    }

    public var confirmedWords: [Word] { confirmed }
    public var volatileWords: [Word] { lastVolatile }
    public var confirmedText: String { confirmed.map(\.text).joined() }
    public var volatileText: String { lastVolatile.map(\.text).joined() }
    public var text: String { confirmedText + volatileText }

    /// Ingest 16 kHz mono samples. Re-decodes on tick boundaries; returns the current
    /// (confirmed, volatile) split.
    @discardableResult
    public func step(_ samples: [Float]) -> (confirmed: String, volatile: String) {
        buffer.append(contentsOf: samples)
        sinceTick += samples.count
        if sinceTick >= tickSamples { sinceTick = 0; redecode(final: false) }
        return (confirmedText, volatileText)
    }

    /// End of stream: final re-decode with full right context, then promote the
    /// remaining tail to confirmed (nothing left to revise).
    @discardableResult
    public func finish() -> (confirmed: String, volatile: String) {
        redecode(final: true)
        confirmed.append(contentsOf: lastVolatile)
        lastVolatile = []
        return (confirmedText, "")
    }

    private func redecode(final: Bool) {
        guard !buffer.isEmpty else { return }
        let raw = wordsFrom(tokens: model.transcribeTokens(audio: MLXArray(buffer)))
        // Absolute end-times; drop any word ending inside already-committed time — the
        // retained left-context re-emits the last committed word, which we must not
        // re-commit (else "early Early dawn's" duplication).
        // Time filter: re-emitted words (from the retained left context, incl. misreads
        // like "stroll"→"Troll") end inside already-committed time — drop them.
        let confEnd = confirmed.last?.endS ?? -1
        var words = raw.map { Word(text: $0.text, endS: bufferStartS + $0.endS) }
                       .filter { $0.endS > confEnd + 0.2 }
        // Overlap-join: also drop a leading prefix that matches the committed suffix
        // (by normalized text) — catches exact re-emits whose segmentation shifted.
        let maxOv = min(words.count, confirmed.count, 5)
        var skip = 0
        for k in stride(from: maxOv, through: 1, by: -1) where
            confirmed.suffix(k).map({ Self.norm($0.text) }) == words.prefix(k).map({ Self.norm($0.text) }) {
            skip = k; break
        }
        if skip > 0 { words.removeFirst(skip) }
        let cur = words.map { Self.norm($0.text) }

        // LA-2: prefix two consecutive decodes agree on is stable. On the final flush
        // there is no "next" decode, so commit everything.
        let agree = final ? words.count : Self.commonPrefix(prevNorm, cur)

        if agree > 0 {
            let committedNow = Array(words[0..<agree])
            confirmed.append(contentsOf: committedNow)
            // Trim committed audio out of the window (keep a little left context).
            let cutAbs = max(bufferStartS, committedNow.last!.endS - leftContextS)
            let dropN = min(buffer.count, Int((cutAbs - bufferStartS) * 16000))
            if dropN > 0 { buffer.removeFirst(dropN); bufferStartS += Double(dropN) / 16000 }
            prevNorm = Array(cur[agree...])   // tail carries over by text (frame-independent)
            lastVolatile = Array(words[agree...])
        } else {
            prevNorm = cur
            lastVolatile = words
        }

        // Safety: if nothing commits and the window grows too large, hard-trim oldest.
        let windowS = Double(buffer.count) / 16000
        if windowS > maxWindowS {
            let dropN = Int((windowS - maxWindowS) * 16000)
            buffer.removeFirst(min(dropN, buffer.count)); bufferStartS += Double(dropN) / 16000
            prevNorm = []
        }

        if debug, redecodes < 4 {
            redecodes += 1
            FileHandle.standardError.write(Data(
                "[VOXLA] #\(redecodes): win=\(String(format: "%.1f", windowS))s, \(words.count)w, agree=\(agree), conf=\(confirmed.count) | tail: \(lastVolatile.map(\.text).joined())\n".utf8))
        }
    }

    /// Split the token stream on [STREAMING_WORD]; each segment's text is the word, its
    /// end-time the (delay-corrected) frame index of the boundary token.
    private func wordsFrom(tokens: [Int]) -> [Word] {
        var words: [Word] = []
        var segStart = 0
        for (i, t) in tokens.enumerated() where t == streamingWordTokenId {
            appendWord(Array(tokens[segStart..<i]), endIndex: i, into: &words)
            segStart = i + 1
        }
        if segStart < tokens.count {   // trailing in-progress word (inherently volatile)
            appendWord(Array(tokens[segStart...]), endIndex: tokens.count, into: &words)
        }
        return words
    }

    private func appendWord(_ seg: [Int], endIndex: Int, into words: inout [Word]) {
        let txt = model.decodeStreaming(seg)   // tokenizer strips [STREAMING_PAD]/[STREAMING_WORD]
        guard !txt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        let endS = max(0, Double(endIndex) / framesPerSec - delayS)
        words.append(Word(text: txt, endS: endS))
    }

    private static func norm(_ s: String) -> String {
        s.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
    }

    private static func commonPrefix(_ a: [String], _ b: [String]) -> Int {
        let n = min(a.count, b.count)
        var k = 0
        while k < n, a[k] == b[k] { k += 1 }
        return k
    }
}

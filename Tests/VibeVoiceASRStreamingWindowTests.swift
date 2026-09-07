//
//  VibeVoiceASRStreamingWindowTests.swift
//
//  The overlapping-window cursor is the part of VibeVoice ASR streaming that
//  is easiest to get subtly wrong and hardest to notice: an off-by-one in the
//  stride or a dropped tail window does not crash, it just quietly degrades
//  the transcript. These tests pin `VibeVoiceWindowBuffer` against a direct
//  transcription of the reference loop in
//  `vibevoice/modular/modeling_vibevoice_asr.py:495-515`:
//
//      text_start = 0
//      while text_start < total:
//          end = min(text_start + chunk + lookahead, total)
//          if end > text_start:
//              seg = audio[text_start:end]
//              if pad_last_chunk and len(seg) < chunk + lookahead:
//                  seg = pad(seg, chunk + lookahead)
//              emit(seg)
//          text_start = min(text_start + chunk, total)
//
//  Run:
//    xcodebuild test -scheme MLXAudio-Package -destination 'platform=macOS' \
//      -only-testing:MLXAudioTests/VibeVoiceWindowBufferTests
//

import Foundation
import Testing

@testable import MLXAudioSTT

@Suite("VibeVoiceWindowBufferTests")
struct VibeVoiceWindowBufferTests {

    /// Direct transcription of the reference Python loop.
    private func referenceWindows(
        total: Int, chunk: Int, lookahead: Int, padLastChunk: Bool
    ) -> [[Float]] {
        let audio = (0..<total).map { Float($0) }
        let windowSamples = chunk + lookahead

        var out: [[Float]] = []
        var textStart = 0
        while textStart < total {
            let end = min(textStart + windowSamples, total)
            if end > textStart {
                var seg = Array(audio[textStart..<end])
                if padLastChunk && seg.count < windowSamples {
                    seg.append(contentsOf: [Float](repeating: 0, count: windowSamples - seg.count))
                }
                out.append(seg)
            }
            textStart = min(textStart + chunk, total)
        }
        return out
    }

    private func bufferWindows(
        total: Int, chunk: Int, lookahead: Int, padLastChunk: Bool, pushSizes: [Int]
    ) -> [[Float]] {
        let audio = (0..<total).map { Float($0) }
        var buffer = VibeVoiceWindowBuffer(
            chunkSamples: chunk, lookaheadSamples: lookahead, padLastChunk: padLastChunk)

        var out: [[Float]] = []
        var offset = 0
        var sizeIndex = 0
        while offset < total {
            let size = min(pushSizes[sizeIndex % pushSizes.count], total - offset)
            out.append(contentsOf: buffer.push(Array(audio[offset..<(offset + size)])))
            offset += size
            sizeIndex += 1
        }
        out.append(contentsOf: buffer.flush())
        return out
    }

    @Test("whole-clip push reproduces the reference window schedule")
    func wholeClipMatchesReference() {
        let chunk = 48000
        let lookahead = 12800

        // Exact multiples, ragged tails, shorter-than-one-window clips.
        for total in [
            chunk, chunk + lookahead, 3 * chunk, 3 * chunk + 1, 5 * chunk - 7,
            chunk - 1, 1, 2 * chunk + lookahead,
        ] {
            let expected = referenceWindows(
                total: total, chunk: chunk, lookahead: lookahead, padLastChunk: true)
            let actual = bufferWindows(
                total: total, chunk: chunk, lookahead: lookahead, padLastChunk: true,
                pushSizes: [total])

            #expect(actual.count == expected.count, "window count for total=\(total)")
            #expect(actual == expected, "window contents for total=\(total)")
        }
    }

    @Test("chunk schedule is independent of how audio is fed in")
    func irregularPushesMatchWholeClip() {
        let chunk = 4800
        let lookahead = 1280
        let total = 7 * chunk + 137

        let expected = referenceWindows(
            total: total, chunk: chunk, lookahead: lookahead, padLastChunk: true)

        // Feed sizes both far below and far above one window, plus sizes
        // that are coprime with the stride so boundaries never line up.
        for pushSizes in [[1], [7], [chunk / 3], [chunk], [chunk + lookahead], [3, 1777, 40001]] {
            let actual = bufferWindows(
                total: total, chunk: chunk, lookahead: lookahead, padLastChunk: true,
                pushSizes: pushSizes)
            #expect(actual == expected, "push pattern \(pushSizes)")
        }
    }

    @Test("windows overlap by exactly the lookahead")
    func consecutiveWindowsOverlapByLookahead() {
        let chunk = 100
        let lookahead = 30
        let total = 5 * chunk

        let windows = bufferWindows(
            total: total, chunk: chunk, lookahead: lookahead, padLastChunk: true,
            pushSizes: [total])

        #expect(windows.count >= 2)
        for i in 0..<(windows.count - 1) {
            let current = windows[i]
            let next = windows[i + 1]
            // The tail of window i beyond the stride must be the head of
            // window i+1 -- that shared region is the lookahead being
            // re-encoded, which is what makes the scheme stateless.
            #expect(Array(current.suffix(lookahead)) == Array(next.prefix(lookahead)), "overlap at \(i)")
        }
    }

    @Test("mid-stream windows are never padded")
    func midStreamWindowsAreFullLength() {
        let chunk = 100
        let lookahead = 30
        var buffer = VibeVoiceWindowBuffer(
            chunkSamples: chunk, lookaheadSamples: lookahead, padLastChunk: true)

        // Only a partial window is available: nothing may be emitted yet,
        // because padding it here would feed the model silence that the
        // next push is about to supply for real.
        #expect(buffer.push([Float](repeating: 1, count: chunk)).isEmpty)

        let emitted = buffer.push([Float](repeating: 1, count: lookahead))
        #expect(emitted.count == 1)
        #expect(emitted[0].count == chunk + lookahead)
    }

    @Test("unpadded tail keeps its true length")
    func unpaddedTailIsNotExtended() {
        let chunk = 100
        let lookahead = 30
        let total = chunk + 10

        let windows = bufferWindows(
            total: total, chunk: chunk, lookahead: lookahead, padLastChunk: false,
            pushSizes: [total])
        let expected = referenceWindows(
            total: total, chunk: chunk, lookahead: lookahead, padLastChunk: false)

        #expect(windows == expected)
        // Without padding the tail window is exactly the leftover audio
        // (total - chunk = 10 samples), not a full-length window.
        #expect(windows.count == 2)
        #expect(windows.last?.count == total - chunk)
    }

    @Test("every sample reaches at least one window")
    func noSamplesAreDropped() {
        let chunk = 320
        let lookahead = 96
        let total = 4 * chunk + 41

        let windows = bufferWindows(
            total: total, chunk: chunk, lookahead: lookahead, padLastChunk: true,
            pushSizes: [521])

        // Samples are the integers 0..<total, so a set union over the
        // windows must cover all of them (zero padding aside).
        var seen = Set<Int>()
        for window in windows {
            for value in window { seen.insert(Int(value)) }
        }
        for sample in 0..<total {
            #expect(seen.contains(sample), "sample \(sample) never encoded")
        }
    }

    @Test("flush is idempotent and push after flush is ignored")
    func flushIsTerminal() {
        let chunk = 100
        let lookahead = 30
        var buffer = VibeVoiceWindowBuffer(
            chunkSamples: chunk, lookaheadSamples: lookahead, padLastChunk: true)

        _ = buffer.push([Float](repeating: 1, count: chunk + lookahead + 10))
        let first = buffer.flush()
        #expect(!first.isEmpty)
        #expect(buffer.flush().isEmpty)
        #expect(buffer.push([Float](repeating: 1, count: 10_000)).isEmpty)
    }

    @Test("empty input produces no windows")
    func emptyInputIsNoOp() {
        var buffer = VibeVoiceWindowBuffer(
            chunkSamples: 100, lookaheadSamples: 30, padLastChunk: true)
        #expect(buffer.push([]).isEmpty)
        #expect(buffer.flush().isEmpty)
    }
}

@Suite("VibeVoiceStreamingTextTests")
struct VibeVoiceStreamingTextTests {

    @Test("framing tokens are stripped from chunk text")
    func framingTokensAreStripped() {
        // `skipSpecialTokens` does not drop these -- they are ordinary added
        // tokens in this vocabulary -- so the literal strip is load-bearing.
        let raw = "<|object_ref_start|>Speaker 1: hello<|object_ref_end|><|text_chunk_end|>"
        #expect(VibeVoiceASRStreamSession.cleanChunkText(raw) == "Speaker 1: hello")
    }

    @Test("ordinary text is left untouched")
    func ordinaryTextSurvives() {
        let raw = "Speaker 2: the price is <5 dollars | maybe"
        #expect(VibeVoiceASRStreamSession.cleanChunkText(raw) == raw)
    }

    @Test("default prompt matches the reference wording")
    func defaultPromptMatchesReference() {
        // The streaming checkpoints were trained on this exact string
        // (`modeling_vibevoice_asr.py:597-608`); drift here silently costs
        // accuracy rather than failing loudly.
        #expect(
            VibeVoiceASRStreamSession.defaultPrompt(contextInfo: nil)
                == "You are a helpful assistant that transcribes audio input into text output. "
                    + "Please transcribe the following audios streamingly with these keys: speaker, content\n"
        )
        #expect(
            VibeVoiceASRStreamSession.defaultPrompt(contextInfo: "Acme Corp")
                == "You are a helpful assistant that transcribes audio input into text output. "
                    + "Please transcribe the following audios streamingly with these keys: speaker, content "
                    + "and extra info: Acme Corp\n"
        )
    }
}

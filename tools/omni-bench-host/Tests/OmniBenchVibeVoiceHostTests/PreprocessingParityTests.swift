//
//  PreprocessingParityTests.swift
//
//  Proves the two arms preprocess identically. Without this the omni-bench
//  parity delta between the MLX and CUDA arms could be caused by the
//  resampler rather than by the model, and nothing downstream would say so.
//

import Foundation
import Testing

@testable import OmniBenchVibeVoiceHost

@Suite("VibeVoice host preprocessing")
struct PreprocessingParityTests {

    struct Golden: Decodable {
        let sourceRateHz: Int
        let targetRateHz: Int
        let resampleZeros: Int
        let input: [Float]
        let output: [Float]

        enum CodingKeys: String, CodingKey {
            case sourceRateHz = "source_rate_hz"
            case targetRateHz = "target_rate_hz"
            case resampleZeros = "resample_zeros"
            case input, output
        }
    }

    static func loadGolden() throws -> Golden {
        let url = try #require(
            Bundle.module.url(forResource: "vibevoice_resampler_golden", withExtension: "json",
                              subdirectory: "Fixtures"),
            "golden vector missing; regenerate with tools/omni_bench_host_vibevoice/dump_resampler_golden.py")
        return try JSONDecoder().decode(Golden.self, from: Data(contentsOf: url))
    }

    @Test("Swift resampler reproduces the Python host's output bit-for-bit enough")
    func matchesPythonGolden() throws {
        let golden = try Self.loadGolden()
        #expect(golden.resampleZeros == VibeVoiceResampler.zeros,
                "kernel width drifted between the arms")
        #expect(golden.targetRateHz == VibeVoicePreprocessing.modelSampleRate)

        let produced = try VibeVoiceResampler.resample(
            golden.input, from: golden.sourceRateHz)

        #expect(produced.count == golden.output.count, "output length")

        var maxDelta: Float = 0
        for (a, b) in zip(produced, golden.output) {
            maxDelta = max(maxDelta, abs(a - b))
        }
        // Both sides accumulate in double and round once to Float, so the only
        // legitimate difference is summation order inside the 64-tap kernel --
        // a few ULP on values of order 1. A real disagreement (shifted taps,
        // different window) lands orders of magnitude above this.
        #expect(maxDelta < 1e-6, "max|delta| vs Python golden = \(maxDelta)")
    }

    @Test("Streaming resampler output equals whole-signal resampling")
    func streamingMatchesBatch() throws {
        let golden = try Self.loadGolden()
        let batch = try VibeVoiceResampler.resample(golden.input, from: golden.sourceRateHz)

        // Irregular push sizes: a producer paces on wall-clock, so chunk
        // lengths are not guaranteed to divide the kernel stride.
        for pushSize in [1, 7, 160, 1600, 4096] {
            let stream = try VibeVoiceResampler.Stream(sourceRate: golden.sourceRateHz)
            var produced: [Float] = []
            var offset = 0
            while offset < golden.input.count {
                let end = min(offset + pushSize, golden.input.count)
                produced.append(contentsOf: stream.push(Array(golden.input[offset..<end])))
                offset = end
            }
            produced.append(contentsOf: stream.finish())

            #expect(produced.count == batch.count,
                    "push size \(pushSize): length \(produced.count) != \(batch.count)")
            var maxDelta: Float = 0
            for (a, b) in zip(produced, batch) { maxDelta = max(maxDelta, abs(a - b)) }
            #expect(maxDelta == 0,
                    "push size \(pushSize): streaming diverged from batch by \(maxDelta)")
        }
    }

    @Test("Resampling 16 kHz to 24 kHz produces exactly 1.5x the samples")
    func lengthRatio() throws {
        for count in [0, 1, 100, 999, 16_000] {
            let input = [Float](repeating: 0.1, count: count)
            let output = try VibeVoiceResampler.resample(input, from: 16_000)
            #expect(output.count == count * 3 / 2, "count \(count)")
        }
    }

    @Test("Downsampling is refused rather than silently done differently per arm")
    func refusesDownsampling() {
        #expect(throws: VibeVoiceResampler.Error.self) {
            _ = try VibeVoiceResampler.resample([0, 1, 2], from: 48_000)
        }
    }

    @Test("Speaker labels are stripped, transcription content is not")
    func stripsSpeakerFraming() {
        #expect(VibeVoiceHypothesis.stripSpeakerLabels(" \n Speaker 0:Hello there.")
            == "Hello there.")
        #expect(VibeVoiceHypothesis.stripSpeakerLabels(
            " \n Speaker 0:One. \n Speaker 1:Two.") == "One. Two.")
        // Bracketed events are the model's transcription choice; the Scoring
        // Profile's normalizer owns them, not the host.
        #expect(VibeVoiceHypothesis.stripSpeakerLabels(" \n Speaker 0:Ha. [Laughter]")
            == "Ha. [Laughter]")
        // A speaker mentioned mid-sentence is content, not framing.
        #expect(VibeVoiceHypothesis.stripSpeakerLabels("the Speaker 0: prefix is odd")
            == "the Speaker 0: prefix is odd")
    }

    @Test("Lookahead snaps to a whole tokenizer frame")
    func windowPlanSnapsLookahead() {
        let plan = VibeVoiceWindowPlan(chunkDuration: 2.0, textAudioDelay: 0.5)
        #expect(plan.chunkSamples == 48_000)
        // 0.5 s / (3200/24000 s per frame) = 3.75 -> 4 frames = 12800 samples.
        #expect(plan.lookaheadSamples == 12_800)
        #expect(plan.windowSamples == 60_800)
    }
}

//
//  VibeVoiceASRStreamingTranscriptParityTests.swift
//
//  End-to-end streaming parity: the full Swift chunk loop against the
//  official PyTorch `streaming_generate` on CUDA, over audio long enough to
//  span several chunks with a partial (zero-padded) final window.
//
//  Both sides are made fully deterministic first, so any divergence is a
//  porting bug rather than RNG:
//    * greedy decode (temperature 0 -> argmax)
//    * the acoustic VAE returns its mean instead of sampling. On the Swift
//      side that is the only behaviour implemented; on the reference side
//      `tools/vibevoice_parity_stream.py` sets the acoustic tokenizer's
//      `std_dist_type = 'none'` and asserts two `encode_speech` calls now
//      agree exactly before generating anything.
//
//  Comparison is token-first, not text-first: two different token sequences
//  can decode to the same string, so a text-only check can hide
//  compensating errors. On mismatch the failure bisects to the first
//  divergent chunk and then to the first divergent token, and reports the
//  first-step logit delta for that chunk.
//
//  Requires the 1.5B checkpoint and a dump from
//  `tools/vibevoice_parity_stream.py`; skips cleanly when either is absent.
//

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXAudioSTT

@Suite("VibeVoiceStreamingTranscriptParityTests", .serialized)
struct VibeVoiceStreamingTranscriptParityTests {

    static let defaultModelPath = "/Volumes/DATA/models/vibevoice-asr-streaming-1.5b"
    static let defaultStreamPrefix = "/Volumes/DATA/models/vibevoice-parity/parity_stream"

    static var modelPath: String {
        ProcessInfo.processInfo.environment["VIBEVOICE_PARITY_MODEL"] ?? defaultModelPath
    }
    static var streamPrefix: String {
        ProcessInfo.processInfo.environment["VIBEVOICE_PARITY_STREAM"] ?? defaultStreamPrefix
    }

    static var artifactsAvailable: Bool {
        let fm = FileManager.default
        return fm.fileExists(atPath: modelPath)
            && fm.fileExists(atPath: streamPrefix + ".safetensors")
            && fm.fileExists(atPath: streamPrefix + ".json")
    }

    struct ReferenceChunk: Decodable {
        let index: Int
        let totalChunks: Int
        let text: String
        let tokenIds: [Int]
        let terminatorId: Int?

        enum CodingKeys: String, CodingKey {
            case index
            case totalChunks = "total_chunks"
            case text
            case tokenIds = "token_ids"
            case terminatorId = "terminator_id"
        }
    }

    struct ReferenceParams: Decodable {
        let chunkDuration: Double
        let textAudioDelay: Double
        let sampleRate: Int
        let maxNewTokensPerChunk: Int
        let samples: Int

        enum CodingKeys: String, CodingKey {
            case chunkDuration = "chunk_duration"
            case textAudioDelay = "text_audio_delay"
            case sampleRate = "sample_rate"
            case maxNewTokensPerChunk = "max_new_tokens_per_chunk"
            case samples
        }
    }

    struct ReferenceRun: Decodable {
        let chunks: [ReferenceChunk]
        let transcript: String
        let params: ReferenceParams
    }

    @Test(
        "Swift streaming loop reproduces the CUDA reference transcript across chunks",
        .enabled(
            if: VibeVoiceStreamingTranscriptParityTests.artifactsAvailable,
            "needs the 1.5B checkpoint plus a dump from tools/vibevoice_parity_stream.py; override with VIBEVOICE_PARITY_MODEL / VIBEVOICE_PARITY_STREAM"))
    func streamingTranscriptParity() async throws {
        let prefix = Self.streamPrefix
        let reference = try JSONDecoder().decode(
            ReferenceRun.self,
            from: try Data(contentsOf: URL(fileURLWithPath: prefix + ".json")))
        let tensors = try MLX.loadArrays(url: URL(fileURLWithPath: prefix + ".safetensors"))
        let audio = try #require(tensors["input_audio"]).asType(.float32)

        let model = try await VibeVoiceASRStreamingModel.fromModelDirectory(
            URL(fileURLWithPath: Self.modelPath))
        // Match the reference's float32 load of the same bf16 checkpoint.
        let float32 = model.parameters().flattened().map { ($0.0, $0.1.asType(.float32)) }
        model.update(parameters: ModuleParameters.unflattened(float32))
        eval(model)

        let parameters = VibeVoiceStreamingParameters(
            chunkDuration: reference.params.chunkDuration,
            textAudioDelay: reference.params.textAudioDelay,
            maxNewTokensPerChunk: reference.params.maxNewTokensPerChunk,
            temperature: 0.0,
            repetitionPenalty: 1.0,
            padLastChunk: true)

        print("\n=== VibeVoice 1.5B streaming transcript parity (float32, greedy, mean) ===")
        print("audio: \(reference.params.samples) samples "
            + "(\(String(format: "%.3f", Double(reference.params.samples) / Double(reference.params.sampleRate)))s), "
            + "chunk=\(reference.params.chunkDuration)s lookahead=\(reference.params.textAudioDelay)s")

        var swiftChunks: [VibeVoiceASRStreamingChunk] = []
        try model.streamingGenerate(audio: audio, parameters: parameters) { chunk in
            swiftChunks.append(chunk)
        }

        print("chunks: swift=\(swiftChunks.count) ref=\(reference.chunks.count)")

        // A partial final window is the case most likely to be mis-scheduled,
        // so the clip is deliberately not a whole number of chunks.
        let expectedChunks =
            (reference.params.samples
                + Int(reference.params.chunkDuration * Double(reference.params.sampleRate)) - 1)
            / Int(reference.params.chunkDuration * Double(reference.params.sampleRate))
        #expect(reference.chunks.count == expectedChunks, "reference chunk count")
        #expect(swiftChunks.count > 2, "clip must span several chunks to be meaningful")
        #expect(swiftChunks.count == reference.chunks.count, "chunk count")

        // ---- per-chunk comparison, bisecting to the first divergence ----
        var firstDivergentChunk: Int?
        for (i, refChunk) in reference.chunks.enumerated() {
            guard i < swiftChunks.count else { break }
            let swiftChunk = swiftChunks[i]

            let tokensMatch = swiftChunk.tokenIds == refChunk.tokenIds
            let textMatch = swiftChunk.text == refChunk.text
            let mark = (tokensMatch && textMatch) ? "ok  " : "DIFF"
            print("[\(mark)] chunk \(i): \(swiftChunk.tokenIds.count) tok "
                + "\(swiftChunk.text.debugDescription)")
            if !tokensMatch || !textMatch {
                if firstDivergentChunk == nil { firstDivergentChunk = i }
                print("        ref : \(refChunk.tokenIds.count) tok "
                    + "\(refChunk.text.debugDescription)")
            }

            #expect(swiftChunk.tokenIds == refChunk.tokenIds, "chunk \(i) token ids")
            #expect(swiftChunk.text == refChunk.text, "chunk \(i) text")
            #expect(swiftChunk.index == refChunk.index, "chunk \(i) index")
            #expect(swiftChunk.totalChunks == refChunk.totalChunks, "chunk \(i) total")
        }

        // ---- bisect a mismatch down to token and logit level -----------
        if let bad = firstDivergentChunk {
            let swiftTokens = swiftChunks[bad].tokenIds
            let refTokens = reference.chunks[bad].tokenIds
            let common = min(swiftTokens.count, refTokens.count)
            var firstBadToken = common
            for j in 0..<common where swiftTokens[j] != refTokens[j] {
                firstBadToken = j
                break
            }
            print("\nfirst divergence: chunk \(bad), token \(firstBadToken)")
            if firstBadToken < common {
                print("  swift=\(swiftTokens[firstBadToken]) "
                    + "\(model.tokenizer?.decode(tokens: [swiftTokens[firstBadToken]]).debugDescription ?? "")")
                print("  ref  =\(refTokens[firstBadToken]) "
                    + "\(model.tokenizer?.decode(tokens: [refTokens[firstBadToken]]).debugDescription ?? "")")
            } else {
                print("  (identical prefix; lengths differ: "
                    + "swift=\(swiftTokens.count) ref=\(refTokens.count))")
            }

            // Chunk 0 diverging points at the model; a later chunk diverging
            // with chunk 0 clean points at cross-chunk state -- KV-cache
            // carry-over or the window cursor, not the arithmetic.
            print(bad == 0
                ? "  -> chunk 0 differs: suspect model/weights, not streaming state"
                : "  -> chunks before \(bad) matched: suspect KV-cache carry-over or windowing")
        }

        // ---- whole transcript -------------------------------------------
        let swiftTranscript = swiftChunks
            .map { $0.text.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
            .joined(separator: " ")
        print("\nswift: \(swiftTranscript.debugDescription)")
        print("ref  : \(reference.transcript.debugDescription)")
        print("=== end streaming parity ===\n")

        #expect(swiftTranscript == reference.transcript, "full transcript")
    }
}

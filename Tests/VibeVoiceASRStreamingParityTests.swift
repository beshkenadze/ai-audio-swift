//
//  VibeVoiceASRStreamingParityTests.swift
//
//  Numeric parity of the Swift MLX port against the official PyTorch
//  implementation running on CUDA, at three points along the pipeline:
//
//    1. tokenizer encoder outputs (acoustic + semantic VAE means)
//    2. connector sum -- acoustic_connector(a) + semantic_connector(s)
//    3. logits at the first decode step, over an identical prefix
//
//  Both sides run in float32. The checkpoint is bf16, but bf16 accumulation
//  differs between CUDA and Metal by enough to hide a genuine porting bug,
//  so both sides upcast and parity is judged against the algorithm rather
//  than against a particular reduced precision.
//
//  The reference dump also carries the exact input waveform and prompt token
//  ids, so neither resampling nor tokenization can contribute a difference
//  that would be misread as a model bug.
//
//  Requires the 1.5B checkpoint and a reference dump produced by
//  `tools/vibevoice_parity_dump.py`; skips cleanly when either is absent.
//
//    VIBEVOICE_PARITY_MODEL=/path/to/streaming-1.5b \
//    VIBEVOICE_PARITY_REF=/path/to/parity_ref.safetensors \
//    swift test --filter VibeVoiceParityTests
//

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXAudioSTT

@Suite("VibeVoiceParityTests", .serialized)
struct VibeVoiceParityTests {

    // Machine-specific defaults for local runs. Absent anywhere else --
    // including CI and upstream checkouts -- so the test must SKIP rather
    // than fail when they are missing; a 5 GB checkpoint is not something a
    // normal test run can be expected to have.
    static let defaultModelPath = "/Volumes/DATA/models/vibevoice-asr-streaming-1.5b"
    static let defaultRefPath = "/Volumes/DATA/models/vibevoice-parity/parity_ref.safetensors"

    static var modelPath: String {
        ProcessInfo.processInfo.environment["VIBEVOICE_PARITY_MODEL"] ?? defaultModelPath
    }
    static var refPath: String {
        ProcessInfo.processInfo.environment["VIBEVOICE_PARITY_REF"] ?? defaultRefPath
    }

    /// Drives `.enabled(if:)`, so a missing checkpoint or reference dump
    /// reports as a skipped test instead of a red one.
    static var artifactsAvailable: Bool {
        let fm = FileManager.default
        return fm.fileExists(atPath: modelPath) && fm.fileExists(atPath: refPath)
    }

    struct Metrics {
        let maxAbs: Float
        let meanAbs: Float
        let cosine: Float
        let maxAbsIndex: Int
        let refMaxAbs: Float

        /// Difference relative to the magnitude of the reference tensor --
        /// a 1e-2 absolute gap means very different things on logits
        /// spanning ±30 and on latents spanning ±0.01.
        var relative: Float { refMaxAbs > 0 ? maxAbs / refMaxAbs : maxAbs }

        var summary: String {
            String(
                format: "max|d|=%.3e mean|d|=%.3e rel=%.3e cos=%.6f (ref max|x|=%.3e, worst @%d)",
                maxAbs, meanAbs, relative, cosine, refMaxAbs, maxAbsIndex)
        }
    }

    static func compare(_ swift: MLXArray, _ reference: MLXArray) -> Metrics {
        let a = swift.asType(.float32).flattened()
        let b = reference.asType(.float32).flattened()
        precondition(a.size == b.size, "size mismatch: \(a.size) vs \(b.size)")

        let diff = MLX.abs(a - b)
        let maxAbs = diff.max().item(Float.self)
        let meanAbs = diff.mean().item(Float.self)
        let maxIndex = MLX.argMax(diff).item(Int.self)

        let dot = (a * b).sum().item(Float.self)
        let na = MLX.sqrt((a * a).sum()).item(Float.self)
        let nb = MLX.sqrt((b * b).sum()).item(Float.self)
        let cosine = (na > 0 && nb > 0) ? dot / (na * nb) : 0

        return Metrics(
            maxAbs: maxAbs, meanAbs: meanAbs, cosine: cosine,
            maxAbsIndex: maxIndex, refMaxAbs: MLX.abs(b).max().item(Float.self))
    }

    /// Upcast every parameter to float32 so the Swift side matches the
    /// reference's `torch_dtype=float32` load of the same bf16 checkpoint.
    static func upcast(_ model: VibeVoiceASRStreamingModel) {
        let float32 = model.parameters().flattened().map { ($0.0, $0.1.asType(.float32)) }
        model.update(parameters: ModuleParameters.unflattened(float32))
        eval(model)
    }

    @Test(
        "Swift port matches the CUDA reference at encoder, connector and first logits",
        .enabled(
            if: VibeVoiceParityTests.artifactsAvailable,
            "needs the 1.5B checkpoint plus a dump from tools/vibevoice_parity_dump.py; override with VIBEVOICE_PARITY_MODEL / VIBEVOICE_PARITY_REF"))
    func parityAgainstCUDAReference() async throws {
        let modelPath = Self.modelPath
        let refPath = Self.refPath

        // Past this point the artifacts exist, so anything that still goes
        // wrong is a genuine failure rather than a missing-file skip.

        let reference = try MLX.loadArrays(url: URL(fileURLWithPath: refPath))
        for key in [
            "input_audio", "acoustic_latent", "semantic_latent", "connector_sum",
            "first_logits", "prompt_ids", "special_ids",
        ] {
            try #require(reference[key] != nil, "reference dump is missing \(key)")
        }

        let model = try await VibeVoiceASRStreamingModel.fromModelDirectory(
            URL(fileURLWithPath: modelPath))
        Self.upcast(model)

        let audio = reference["input_audio"]!.asType(.float32)
        print("\n=== VibeVoice 1.5B Swift <-> CUDA parity (float32 both sides) ===")
        print("audio: \(audio.shape)")

        // The reference resolved these from the same tokenizer; if the Swift
        // side disagrees, every downstream logit comparison is meaningless.
        let specialIds = reference["special_ids"]!.asArray(Int32.self)
        #expect(model.speechStartId == Int(specialIds[0]), "speech_start id")
        #expect(model.speechEndId == Int(specialIds[1]), "speech_end id")
        #expect(model.textChunkEndId == Int(specialIds[2]), "text_chunk_end id")
        #expect(model.eosTokenId == Int(specialIds[3]), "eos id")

        // ---- point 1: tokenizer encoders ----------------------------
        let acousticLatent = model.acousticTokenizer.encode(audio)
        let semanticLatent = model.semanticTokenizer.encode(audio)
        eval(acousticLatent, semanticLatent)

        let refAcoustic = reference["acoustic_latent"]!
        let refSemantic = reference["semantic_latent"]!
        #expect(acousticLatent.shape == refAcoustic.shape, "acoustic latent shape")
        #expect(semanticLatent.shape == refSemantic.shape, "semantic latent shape")

        let acousticMetrics = Self.compare(acousticLatent, refAcoustic)
        let semanticMetrics = Self.compare(semanticLatent, refSemantic)
        print("acoustic_latent \(acousticLatent.shape): \(acousticMetrics.summary)")
        print("semantic_latent \(semanticLatent.shape): \(semanticMetrics.summary)")

        // ---- point 2: connector sum ---------------------------------
        let connectorSum = model.encodeSpeech(audio)
        eval(connectorSum)
        let refSum = reference["connector_sum"]!
        #expect(connectorSum.shape == refSum.shape, "connector sum shape")
        let sumMetrics = Self.compare(connectorSum, refSum)
        print("connector_sum   \(connectorSum.shape): \(sumMetrics.summary)")

        if let refAcousticFeat = reference["acoustic_features"],
            let refSemanticFeat = reference["semantic_features"]
        {
            // Split the sum so a discrepancy points at one branch.
            let acousticFeat = model.acousticConnector(acousticLatent)
            let semanticFeat = model.semanticConnector(semanticLatent)
            eval(acousticFeat, semanticFeat)
            print("  acoustic_features: \(Self.compare(acousticFeat, refAcousticFeat).summary)")
            print("  semantic_features: \(Self.compare(semanticFeat, refSemanticFeat).summary)")
        }

        // ---- point 3: first-step logits -----------------------------
        // Reuse the reference's own prompt ids so tokenizer differences
        // cannot leak into this comparison.
        let promptIds = reference["prompt_ids"]!.asArray(Int32.self).map { Int($0) }
        let cache = model.makeCache()
        let promptLogits = model(inputsEmbeds: model.embed(tokenIds: promptIds), cache: cache)
        eval(promptLogits)

        if let refPromptLogits = reference["prompt_last_logits"] {
            // Isolates the decoder: this path involves no audio at all, so a
            // failure here is a pure Qwen2/embedding problem.
            let lastPromptLogits = promptLogits[0, -1, 0...][.newAxis, .ellipsis]
            print("prompt_last_logits: \(Self.compare(lastPromptLogits, refPromptLogits).summary)")
        }

        let audioEmbeds = MLX.concatenated(
            [
                model.embed(tokenIds: [model.speechStartId]),
                connectorSum,
                model.embed(tokenIds: [model.speechEndId]),
            ], axis: 1)
        let logits = model(inputsEmbeds: audioEmbeds, cache: cache)
        eval(logits)

        let firstLogits = logits[0, -1, 0...][.newAxis, .ellipsis]
        let refLogits = reference["first_logits"]!
        #expect(firstLogits.shape == refLogits.shape, "first logits shape")
        let logitMetrics = Self.compare(firstLogits, refLogits)
        print("first_logits    \(firstLogits.shape): \(logitMetrics.summary)")

        // Absolute logit deltas matter far less than whether the same token
        // wins -- that is what actually changes the transcript.
        let swiftTop = MLX.argMax(firstLogits.flattened()).item(Int.self)
        let refTop = MLX.argMax(refLogits.flattened()).item(Int.self)
        let swiftTop5 = Self.topK(firstLogits.flattened(), 5)
        let refTop5 = Self.topK(refLogits.flattened(), 5)
        print("argmax: swift=\(swiftTop) ref=\(refTop)")
        print("top5:   swift=\(swiftTop5)")
        print("        ref  =\(refTop5)")
        if let tokenizer = model.tokenizer {
            print("decoded: swift=\(tokenizer.decode(tokens: [swiftTop]).debugDescription) "
                + "ref=\(tokenizer.decode(tokens: [refTop]).debugDescription)")
        }
        print("=== end parity ===\n")

        // Thresholds are relative to each tensor's own scale. Latents and
        // features are small-magnitude, logits are not, so a single absolute
        // bound would be simultaneously too tight and too loose.
        #expect(acousticMetrics.relative < 1e-3, "acoustic latent: \(acousticMetrics.summary)")
        #expect(semanticMetrics.relative < 1e-3, "semantic latent: \(semanticMetrics.summary)")
        #expect(sumMetrics.relative < 1e-3, "connector sum: \(sumMetrics.summary)")
        #expect(logitMetrics.relative < 5e-3, "first logits: \(logitMetrics.summary)")
        #expect(logitMetrics.cosine > 0.9999, "first logits cosine: \(logitMetrics.summary)")
        #expect(swiftTop == refTop, "argmax token differs: \(swiftTop) vs \(refTop)")
        #expect(swiftTop5 == refTop5, "top-5 differs")
    }

    static func topK(_ array: MLXArray, _ k: Int) -> [Int] {
        let values = array.asType(.float32).asArray(Float.self)
        return values.enumerated()
            .sorted { $0.element > $1.element }
            .prefix(k)
            .map(\.offset)
    }
}

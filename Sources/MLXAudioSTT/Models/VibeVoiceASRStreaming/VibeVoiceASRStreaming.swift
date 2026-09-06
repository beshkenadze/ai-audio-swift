//
//  VibeVoiceASRStreaming.swift
//  MLXAudioSTT
//
//  Top-level `microsoft/VibeVoice-ASR-Streaming-{1.5B,7B}` model: acoustic +
//  semantic tokenizer encoders -> SpeechConnector projections (summed) ->
//  merged with text embeddings at speech-pad-token positions -> Qwen2
//  decoder. Ported from `mlx_audio/stt/models/vibevoice_asr/vibevoice_asr.py`
//  (Python MLX reference) and cross-checked against the official PyTorch
//  streaming implementation in `microsoft/VibeVoice`
//  (`vibevoice/modular/modeling_vibevoice_asr.py`).
//
//  NOTE ON SCOPE: this file implements weight loading and a single-window
//  (non-chunked) forward/generate path -- sufficient to validate the model
//  definition and weight mapping against the CUDA reference layer-by-layer
//  before building the full incremental `streaming_generate` loop (which
//  needs to reproduce `modeling_vibevoice_asr.py:495-538`'s overlapping
//  chunk_frames/lookahead_frames windowing with persistent Qwen2 KV-cache
//  across chunks -- see `VibeVoiceASRStreamingConfig.swift` for the exact
//  semantics). That incremental loop belongs in a follow-up
//  `StreamingTranscriber`-conforming host, not in the base model type.
//

import Foundation
import HuggingFace
import MLX
import MLXAudioCore
import MLXLMCommon
import MLXNN
import Tokenizers

// MARK: - SpeechConnector

/// Linear -> RMSNorm -> Linear projection from a tokenizer's `vaeDim` into
/// the decoder's `hiddenSize`.
final class VibeVoiceSpeechConnector: Module {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo var norm: RMSNorm
    @ModuleInfo(key: "fc2") var fc2: Linear

    init(inputDim: Int, outputDim: Int, eps: Float = 1e-6) {
        self._fc1.wrappedValue = Linear(inputDim, outputDim)
        self._norm.wrappedValue = RMSNorm(dimensions: outputDim, eps: eps)
        self._fc2.wrappedValue = Linear(outputDim, outputDim)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        fc2(norm(fc1(x)))
    }
}

// MARK: - Model

public final class VibeVoiceASRStreamingModel: Module {
    public let config: VibeVoiceASRStreamingModelConfig
    public let preprocessorConfig: VibeVoiceASRPreprocessorConfig

    @ModuleInfo(key: "acoustic_tokenizer") var acousticTokenizer: VibeVoiceAcousticTokenizerEncoder
    @ModuleInfo(key: "semantic_tokenizer") var semanticTokenizer: VibeVoiceSemanticTokenizerEncoder
    @ModuleInfo(key: "acoustic_connector") var acousticConnector: VibeVoiceSpeechConnector
    @ModuleInfo(key: "semantic_connector") var semanticConnector: VibeVoiceSpeechConnector
    @ModuleInfo(key: "language_model") var languageModel: VibeVoiceLanguageModel

    public var tokenizer: Tokenizers.Tokenizer?

    /// Resolved once the tokenizer is loaded; repurposed Qwen2.5 special
    /// tokens (matches `_build_prompt_tokens` in the Python reference).
    private var speechStartId: Int = 0
    private var speechPadId: Int = 0
    private var speechEndId: Int = 0
    private var eosTokenIds: [Int] = []

    public let sampleRate: Int

    public init(_ config: VibeVoiceASRStreamingModelConfig, preprocessorConfig: VibeVoiceASRPreprocessorConfig) {
        self.config = config
        self.preprocessorConfig = preprocessorConfig
        self.sampleRate = config.targetSampleRate

        self._acousticTokenizer.wrappedValue = VibeVoiceAcousticTokenizerEncoder(
            config: config.acousticTokenizerConfig)
        self._semanticTokenizer.wrappedValue = VibeVoiceSemanticTokenizerEncoder(
            config: config.semanticTokenizerConfig)
        self._acousticConnector.wrappedValue = VibeVoiceSpeechConnector(
            inputDim: config.acousticVaeDim, outputDim: config.decoderConfig.hiddenSize)
        self._semanticConnector.wrappedValue = VibeVoiceSpeechConnector(
            inputDim: config.semanticVaeDim, outputDim: config.decoderConfig.hiddenSize)
        self._languageModel.wrappedValue = VibeVoiceLanguageModel(config.decoderConfig)
    }

    // MARK: - Speech Encoding

    /// Deterministic acoustic+semantic feature encoding for ASR inference.
    /// Both tokenizers are always used (per the Python reference); the
    /// acoustic tokenizer's mean latent is taken directly with no Gaussian
    /// VAE sampling (see `VibeVoiceAcousticTokenizerEncoder.encode`).
    ///
    /// - Parameter speech: `[B, T]` or `[B, T, 1]` raw waveform at
    ///   `sampleRate`.
    /// - Returns: `[B, T', hiddenSize]` combined speech features.
    public func encodeSpeech(_ speech: MLXArray) -> MLXArray {
        var x = speech
        if x.ndim == 1 {
            x = x[.newAxis, .ellipsis]
        }

        let acousticLatent = acousticTokenizer.encode(x)
        let acousticFeatures = acousticConnector(acousticLatent)

        let semanticLatent = semanticTokenizer.encode(x)
        let semanticFeatures = semanticConnector(semanticLatent)

        return acousticFeatures + semanticFeatures
    }

    // MARK: - Merge Speech + Text Embeddings

    /// Inserts `speechFeatures` into `textEmbeds` at every position where
    /// `acousticInputMask` is true, in order (cumulative-sum trick, mirrors
    /// `Model._merge_speech_text_embeddings` in the Python reference).
    public func mergeSpeechTextEmbeddings(
        inputIds: MLXArray,
        speechFeatures: MLXArray?,
        acousticInputMask: MLXArray?
    ) -> MLXArray {
        let textEmbeds = languageModel.model.embedTokens(inputIds)
        guard let speechFeatures, let acousticInputMask else { return textEmbeds }

        let batchSize = textEmbeds.dim(0)
        var rows: [MLXArray] = []
        for b in 0..<batchSize {
            let maskRow = acousticInputMask[b]
            let cumsum = MLX.cumsum(maskRow.asType(.int32), axis: 0)
            var speechIdx = cumsum - MLXArray(Int32(1))
            speechIdx = MLX.clip(
                speechIdx, min: MLXArray(Int32(0)),
                max: MLXArray(Int32(speechFeatures.dim(1) - 1)))

            let gathered = MLX.take(speechFeatures[b], speechIdx, axis: 0)
            let maskExpanded = MLX.expandedDimensions(maskRow, axis: -1)
            rows.append(MLX.where(maskExpanded, gathered, textEmbeds[b]))
        }
        return MLX.stacked(rows, axis: 0)
    }

    // MARK: - Forward Pass

    public func callAsFunction(
        inputIds: MLXArray,
        speechTensors: MLXArray? = nil,
        acousticInputMask: MLXArray? = nil,
        speechFeatures: MLXArray? = nil,
        cache: [KVCache]? = nil
    ) -> MLXArray {
        var features = speechFeatures
        if let speechTensors, features == nil {
            features = encodeSpeech(speechTensors)
        }

        let inputsEmbeds: MLXArray
        // Only merge speech features on a fresh (empty) cache -- once the
        // KV-cache has advanced past the prompt/audio prefix, subsequent
        // calls feed one text token's embedding at a time and must not
        // re-run the merge (matches the Python reference's
        // `cache[0].offset > 0` short-circuit).
        if let cache, let first = cache.first, first.offset > 0 {
            inputsEmbeds = languageModel.model.embedTokens(inputIds)
        } else {
            inputsEmbeds = mergeSpeechTextEmbeddings(
                inputIds: inputIds, speechFeatures: features, acousticInputMask: acousticInputMask)
        }

        return languageModel(inputsEmbeds: inputsEmbeds, cache: cache)
    }

    public func makeCache() -> [KVCache] {
        languageModel.makeCache()
    }

    // MARK: - Weight Sanitization

    public static func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]

        for (rawKey, rawValue) in weights {
            // Drop training-only / TTS-only components entirely: the
            // diffusion head is never used at ASR-inference time, and
            // buffers like `position_ids`/`fix_std` aren't model
            // parameters.
            if rawKey.contains("diffusion_head") || rawKey.contains("prediction_head") {
                continue
            }
            if rawKey.hasSuffix("position_ids") || rawKey.hasSuffix("fix_std")
                || rawKey.contains("num_batches_tracked")
            {
                continue
            }

            var key = rawKey
            // Strip a leading "model." wrapper some checkpoints use.
            if key.hasPrefix("model.") {
                key = String(key.dropFirst("model.".count))
            }

            // Tokenizer encoder path fixups (downsample stem wrapper +
            // depthwise-mixer double-nesting), matching `Model.sanitize`
            // lines 320-350 in the Python reference.
            key = key.replacingOccurrences(
                of: #"\.downsample_layers\.(\d+)\.0\.conv\.conv\."#,
                with: ".downsample_layers.$1.conv.", options: .regularExpression)
            key = key.replacingOccurrences(
                of: #"\.head\.conv\.conv\."#, with: ".head.conv.", options: .regularExpression)
            key = key.replacingOccurrences(
                of: #"\.mixer\.conv\.conv\.conv\."#, with: ".mixer.conv.conv.",
                options: .regularExpression)

            // Language-model key remap: PyTorch's flat
            // `language_model.{layers,embed_tokens,norm}.*` needs the
            // `model.` wrapper this Swift port's `VibeVoiceLanguageModel`
            // expects, and a bare `lm_head.*` belongs under
            // `language_model.`.
            if key.hasPrefix("language_model.layers.") {
                key = "language_model.model." + key.dropFirst("language_model.".count)
            } else if key.hasPrefix("language_model.embed_tokens") {
                key = "language_model.model.embed_tokens"
                    + key.dropFirst("language_model.embed_tokens".count)
            } else if key.hasPrefix("language_model.norm") {
                key = "language_model.model.norm" + key.dropFirst("language_model.norm".count)
            }
            if key.hasPrefix("lm_head.") {
                key = "language_model." + key
            }

            var value = rawValue
            // PyTorch Conv1d weight: [out, in/groups, kernel] -> MLX
            // Conv1d weight: [out, kernel, in/groups].
            if key.lowercased().contains("conv") && key.hasSuffix("weight") && value.ndim == 3 {
                value = value.transposed(0, 2, 1)
            }

            sanitized[key] = value
        }

        return sanitized
    }

    // MARK: - Tokenizer Wiring

    /// Resolves the repurposed Qwen2.5 special-token IDs
    /// (`<|object_ref_start|>` == speech_start, `<|box_start|>` ==
    /// speech_pad, `<|object_ref_end|>` == speech_end) dynamically from the
    /// loaded tokenizer's vocab -- these IDs differ between the 1.5B
    /// (vocab_size=151936) and 7B (vocab_size=152064) checkpoints, so they
    /// must never be hardcoded.
    public func resolveSpecialTokens() {
        guard let tokenizer else { return }
        speechStartId = tokenizer.convertTokenToId("<|object_ref_start|>") ?? 0
        speechPadId = tokenizer.convertTokenToId("<|box_start|>") ?? 0
        speechEndId = tokenizer.convertTokenToId("<|object_ref_end|>") ?? 0
        let eot = tokenizer.convertTokenToId("<|endoftext|>")
        let imEnd = tokenizer.convertTokenToId("<|im_end|>")
        eosTokenIds = [eot, imEnd].compactMap { $0 }
    }

    /// Builds `(inputIds, acousticInputMask)` for a single full-context
    /// prompt: system + user message with `speechPadId` repeated once per
    /// speech-feature timestep, matching `_build_prompt_tokens` in the
    /// Python reference (JSON-transcription instruction, optional hotwords
    /// via `context`).
    public func buildPromptTokens(
        speechFeatures: MLXArray, audioDuration: Double, context: String?
    ) throws -> (inputIds: MLXArray, acousticInputMask: MLXArray) {
        guard let tokenizer else {
            throw STTModelError.unsupportedModelType("VibeVoiceASRStreaming: tokenizer not loaded")
        }

        let vaeTokLen = speechFeatures.dim(1)
        let showKeys = "Start time, End time, Speaker ID, Content"
        let suffix: String
        if let context, !context.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            suffix =
                "This is a \(String(format: "%.2f", audioDuration)) seconds audio, "
                + "with extra info: \(context.trimmingCharacters(in: .whitespacesAndNewlines))\n\n"
                + "Please transcribe it with these keys: \(showKeys)"
        } else {
            suffix =
                "This is a \(String(format: "%.2f", audioDuration)) seconds audio, "
                + "please transcribe it with these keys: \(showKeys)"
        }

        let speechStartToken = "<|object_ref_start|>"
        let speechPadToken = "<|box_start|>"
        let speechEndToken = "<|object_ref_end|>"
        let userContent =
            speechStartToken + String(repeating: speechPadToken, count: vaeTokLen)
            + speechEndToken + "\n" + suffix

        let messages: [Tokenizers.Message] = [
            ["role": "system", "content": "You are a helpful assistant that transcribes audio input into text output in JSON format."],
            ["role": "user", "content": userContent],
        ]

        let tokens = try tokenizer.applyChatTemplate(messages: messages)
        let inputIds = MLXArray(tokens.map { Int32($0) })[.newAxis, .ellipsis]
        let maskValues = tokens.map { $0 == speechPadId }
        let acousticInputMask = MLXArray(maskValues)[.newAxis, .ellipsis]

        return (inputIds, acousticInputMask)
    }

    public func isEOS(_ tokenId: Int) -> Bool { eosTokenIds.contains(tokenId) }

    // MARK: - Load from Pretrained

    public static func fromPretrained(
        _ modelPath: String,
        cache: HubCache = .default
    ) async throws -> VibeVoiceASRStreamingModel {
        guard let repoID = Repo.ID(rawValue: modelPath) else {
            throw STTModelError.invalidRepositoryID(modelPath)
        }
        let modelDir = try await ModelUtils.resolveOrDownloadModel(
            repoID: repoID,
            requiredExtension: ".safetensors",
            cache: cache
        )
        return try await fromModelDirectory(modelDir)
    }

    public static func fromModelDirectory(_ modelDir: URL) async throws -> VibeVoiceASRStreamingModel {
        let configData = try Data(contentsOf: modelDir.appendingPathComponent("config.json"))
        let config = try JSONDecoder().decode(VibeVoiceASRStreamingModelConfig.self, from: configData)

        var preprocessorConfig = VibeVoiceASRPreprocessorConfig(
            chunkFrames: 22, lookaheadFrames: 4, targetSampleRate: config.targetSampleRate,
            normalizeAudio: false, processorClass: "VibeVoiceASRProcessor")
        let preprocessorURL = modelDir.appendingPathComponent("preprocessor_config.json")
        if let preprocessorData = try? Data(contentsOf: preprocessorURL) {
            preprocessorConfig = try JSONDecoder().decode(
                VibeVoiceASRPreprocessorConfig.self, from: preprocessorData)
        }

        let model = VibeVoiceASRStreamingModel(config, preprocessorConfig: preprocessorConfig)

        model.tokenizer = try await AutoTokenizer.from(modelFolder: modelDir)
        model.resolveSpecialTokens()

        var weights: [String: MLXArray] = [:]
        let fileManager = FileManager.default
        let files = try fileManager.contentsOfDirectory(at: modelDir, includingPropertiesForKeys: nil)
        let safetensorFiles = files.filter { $0.pathExtension == "safetensors" }
        for file in safetensorFiles {
            let fileWeights = try MLX.loadArrays(url: file)
            weights.merge(fileWeights) { _, new in new }
        }

        let sanitizedWeights = VibeVoiceASRStreamingModel.sanitize(weights: weights)

        if let perLayerQuantization = config.perLayerQuantization {
            quantize(model: model) { path, _ in
                if sanitizedWeights["\(path).scales"] != nil {
                    return perLayerQuantization.quantization(layer: path)?.asTuple
                }
                return nil
            }
        }

        try model.update(
            parameters: ModuleParameters.unflattened(sanitizedWeights),
            verify: .all
        )
        eval(model)

        return model
    }
}

// MARK: - STTGenerationModel Conformance
//
// Single-window (non-chunked) greedy decode: encodes the WHOLE input clip
// through `encodeSpeech` in one pass and autoregressively decodes until EOS
// or `maxTokens`. This is the correctness baseline for CUDA-parity checks;
// the paced, chunk_frames/lookahead_frames-based incremental streaming path
// (mirroring `streaming_generate`/`streaming_generate_step` in the official
// PyTorch reference) belongs in a dedicated `StreamingTranscriber` host, not
// here -- see the file-level note at the top of this file.

extension VibeVoiceASRStreamingModel: STTGenerationModel {
    public var defaultGenerationParameters: STTGenerateParameters {
        STTGenerateParameters(
            maxTokens: 8192,
            temperature: 0.0,
            topP: 1.0,
            topK: 0,
            verbose: false,
            language: nil
        )
    }

    public func generate(audio: MLXArray, generationParameters: STTGenerateParameters) -> STTOutput {
        let start = Date()

        let speechFeatures = encodeSpeech(audio)
        let audioDuration = Double(audio.shape.last ?? 0) / Double(sampleRate)

        let (inputIds, acousticInputMask): (MLXArray, MLXArray)
        do {
            (inputIds, acousticInputMask) = try buildPromptTokens(
                speechFeatures: speechFeatures, audioDuration: audioDuration, context: nil)
        } catch {
            return STTOutput(text: "", totalTime: Date().timeIntervalSince(start))
        }

        let cache = makeCache()
        var logits = self(
            inputIds: inputIds,
            acousticInputMask: acousticInputMask,
            speechFeatures: speechFeatures,
            cache: cache
        )
        eval(logits)

        let promptTokenCount = inputIds.dim(1)
        var generatedTokens: [Int] = []

        for _ in 0..<generationParameters.maxTokens {
            let nextTokenArray = argMax(logits[0..., -1, 0...], axis: -1)
            let nextToken = nextTokenArray.item(Int.self)
            if isEOS(nextToken) { break }
            generatedTokens.append(nextToken)

            let nextIds = MLXArray([Int32(nextToken)])[.newAxis, .ellipsis]
            logits = self(inputIds: nextIds, cache: cache)
            eval(logits)
        }

        let rawText = tokenizer?.decode(tokens: generatedTokens, skipSpecialTokens: true) ?? ""
        let totalTime = Date().timeIntervalSince(start)
        let generationTps = totalTime > 0 ? Double(generatedTokens.count) / totalTime : 0

        return STTOutput(
            text: rawText.trimmingCharacters(in: .whitespacesAndNewlines),
            promptTokens: promptTokenCount,
            generationTokens: generatedTokens.count,
            totalTokens: promptTokenCount + generatedTokens.count,
            generationTps: generationTps,
            totalTime: totalTime
        )
    }

    public func generateStream(
        audio: MLXArray, generationParameters: STTGenerateParameters
    ) -> AsyncThrowingStream<STTGeneration, Error> {
        AsyncThrowingStream { continuation in
            let output = self.generate(audio: audio, generationParameters: generationParameters)
            continuation.yield(.result(output))
            continuation.finish()
        }
    }
}

extension VibeVoiceASRPreprocessorConfig {
    init(
        chunkFrames: Int, lookaheadFrames: Int, targetSampleRate: Int,
        normalizeAudio: Bool, processorClass: String
    ) {
        self.chunkFrames = chunkFrames
        self.lookaheadFrames = lookaheadFrames
        self.targetSampleRate = targetSampleRate
        self.normalizeAudio = normalizeAudio
        self.processorClass = processorClass
    }
}

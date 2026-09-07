//
//  VibeVoiceASRStreamingSession.swift
//  MLXAudioSTT
//
//  Incremental chunked ASR streaming for VibeVoice-ASR-Streaming, ported
//  from the official PyTorch reference
//  (`vibevoice/modular/modeling_vibevoice_asr.py`):
//    - `streaming_generate`      (line 429) -- offline, whole-clip
//    - `init_streaming_state`    (line 592) -- prompt prefill
//    - `streaming_generate_step` (line 630) -- one chunk against a live cache
//
//  Transcript layout the model was trained on (see `vllm_plugin/
//  asr_streaming.py:5`):
//
//      Prompt + [ <sp_start> AudioWindow <sp_end> Text <|text_chunk_end|> ] * N
//
//  Two properties of this scheme drive the whole implementation:
//
//  1. Audio windows OVERLAP. Each window spans `chunk + lookahead` samples
//     but the cursor advances only `chunk` samples, so the lookahead tail is
//     re-encoded as the head of the next window. There is deliberately NO
//     cross-chunk convolution state: each window is encoded from raw audio
//     completely independently (`split_then_encode`,
//     `modeling_vibevoice_asr.py:495-515`). The tokenizer streaming cache in
//     `modeling_vibevoice_streaming_inference.py` is TTS-only and must not
//     be used here.
//
//  2. The ONLY state carried across chunks is the decoder KV-cache. Every
//     chunk is terminated by feeding `<|text_chunk_end|>` into that cache --
//     unconditionally, including when generation stopped because the model
//     emitted EOS or hit the per-chunk token budget. Skipping it desyncs the
//     cache from the training layout and every later chunk degrades.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Parameters

public struct VibeVoiceStreamingParameters: Sendable {
    /// Seconds of *new* audio consumed per chunk (the cursor stride).
    public var chunkDuration: Double
    /// Seconds of extra right-context appended to each window. Rounded to a
    /// whole tokenizer frame, exactly as the reference does, so the window
    /// always lands on a frame boundary.
    public var textAudioDelay: Double
    public var maxNewTokensPerChunk: Int
    public var temperature: Float
    public var repetitionPenalty: Float
    /// Zero-pad a short final window up to the full window length so every
    /// chunk produces an identical feature count (reference default: true).
    public var padLastChunk: Bool
    /// Optional hotwords / domain hints spliced into the prompt.
    public var contextInfo: String?
    /// Overrides the built-in prompt entirely.
    public var promptText: String?

    public init(
        chunkDuration: Double = 2.0,
        textAudioDelay: Double = 0.5,
        maxNewTokensPerChunk: Int = 256,
        temperature: Float = 0.0,
        repetitionPenalty: Float = 1.0,
        padLastChunk: Bool = true,
        contextInfo: String? = nil,
        promptText: String? = nil
    ) {
        self.chunkDuration = chunkDuration
        self.textAudioDelay = textAudioDelay
        self.maxNewTokensPerChunk = maxNewTokensPerChunk
        self.temperature = temperature
        self.repetitionPenalty = repetitionPenalty
        self.padLastChunk = padLastChunk
        self.contextInfo = contextInfo
        self.promptText = promptText
    }

    public static let `default` = VibeVoiceStreamingParameters()
}

public struct VibeVoiceASRStreamingChunk: Sendable {
    public let index: Int
    /// Total chunk count when known up-front (offline clip), else `nil` for
    /// an open-ended live session.
    public let totalChunks: Int?
    public let text: String

    public init(index: Int, totalChunks: Int?, text: String) {
        self.index = index
        self.totalChunks = totalChunks
        self.text = text
    }
}

public enum VibeVoiceStreamingError: Error, LocalizedError {
    case tokenizerMissing
    case textChunkEndTokenMissing

    public var errorDescription: String? {
        switch self {
        case .tokenizerMissing:
            return "VibeVoice ASR streaming: tokenizer not loaded."
        case .textChunkEndTokenMissing:
            return
                "VibeVoice ASR streaming: this checkpoint's tokenizer has no "
                + "<|text_chunk_end|>, so no chunk could ever terminate. Use a "
                + "streaming checkpoint (its added_tokens.json defines the token)."
        }
    }
}

// MARK: - Window scheduling

/// The overlapping-window cursor, split out from the session so it can be
/// exercised without loading a checkpoint. Pure value type: no model, no
/// MLX, no I/O.
///
/// Reproduces `split_then_encode` (`modeling_vibevoice_asr.py:495-515`):
/// windows span `chunk + lookahead` samples, the cursor advances `chunk`
/// samples at a time, and a short final window is zero-padded back up to a
/// full window when `padLastChunk` is set.
struct VibeVoiceWindowBuffer {
    let chunkSamples: Int
    let lookaheadSamples: Int
    let padLastChunk: Bool

    var windowSamples: Int { chunkSamples + lookaheadSamples }

    /// Audio not yet fully consumed; `buffer[0]` sits at absolute sample
    /// index `cursor`. The lookahead tail intentionally survives each
    /// advance so the next, overlapping window can re-read it.
    private var buffer: [Float] = []
    private var cursor = 0
    private var totalPushed = 0
    private var flushed = false

    init(chunkSamples: Int, lookaheadSamples: Int, padLastChunk: Bool) {
        self.chunkSamples = chunkSamples
        self.lookaheadSamples = lookaheadSamples
        self.padLastChunk = padLastChunk
    }

    /// Appends samples and returns every window that is now complete.
    /// Mid-stream windows are never padded -- a window is only emitted once
    /// its full right-context has actually arrived.
    mutating func push(_ samples: [Float]) -> [[Float]] {
        guard !flushed, !samples.isEmpty else { return [] }
        buffer.append(contentsOf: samples)
        totalPushed += samples.count

        var windows: [[Float]] = []
        while buffer.count >= windowSamples {
            windows.append(Array(buffer[0..<windowSamples]))
            buffer.removeFirst(chunkSamples)
            cursor += chunkSamples
        }
        return windows
    }

    /// Drains the tail: keeps advancing until every pushed sample has been
    /// covered by some window, padding the short ones.
    mutating func flush() -> [[Float]] {
        guard !flushed else { return [] }
        flushed = true

        var windows: [[Float]] = []
        while cursor < totalPushed && !buffer.isEmpty {
            var window = Array(buffer.prefix(windowSamples))
            if padLastChunk && window.count < windowSamples {
                window.append(
                    contentsOf: [Float](repeating: 0, count: windowSamples - window.count))
            }
            windows.append(window)

            buffer.removeFirst(min(chunkSamples, buffer.count))
            cursor += chunkSamples
        }
        return windows
    }
}
// MARK: - Session

/// A live streaming ASR session: push audio as it arrives, receive chunk
/// transcripts as each window completes. Not thread-safe -- drive it from a
/// single task.
public final class VibeVoiceASRStreamSession {
    private let model: VibeVoiceASRStreamingModel
    private let parameters: VibeVoiceStreamingParameters

    private var cache: [KVCache]
    private let spStartEmbed: MLXArray
    private let spEndEmbed: MLXArray
    private let tceEmbed: MLXArray
    private let textChunkEndId: Int
    private let eosId: Int

    public let chunkSamples: Int
    public let lookaheadSamples: Int
    /// Full window length fed to the encoder: stride + right-context.
    public var windowSamples: Int { chunkSamples + lookaheadSamples }

    /// Owns all window/cursor arithmetic.
    private var windows: VibeVoiceWindowBuffer
    private var chunkIndex: Int = 0
    private var finished = false

    /// Invoked as soon as each chunk's text is decoded, before `push` /
    /// `finish` return. This -- not the returned array -- is what makes the
    /// session genuinely incremental.
    public var onChunk: ((VibeVoiceASRStreamingChunk) -> Void)?

    /// Set by the offline whole-clip path, where the window count is known
    /// in advance; `nil` for an open-ended live session.
    public var totalChunksHint: Int?

    public init(
        model: VibeVoiceASRStreamingModel,
        parameters: VibeVoiceStreamingParameters = .default
    ) throws {
        guard let tokenizer = model.tokenizer else {
            throw VibeVoiceStreamingError.tokenizerMissing
        }
        guard let tceId = model.textChunkEndId else {
            throw VibeVoiceStreamingError.textChunkEndTokenMissing
        }

        self.model = model
        self.parameters = parameters
        self.textChunkEndId = tceId
        self.eosId = model.eosTokenId

        let sampleRate = model.sampleRate
        self.chunkSamples = Int(parameters.chunkDuration * Double(sampleRate))

        // Snap the lookahead to a whole tokenizer frame
        // (`modeling_vibevoice_asr.py:499-502`): with hop=3200 @ 24 kHz a
        // frame is 2/15 s, so the 0.5 s default rounds up to 4 frames
        // (0.5333 s / 12800 samples) -- matching `lookahead_frames: 4` in
        // preprocessor_config.json.
        let frameDuration = Double(model.config.acousticTokenizerConfig.hopLength) / Double(sampleRate)
        let lookaheadSeconds = (parameters.textAudioDelay / frameDuration).rounded() * frameDuration
        self.lookaheadSamples = Int(lookaheadSeconds * Double(sampleRate))

        self.windows = VibeVoiceWindowBuffer(
            chunkSamples: self.chunkSamples,
            lookaheadSamples: self.lookaheadSamples,
            padLastChunk: parameters.padLastChunk)

        self.cache = model.makeCache()

        // Prefill the cache with the instruction prompt. Encoded WITHOUT the
        // chat template and without special tokens -- the streaming
        // checkpoints were trained on the raw string
        // (`modeling_vibevoice_asr.py:468`).
        let prompt = parameters.promptText ?? Self.defaultPrompt(contextInfo: parameters.contextInfo)
        let promptIds = tokenizer.encode(text: prompt, addSpecialTokens: false)
        let promptEmbeds = model.embed(tokenIds: promptIds)
        _ = model(inputsEmbeds: promptEmbeds, cache: cache)

        self.spStartEmbed = model.embed(tokenIds: [model.speechStartId])
        self.spEndEmbed = model.embed(tokenIds: [model.speechEndId])
        self.tceEmbed = model.embed(tokenIds: [tceId])
        eval(spStartEmbed, spEndEmbed, tceEmbed)
    }

    static func defaultPrompt(contextInfo: String?) -> String {
        let keys = "speaker, content"
        let head =
            "You are a helpful assistant that transcribes audio input into text output. "
            + "Please transcribe the following audios streamingly with these keys: \(keys)"
        if let contextInfo, !contextInfo.isEmpty {
            return head + " and extra info: \(contextInfo)\n"
        }
        return head + "\n"
    }

    // MARK: Feeding audio

    /// Appends newly captured samples and emits a transcript for every
    /// window that has become complete.
    @discardableResult
    public func push(_ samples: [Float]) -> [VibeVoiceASRStreamingChunk] {
        guard !finished, !samples.isEmpty else { return [] }
        return windows.push(samples).map { consume(window: $0) }
    }

    /// Flushes the tail: keeps advancing by `chunkSamples` until every
    /// pushed sample has been covered, zero-padding the short final windows.
    @discardableResult
    public func finish() -> [VibeVoiceASRStreamingChunk] {
        guard !finished else { return [] }
        finished = true
        return windows.flush().map { consume(window: $0) }
    }

    // MARK: One chunk

    private func consume(window: [Float]) -> VibeVoiceASRStreamingChunk {
        let audio = MLXArray(window)[.newAxis, .ellipsis]
        let features = model.encodeSpeech(audio)
        let text = step(features: features)
        let chunk = VibeVoiceASRStreamingChunk(
            index: chunkIndex, totalChunks: totalChunksHint, text: text)
        chunkIndex += 1
        onChunk?(chunk)
        return chunk
    }

    /// Mirrors `streaming_generate_step`: feed `[sp_start, features,
    /// sp_end]`, greedily decode until `<|text_chunk_end|>` / EOS / budget,
    /// then unconditionally commit `<|text_chunk_end|>` to the cache.
    func step(features: MLXArray) -> String {
        let audioEmbeds = MLX.concatenated([spStartEmbed, features, spEndEmbed], axis: 1)
        var logits = model(inputsEmbeds: audioEmbeds, cache: cache)
        eval(logits)

        var chunkTokens: [Int] = []
        for _ in 0..<parameters.maxNewTokensPerChunk {
            // `logits[0, -1, 0...]` slices into a fresh array (mlx_slice
            // allocates), so the in-place penalty update below cannot write
            // back through to `logits`. `let` is correct: MLXArray is a
            // reference type, and the subscript setter mutates the object
            // rather than the binding.
            let lastLogits = logits[0, -1, 0...]

            if parameters.repetitionPenalty != 1.0 && !chunkTokens.isEmpty {
                let ids = MLXArray(chunkTokens.map { Int32($0) })
                let previous = MLX.take(lastLogits, ids, axis: 0)
                let penalized = MLX.where(
                    previous .> 0,
                    previous / parameters.repetitionPenalty,
                    previous * parameters.repetitionPenalty)
                lastLogits[ids] = penalized
            }

            let nextToken: Int
            if parameters.temperature <= 0 {
                nextToken = argMax(lastLogits, axis: -1).item(Int.self)
            } else {
                nextToken = MLXRandom.categorical(lastLogits / parameters.temperature).item(Int.self)
            }

            if nextToken == textChunkEndId || nextToken == eosId { break }
            chunkTokens.append(nextToken)

            logits = model(inputsEmbeds: model.embed(tokenIds: [nextToken]), cache: cache)
            eval(logits)
        }

        // Always commit the chunk terminator, even when the loop broke on
        // EOS or exhausted its budget -- see the note at the top of this
        // file.
        _ = model(inputsEmbeds: tceEmbed, cache: cache)

        return Self.cleanChunkText(
            model.tokenizer?.decode(tokens: chunkTokens, skipSpecialTokens: true) ?? "")
    }

    /// `skipSpecialTokens` does not remove the repurposed framing tokens
    /// (they are ordinary added tokens here), so strip them literally, as
    /// the reference does at `modeling_vibevoice_asr.py:586-588`.
    static func cleanChunkText(_ text: String) -> String {
        var cleaned = text
        for token in [
            "<|text_chunk_end|>", "<|object_ref_start|>", "<|object_ref_end|>",
            "<|box_start|>", "<|speech_start|>", "<|speech_end|>", "<|speech_pad|>",
        ] {
            cleaned = cleaned.replacingOccurrences(of: token, with: "")
        }
        return cleaned
    }
}

// MARK: - Offline whole-clip streaming

extension VibeVoiceASRStreamingModel {
    /// Chunk-by-chunk transcription of a complete clip, invoking `onChunk`
    /// as each chunk is decoded. Equivalent to the reference
    /// `streaming_generate`.
    ///
    /// The window schedule is not re-derived here: pushing the whole clip
    /// and then flushing reproduces the reference's cursor sequence exactly
    /// (full windows while the buffer allows, then zero-padded tail
    /// windows), so both the live and the offline path go through one
    /// implementation.
    public func streamingGenerate(
        audio: MLXArray,
        parameters: VibeVoiceStreamingParameters = .default,
        onChunk: (VibeVoiceASRStreamingChunk) -> Void
    ) throws {
        let session = try VibeVoiceASRStreamSession(model: self, parameters: parameters)

        var flat = audio
        while flat.ndim > 1 { flat = flat.squeezed(axis: 0) }
        let samples = flat.asArray(Float.self)

        // The cursor advances by `chunkSamples` and stops once it reaches
        // the end, so the window count is ceil(total / chunk) -- known
        // up-front for a finite clip.
        if !samples.isEmpty {
            session.totalChunksHint =
                (samples.count + session.chunkSamples - 1) / session.chunkSamples
        }

        // The callback genuinely does not outlive this call, so borrow it
        // rather than forcing callers to hand over an escaping closure.
        withoutActuallyEscaping(onChunk) { escapable in
            session.onChunk = escapable
            session.push(samples)
            session.finish()
            session.onChunk = nil
        }
    }

    /// Convenience: run the streaming path to completion and join the chunk
    /// transcripts.
    public func streamingTranscribe(
        audio: MLXArray,
        parameters: VibeVoiceStreamingParameters = .default
    ) throws -> String {
        var pieces: [String] = []
        try streamingGenerate(audio: audio, parameters: parameters) { chunk in
            let trimmed = chunk.text.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmed.isEmpty { pieces.append(trimmed) }
        }
        return pieces.joined(separator: " ")
    }
}

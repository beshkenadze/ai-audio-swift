//
//  VibeVoiceOmniBenchHost.swift
//
//  The MLX arm of the VibeVoice ASR comparison: an omni-bench host adapter
//  that runs `VibeVoiceASRStreamingModel` natively on Apple silicon.
//
//  The CUDA arm is `tools/omni_bench_host_vibevoice/vibevoice_host.py`. The
//  two run independently on their own hardware and are compared with
//  `omni-bench parity`; neither proxies to the other, so each Result's
//  `hardware` axis describes the machine that actually ran the model and the
//  measured latency is inference rather than transport.
//
//  Determinism is the whole point of the comparison, so both arms are pinned
//  the same way: greedy decoding, and the acoustic VAE's mean instead of a
//  sample. On this side taking the mean is the only behaviour the port
//  implements; on the CUDA side the Python host forces it and asserts the
//  switch took effect.
//

import Foundation
import MLX
import MLXAudioSTT
import OmniBench

public enum VibeVoiceHostError: Error, CustomStringConvertible {
    case audioDecodeFailed(String)
    case emptyAudio(String)
    case unsupportedSampleRate(Int)

    public var description: String {
        switch self {
        case let .audioDecodeFailed(path): return "could not decode \(path)"
        case let .emptyAudio(path): return "\(path): no samples after resampling"
        case let .unsupportedSampleRate(rate): return "unsupported input rate \(rate) Hz"
        }
    }
}

/// Both `audio_transcription.v1` seams over one loaded MLX model.
public final class VibeVoiceOmniBenchHost {

    private let model: VibeVoiceASRStreamingModel
    private let parameters: VibeVoiceStreamingParameters

    public init(
        model: VibeVoiceASRStreamingModel,
        parameters: VibeVoiceStreamingParameters = .default
    ) {
        self.model = model
        self.parameters = parameters
    }

    public func capabilities() -> Capabilities {
        Capabilities(supportsTimestamps: false, supportsStreaming: true, maxConcurrency: 1)
    }

    /// One silent window through the real path, so the first scored sample is
    /// not paying Metal pipeline compilation.
    public func warmup() throws {
        let plan = VibeVoiceWindowPlan(
            chunkDuration: parameters.chunkDuration,
            textAudioDelay: parameters.textAudioDelay)
        let silence = [Float](repeating: 0, count: plan.windowSamples)
        let audio = MLXArray(silence)[.newAxis, .ellipsis]
        try model.streamingGenerate(audio: audio, parameters: parameters) { _ in }
    }
}

// MARK: - Batch

extension VibeVoiceOmniBenchHost: Transcriber {

    public func transcribe(
        _ audio: AudioInput, language: String, task: TaskContext
    ) throws -> Transcript {
        _ = language  // the model is multilingual and takes no language hint
        _ = task

        let samples = try readPCM(audio)
        let resampled = try VibeVoiceResampler.resample(samples, from: audio.sampleRateHz)
        guard !resampled.isEmpty else {
            throw VibeVoiceHostError.emptyAudio(audio.path.path)
        }

        var texts: [String] = []
        let array = MLXArray(resampled)[.newAxis, .ellipsis]
        try model.streamingGenerate(audio: array, parameters: parameters) { chunk in
            texts.append(chunk.text)
        }
        return Transcript(text: VibeVoiceHypothesis.stripSpeakerLabels(texts.joined(separator: " ")))
    }
}

// MARK: - Streaming

extension VibeVoiceOmniBenchHost: StreamingTranscriber {

    public func transcribeStream(
        _ stream: AudioChunkStream, language: String, task: TaskContext,
        emit: @escaping (String) -> Void
    ) throws -> Transcript {
        _ = language
        _ = task

        let session = try VibeVoiceASRStreamSession(model: model, parameters: parameters)
        var texts: [String] = []
        var resampler: VibeVoiceResampler.Stream?

        // The producer paces at chunk_ms (typically 100 ms) while the model
        // consumes ~2 s windows; the session buffers and cuts windows itself.
        // Every chunk must be pulled -- returning before the stream is
        // exhausted is a per-sample error -- so this loop always drains.
        for chunk in stream {
            if resampler == nil {
                resampler = try VibeVoiceResampler.Stream(sourceRate: chunk.sampleRateHz)
            }
            guard let resampler else { continue }
            let produced = resampler.push(chunk.samples)
            if produced.isEmpty { continue }
            for emitted in session.push(produced) {
                texts.append(emitted.text)
                let partial = VibeVoiceHypothesis.stripSpeakerLabels(texts.joined(separator: " "))
                if !partial.isEmpty { emit(partial) }
            }
        }

        if let resampler {
            let tail = resampler.finish()
            if !tail.isEmpty {
                for emitted in session.push(tail) {
                    texts.append(emitted.text)
                    let partial = VibeVoiceHypothesis.stripSpeakerLabels(texts.joined(separator: " "))
                    if !partial.isEmpty { emit(partial) }
                }
            }
        }
        for emitted in session.finish() {
            texts.append(emitted.text)
            let partial = VibeVoiceHypothesis.stripSpeakerLabels(texts.joined(separator: " "))
            if !partial.isEmpty { emit(partial) }
        }

        return Transcript(text: VibeVoiceHypothesis.stripSpeakerLabels(texts.joined(separator: " ")))
    }
}

// MARK: - Audio loading

extension VibeVoiceOmniBenchHost {

    /// Reads the prepared canonical PCM16 mono WAV.
    ///
    /// omni-bench has already verified the file's digest and media descriptors
    /// before calling, so this only has to decode at the declared rate; a
    /// mismatch would mean manifest and bytes disagree, which `WAVReader`
    /// reports rather than silently resampling around.
    private func readPCM(_ audio: AudioInput) throws -> [Float] {
        do {
            return try WAVReader.read(
                audio.path, sampleRateHz: audio.sampleRateHz, channels: audio.channels)
        } catch {
            throw VibeVoiceHostError.audioDecodeFailed(
                "\(audio.path.path): \(error)")
        }
    }
}

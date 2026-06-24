//  Voice activity detection tests covering Sortformer configs/features/post-processing, VAD outputs, and Smart Turn behavior.
//  Most suites are fast and local; SmartTurnNetworkTests downloads a model only when MLXAUDIO_ENABLE_NETWORK_TESTS=1.
//
//  Run the VAD suites in this file:
//    xcodebuild test \
//      -scheme MLXAudio-Package \
//      -destination 'platform=macOS' \
//      -parallel-testing-enabled NO \
//      -only-testing:MLXAudioTests/SortformerConfigTests \
//      -only-testing:MLXAudioTests/VADOutputTests \
//      -only-testing:MLXAudioTests/SortformerFeatureTests \
//      -only-testing:MLXAudioTests/SortformerSanitizeTests \
//      -only-testing:MLXAudioTests/SortformerPostprocessingTests \
//      -only-testing:MLXAudioTests/SmartTurnConfigTests \
//      -only-testing:MLXAudioTests/SmartTurnForwardTests \
//      -only-testing:MLXAudioTests/SmartTurnSanitizeTests \
//      -only-testing:MLXAudioTests/SmartTurnNetworkTests \
//      CODE_SIGNING_ALLOWED=NO
//
//  Run a single category:
//    -only-testing:'MLXAudioTests/SortformerConfigTests'
//    -only-testing:'MLXAudioTests/VADOutputTests'
//    -only-testing:'MLXAudioTests/SortformerFeatureTests'
//    -only-testing:'MLXAudioTests/SortformerSanitizeTests'
//    -only-testing:'MLXAudioTests/SortformerPostprocessingTests'
//    -only-testing:'MLXAudioTests/SmartTurnConfigTests'
//    -only-testing:'MLXAudioTests/SmartTurnForwardTests'
//    -only-testing:'MLXAudioTests/SmartTurnSanitizeTests'
//    -only-testing:'MLXAudioTests/SmartTurnNetworkTests'
//
//  Run a single test (note the trailing parentheses for Swift Testing):
//    -only-testing:'MLXAudioTests/SortformerConfigTests/fcEncoderConfigDefaults()'
//
//  Filter test results:
//    2>&1 | grep --color=never -E '(Suite.*started|Test test.*started|passed after|failed after|TEST SUCCEEDED|TEST FAILED|Suite.*passed|Test run)'

import Foundation
import Testing
import MLX
import MLXNN

@testable import MLXAudioCore
@testable import MLXAudioVAD


// MARK: - Configuration Tests

struct SortformerConfigTests {

    @Test func fcEncoderConfigDefaults() throws {
        let json = "{}"
        let data = json.data(using: .utf8)!
        let config = try JSONDecoder().decode(FCEncoderConfig.self, from: data)

        #expect(config.hiddenSize == 512)
        #expect(config.numHiddenLayers == 18)
        #expect(config.numAttentionHeads == 8)
        #expect(config.numKeyValueHeads == 8)
        #expect(config.intermediateSize == 2048)
        #expect(config.numMelBins == 80)
        #expect(config.convKernelSize == 9)
        #expect(config.subsamplingFactor == 8)
        #expect(config.subsamplingConvChannels == 256)
        #expect(config.subsamplingConvKernelSize == 3)
        #expect(config.subsamplingConvStride == 2)
        #expect(config.maxPositionEmbeddings == 5000)
        #expect(config.attentionBias == true)
        #expect(config.scaleInput == true)
    }

    @Test func fcEncoderConfigCustom() throws {
        let json = """
        {
            "hidden_size": 256,
            "num_hidden_layers": 6,
            "num_attention_heads": 4,
            "intermediate_size": 1024,
            "num_mel_bins": 40,
            "scale_input": false
        }
        """
        let data = json.data(using: .utf8)!
        let config = try JSONDecoder().decode(FCEncoderConfig.self, from: data)

        #expect(config.hiddenSize == 256)
        #expect(config.numHiddenLayers == 6)
        #expect(config.numAttentionHeads == 4)
        #expect(config.intermediateSize == 1024)
        #expect(config.numMelBins == 40)
        #expect(config.scaleInput == false)
    }

    @Test func tfEncoderConfigDefaults() throws {
        let json = "{}"
        let data = json.data(using: .utf8)!
        let config = try JSONDecoder().decode(TFEncoderConfig.self, from: data)

        #expect(config.dModel == 192)
        #expect(config.encoderLayers == 18)
        #expect(config.encoderAttentionHeads == 8)
        #expect(config.encoderFfnDim == 768)
        #expect(config.layerNormEps == 1e-5)
        #expect(config.maxSourcePositions == 1500)
        #expect(config.kProjBias == false)
    }

    @Test func tfEncoderConfigCustom() throws {
        let json = """
        {
            "d_model": 128,
            "encoder_layers": 6,
            "encoder_attention_heads": 4,
            "encoder_ffn_dim": 512,
            "k_proj_bias": true
        }
        """
        let data = json.data(using: .utf8)!
        let config = try JSONDecoder().decode(TFEncoderConfig.self, from: data)

        #expect(config.dModel == 128)
        #expect(config.encoderLayers == 6)
        #expect(config.encoderAttentionHeads == 4)
        #expect(config.encoderFfnDim == 512)
        #expect(config.kProjBias == true)
    }

    @Test func modulesConfigDefaults() throws {
        let json = "{}"
        let data = json.data(using: .utf8)!
        let config = try JSONDecoder().decode(ModulesConfig.self, from: data)

        #expect(config.numSpeakers == 4)
        #expect(config.fcDModel == 512)
        #expect(config.tfDModel == 192)
        #expect(config.subsamplingFactor == 8)
        #expect(config.chunkLen == 188)
        #expect(config.fifoLen == 0)
        #expect(config.spkcacheLen == 188)
        #expect(config.useAosc == false)
        #expect(config.silThreshold == 0.1)
    }

    @Test func processorConfigDefaults() throws {
        let json = "{}"
        let data = json.data(using: .utf8)!
        let config = try JSONDecoder().decode(ProcessorConfig.self, from: data)

        #expect(config.featureSize == 80)
        #expect(config.samplingRate == 16000)
        #expect(config.hopLength == 160)
        #expect(config.nFft == 512)
        #expect(config.winLength == 400)
        #expect(config.preemphasis == 0.97)
    }

    @Test func sortformerConfigDecoding() throws {
        let json = """
        {
            "model_type": "sortformer",
            "num_speakers": 4,
            "fc_encoder_config": {
                "hidden_size": 512,
                "num_hidden_layers": 18,
                "num_mel_bins": 80
            },
            "tf_encoder_config": {
                "d_model": 192,
                "encoder_layers": 18
            },
            "modules_config": {
                "num_speakers": 4,
                "fc_d_model": 512,
                "tf_d_model": 192
            },
            "processor_config": {
                "sampling_rate": 16000,
                "hop_length": 160
            }
        }
        """
        let data = json.data(using: .utf8)!
        let config = try JSONDecoder().decode(SortformerConfig.self, from: data)

        #expect(config.modelType == "sortformer")
        #expect(config.numSpeakers == 4)
        #expect(config.fcEncoderConfig.hiddenSize == 512)
        #expect(config.fcEncoderConfig.numHiddenLayers == 18)
        #expect(config.tfEncoderConfig.dModel == 192)
        #expect(config.tfEncoderConfig.encoderLayers == 18)
        #expect(config.modulesConfig.numSpeakers == 4)
        #expect(config.processorConfig.samplingRate == 16000)
    }

    @Test func sortformerConfigAllDefaults() throws {
        let json = "{}"
        let data = json.data(using: .utf8)!
        let config = try JSONDecoder().decode(SortformerConfig.self, from: data)

        #expect(config.modelType == "sortformer")
        #expect(config.numSpeakers == 4)
        #expect(config.fcEncoderConfig.hiddenSize == 512)
        #expect(config.tfEncoderConfig.dModel == 192)
        #expect(config.modulesConfig.fcDModel == 512)
        #expect(config.processorConfig.featureSize == 80)
    }
}

// MARK: - VADOutput Tests

struct VADOutputTests {

    @Test func diarizationSegmentCreation() {
        let segment = DiarizationSegment(start: 1.5, end: 3.0, speaker: 0)

        #expect(segment.start == 1.5)
        #expect(segment.end == 3.0)
        #expect(segment.speaker == 0)
    }

    @Test func diarizationOutputRTTMText() {
        let segments = [
            DiarizationSegment(start: 0.0, end: 1.0, speaker: 0),
            DiarizationSegment(start: 1.5, end: 2.5, speaker: 1),
        ]

        let output = DiarizationOutput(segments: segments, numSpeakers: 2)
        let text = output.text

        #expect(text.contains("speaker_0"))
        #expect(text.contains("speaker_1"))
        #expect(text.contains("SPEAKER audio 1"))
    }

    @Test func diarizationOutputEmpty() {
        let output = DiarizationOutput(segments: [])
        #expect(output.text == "")
        #expect(output.numSpeakers == 0)
    }

    @Test func streamingStateInit() {
        let embDim = 512
        let nSpk = 4
        let state = StreamingState(
            spkcache: MLXArray.zeros([1, 0, embDim]),
            spkcachePreds: MLXArray.zeros([1, 0, nSpk]),
            fifo: MLXArray.zeros([1, 0, embDim]),
            fifoPreds: MLXArray.zeros([1, 0, nSpk]),
            framesProcessed: 0,
            meanSilEmb: MLXArray.zeros([1, embDim]),
            nSilFrames: MLXArray.zeros([1])
        )

        #expect(state.spkcacheLen == 0)
        #expect(state.fifoLen == 0)
        #expect(state.framesProcessed == 0)
    }
}

// MARK: - Feature Extraction Tests

struct SortformerFeatureTests {

    @Test func preemphasisFilterShape() {
        let waveform = MLXArray.ones([16000])
        let filtered = preemphasisFilter(waveform)

        #expect(filtered.shape == waveform.shape)
    }

    @Test func preemphasisFilterFirstSample() {
        let waveform = MLXArray([1.0, 2.0, 3.0, 4.0] as [Float])
        let filtered = preemphasisFilter(waveform, coeff: 0.97)
        eval(filtered)

        // First sample should be unchanged
        let first = filtered[0].item(Float.self)
        #expect(first == 1.0)

        // Second sample: 2.0 - 0.97 * 1.0 = 1.03
        let second = filtered[1].item(Float.self)
        #expect(abs(second - 1.03) < 1e-4)
    }

    @Test func extractMelFeaturesShape() {
        // 1 second of audio at 16kHz
        let waveform = MLXRandom.normal([16000])
        let features = extractMelFeatures(waveform)
        eval(features)

        // Should be (batch=1, nMels=80, numFrames) with numFrames padded to multiple of 16
        #expect(features.ndim == 3)
        #expect(features.dim(0) == 1)
        #expect(features.dim(1) == 80)
        #expect(features.dim(2) % 16 == 0)
        #expect(features.dim(2) > 0)
    }

    @Test func extractMelFeaturesNoPad() {
        let waveform = MLXRandom.normal([16000])
        let features = extractMelFeatures(waveform, padTo: 0)
        eval(features)

        #expect(features.ndim == 3)
        #expect(features.dim(0) == 1)
        #expect(features.dim(1) == 80)
        #expect(features.dim(2) > 0)
    }

    @Test func extractMelFeaturesBatched() {
        let waveform = MLXRandom.normal([2, 16000])
        let features = extractMelFeatures(waveform)
        eval(features)

        #expect(features.ndim == 3)
        #expect(features.dim(0) == 2)
        #expect(features.dim(1) == 80)
    }

    @Test func trimSilenceNoTrim() {
        // All-speech waveform (high energy)
        let waveform = MLXRandom.normal([16000]) * 0.5
        let (trimmed, _) = trimSilence(waveform, sampleRate: 16000)
        eval(trimmed)

        #expect(trimmed.dim(0) > 0)
        // May or may not trim depending on random values, just verify it runs
    }

    @Test func trimSilenceShortAudio() {
        // Very short audio should not be trimmed
        let waveform = MLXRandom.normal([1000])
        let (trimmed, offset) = trimSilence(waveform, sampleRate: 16000)
        eval(trimmed)

        #expect(trimmed.dim(0) == 1000)
        #expect(offset == 0)
    }
}

// MARK: - Weight Sanitization Tests

struct SortformerSanitizeTests {

    @Test func sanitizeConv2dWeights() {
        // Simulate PyTorch Conv2d weights: (O, I, H, W)
        let weights: [String: MLXArray] = [
            "fc_encoder.subsampling.layers.0.weight": MLXArray.ones([256, 1, 3, 3]),
        ]

        let sanitized = SortformerModel.sanitize(weights)

        // Should rename layers.0 → layers_0 and transpose to (O, H, W, I)
        let w = sanitized["fc_encoder.subsampling.layers_0.weight"]!
        #expect(w.shape == [256, 3, 3, 1])
    }

    @Test func sanitizeConv1dWeights() {
        // Simulate PyTorch Conv1d weights: (O, I, K)
        let weights: [String: MLXArray] = [
            "fc_encoder.layers.0.conv.pointwise_conv1.weight": MLXArray.ones([1024, 512, 1]),
        ]

        let sanitized = SortformerModel.sanitize(weights)

        // Should transpose to (O, K, I)
        let w = sanitized["fc_encoder.layers.0.conv.pointwise_conv1.weight"]!
        #expect(w.shape == [1024, 1, 512])
    }

    @Test func sanitizeSkipsNumBatchesTracked() {
        let weights: [String: MLXArray] = [
            "fc_encoder.layers.0.conv.norm.num_batches_tracked": MLXArray([0]),
            "fc_encoder.layers.0.conv.norm.weight": MLXArray.ones([512]),
        ]

        let sanitized = SortformerModel.sanitize(weights)

        #expect(sanitized["fc_encoder.layers.0.conv.norm.num_batches_tracked"] == nil)
        #expect(sanitized["fc_encoder.layers.0.conv.norm.weight"] != nil)
    }

    @Test func sanitizeAlreadyConvertedPassesThrough() {
        // When weights already use layers_ format, skip conversion
        let weights: [String: MLXArray] = [
            "fc_encoder.subsampling.layers_0.weight": MLXArray.ones([256, 3, 3, 1]),
        ]

        let sanitized = SortformerModel.sanitize(weights)

        let w = sanitized["fc_encoder.subsampling.layers_0.weight"]!
        // Should NOT transpose again
        #expect(w.shape == [256, 3, 3, 1])
    }

    @Test func sanitizeDepthwiseConvWeights() {
        let weights: [String: MLXArray] = [
            "fc_encoder.layers.0.conv.depthwise_conv.weight": MLXArray.ones([512, 1, 9]),
        ]

        let sanitized = SortformerModel.sanitize(weights)

        let w = sanitized["fc_encoder.layers.0.conv.depthwise_conv.weight"]!
        #expect(w.shape == [512, 9, 1])
    }
}

// MARK: - Post-Processing Tests

struct SortformerPostprocessingTests {

    @Test func predsToSegmentsBasic() {
        // Create simple predictions: speaker 0 active for frames 0-9, speaker 1 for frames 5-14
        let nFrames = 20
        let nSpk = 2
        var predsData = [Float](repeating: 0.0, count: nFrames * nSpk)

        // Speaker 0: frames 0-9
        for i in 0..<10 { predsData[i * nSpk + 0] = 0.8 }
        // Speaker 1: frames 5-14
        for i in 5..<15 { predsData[i * nSpk + 1] = 0.9 }

        let preds = MLXArray(predsData).reshaped(nFrames, nSpk)
        let frameDuration: Float = 0.08  // 80ms per frame

        let segments = SortformerModel.predsToSegments(preds, frameDuration: frameDuration)

        #expect(segments.count >= 2)

        let speakers = Set(segments.map { $0.speaker })
        #expect(speakers.contains(0))
        #expect(speakers.contains(1))

        // All segments should have positive duration
        for seg in segments {
            #expect(seg.end > seg.start)
        }
    }

    @Test func predsToSegmentsEmpty() {
        // All predictions below threshold
        let preds = MLXArray.zeros([20, 4])
        let segments = SortformerModel.predsToSegments(preds, frameDuration: 0.08)

        #expect(segments.isEmpty)
    }

    @Test func predsToSegmentsWithMinDuration() {
        // Create a very short active region (2 frames = 0.16s)
        let nFrames = 20
        let nSpk = 2
        var predsData = [Float](repeating: 0.0, count: nFrames * nSpk)
        predsData[5 * nSpk + 0] = 0.9
        predsData[6 * nSpk + 0] = 0.9

        let preds = MLXArray(predsData).reshaped(nFrames, nSpk)

        // With minDuration = 0.5, the short segment should be filtered out
        let segments = SortformerModel.predsToSegments(
            preds, frameDuration: 0.08, minDuration: 0.5
        )
        #expect(segments.isEmpty)

        // Without minDuration, it should appear
        let segmentsNoMin = SortformerModel.predsToSegments(
            preds, frameDuration: 0.08, minDuration: 0.0
        )
        #expect(segmentsNoMin.count == 1)
    }

    @Test func predsToSegmentsWithMergeGap() {
        // Two close segments that should be merged
        let nFrames = 30
        let nSpk = 1
        var predsData = [Float](repeating: 0.0, count: nFrames * nSpk)

        // Segment 1: frames 0-4
        for i in 0..<5 { predsData[i] = 0.9 }
        // Gap: frames 5-6 (0.16s)
        // Segment 2: frames 7-14
        for i in 7..<15 { predsData[i] = 0.9 }

        let preds = MLXArray(predsData).reshaped(nFrames, nSpk)

        // Without merge: should have 2 segments
        let segmentsNoMerge = SortformerModel.predsToSegments(
            preds, frameDuration: 0.08, mergeGap: 0.0
        )
        #expect(segmentsNoMerge.count == 2)

        // With mergeGap = 0.5s: should merge into 1
        let segmentsMerged = SortformerModel.predsToSegments(
            preds, frameDuration: 0.08, mergeGap: 0.5
        )
        #expect(segmentsMerged.count == 1)
    }

    @Test func predsToSegmentsSorted() {
        // Multiple speakers — output should be sorted by start time
        let nFrames = 20
        let nSpk = 3
        var predsData = [Float](repeating: 0.0, count: nFrames * nSpk)

        // Speaker 2: early (frames 0-4)
        for i in 0..<5 { predsData[i * nSpk + 2] = 0.9 }
        // Speaker 0: middle (frames 8-12)
        for i in 8..<13 { predsData[i * nSpk + 0] = 0.9 }
        // Speaker 1: late (frames 15-19)
        for i in 15..<20 { predsData[i * nSpk + 1] = 0.9 }

        let preds = MLXArray(predsData).reshaped(nFrames, nSpk)
        let segments = SortformerModel.predsToSegments(preds, frameDuration: 0.08)

        // Should be sorted by start time
        for i in 1..<segments.count {
            #expect(segments[i].start >= segments[i - 1].start)
        }
    }

    @Test func predsToSegmentsExactBoundaries() {
        // Pin the exact frame->time mapping the bulk-readback edge detection must
        // preserve: a start edge, an inactive-close edge, and — critically — a
        // segment active through the FINAL frame (the tail branch other tests miss).
        let frameDuration: Float = 0.08
        let nFrames = 12
        let nSpk = 1
        var predsData = [Float](repeating: 0.0, count: nFrames * nSpk)
        for i in 3...7 { predsData[i] = 0.9 }          // mid segment: frames 3..7
        for i in 10..<nFrames { predsData[i] = 0.9 }   // trailing: frames 10..11 -> end

        let preds = MLXArray(predsData).reshaped(nFrames, nSpk)
        let segments = SortformerModel.predsToSegments(preds, frameDuration: frameDuration)

        #expect(segments.count == 2)

        // Mid segment: starts at frame 3, closes at the first inactive frame (8).
        #expect(abs(segments[0].start - 3 * frameDuration) < 1e-5)   // 0.24
        #expect(abs(segments[0].end - 8 * frameDuration) < 1e-5)     // 0.64

        // Trailing segment: active through the last frame -> end == nFrames * dur.
        #expect(abs(segments[1].start - 10 * frameDuration) < 1e-5)  // 0.80
        #expect(abs(segments[1].end - Float(nFrames) * frameDuration) < 1e-5)  // 0.96
    }
}

// MARK: - Smart Turn Config Tests

struct SmartTurnConfigTests {

    @Test func smartTurnConfigDefaults() throws {
        let json = "{}"
        let data = json.data(using: .utf8)!
        let cfg = try JSONDecoder().decode(SmartTurnConfig.self, from: data)

        #expect(cfg.modelType == "smart_turn")
        #expect(cfg.architecture == "smart_turn")
        #expect(cfg.dtype == "float32")
        #expect(cfg.encoderConfig.numMelBins == 80)
        #expect(cfg.processorConfig.samplingRate == 16000)
        #expect(cfg.processorConfig.maxAudioSeconds == 8)
    }

    @Test func smartTurnConfigFromDict() throws {
        let json = """
        {
            "dtype": "float16",
            "sample_rate": 22050,
            "max_audio_seconds": 6,
            "threshold": 0.42,
            "encoder_config": {
                "num_mel_bins": 8,
                "max_source_positions": 64,
                "d_model": 16,
                "encoder_attention_heads": 2,
                "encoder_layers": 1,
                "encoder_ffn_dim": 32,
                "k_proj_bias": false
            },
            "processor_config": {
                "sampling_rate": 16000,
                "max_audio_seconds": 8,
                "n_fft": 400,
                "hop_length": 160,
                "n_mels": 8,
                "normalize_audio": true,
                "threshold": 0.5
            }
        }
        """
        let data = json.data(using: .utf8)!
        let cfg = try JSONDecoder().decode(SmartTurnConfig.self, from: data)

        #expect(cfg.dtype == "float16")
        #expect(cfg.sampleRate == 22050)
        #expect(cfg.maxAudioSeconds == 6)
        #expect(abs(cfg.threshold - 0.42) < 1e-6)
        #expect(cfg.encoderConfig.dModel == 16)
        #expect(cfg.processorConfig.nMels == 8)
    }

    @Test func smartTurnSynthesizesProcessorConfig() throws {
        let json = """
        {
            "sample_rate": 24000,
            "max_audio_seconds": 5,
            "threshold": 0.33,
            "encoder_config": { "num_mel_bins": 64 }
        }
        """
        let data = json.data(using: .utf8)!
        let cfg = try JSONDecoder().decode(SmartTurnConfig.self, from: data)

        #expect(cfg.processorConfig.samplingRate == 24000)
        #expect(cfg.processorConfig.maxAudioSeconds == 5)
        #expect(cfg.processorConfig.nMels == 64)
        #expect(abs(cfg.processorConfig.threshold - 0.33) < 1e-6)
    }
}

// MARK: - Smart Turn Model Tests

private func makeTinySmartTurnConfig(dtype: String = "float32") -> SmartTurnConfig {
    let encoder = SmartTurnEncoderConfig(
        numMelBins: 8,
        maxSourcePositions: 64,
        dModel: 16,
        encoderAttentionHeads: 2,
        encoderLayers: 1,
        encoderFfnDim: 32,
        kProjBias: false
    )
    let processor = SmartTurnProcessorConfig(
        samplingRate: 16000,
        maxAudioSeconds: 8,
        nFft: 400,
        hopLength: 160,
        nMels: 8,
        normalizeAudio: true,
        threshold: 0.5
    )
    return SmartTurnConfig(dtype: dtype, encoderConfig: encoder, processorConfig: processor)
}

private func makeTinySmartTurnModel(dtype: String = "float32") throws -> SmartTurnModel {
    let model = SmartTurnModel(makeTinySmartTurnConfig(dtype: dtype))
    eval(model.parameters())

    if dtype == "float16" {
        let casted = Dictionary(
            uniqueKeysWithValues: model.parameters().flattened().map { key, value in
                (key, value.asType(.float16))
            }
        )
        try model.update(parameters: ModuleParameters.unflattened(casted), verify: .noUnusedKeys)
        eval(model.parameters())
    }

    return model
}

struct SmartTurnForwardTests {

    @Test func smartTurnForwardShapeAndRange() throws {
        let model = try makeTinySmartTurnModel()
        let input = MLXArray.zeros([1, 8, 64], type: Float.self)
        let out = model(input)
        eval(out)

        #expect(out.shape == [1, 1])
        let minVal = out.min().item(Float.self)
        let maxVal = out.max().item(Float.self)
        #expect(minVal >= 0.0)
        #expect(maxVal <= 1.0)
    }

    @Test func smartTurnForwardReturnLogits() throws {
        let model = try makeTinySmartTurnModel()
        let input = MLXArray.zeros([1, 8, 64], type: Float.self)
        let logits = model(input, returnLogits: true)
        eval(logits)

        #expect(logits.shape == [1, 1])
    }

    @Test func smartTurnForwardBatchDimension() throws {
        let model = try makeTinySmartTurnModel()
        let input = MLXArray.zeros([2, 8, 64], type: Float.self)
        let out = model(input)
        eval(out)

        #expect(out.shape == [2, 1])
    }

    @Test func smartTurnDTypePropagation() throws {
        let fp32Model = try makeTinySmartTurnModel(dtype: "float32")
        let fp32In = MLXArray.zeros([1, 8, 64], type: Float.self)
        let fp32Out = fp32Model(fp32In)
        eval(fp32Out)
        #expect(fp32Model.modelDType == .float32)
        #expect(fp32Out.dtype == .float32)

        let fp16Model = try makeTinySmartTurnModel(dtype: "float16")
        let fp16In = MLXArray.zeros([1, 8, 64], type: Float.self).asType(.float16)
        let fp16Out = fp16Model(fp16In)
        eval(fp16Out)
        #expect(fp16Model.modelDType == .float16)
        #expect(fp16Out.dtype == .float16)
    }

    @Test func smartTurnPrepareAudioArrayLengths() throws {
        let model = try makeTinySmartTurnModel()
        let maxSamples = model.config.processorConfig.maxAudioSeconds * model.config.processorConfig.samplingRate

        let short = MLXArray.ones([16000], type: Float.self)
        let shortOut = try model.prepareAudioSamples(short, sampleRate: 16000)
        #expect(shortOut.count == maxSamples)

        let long = MLXArray.ones([200000], type: Float.self)
        let longOut = try model.prepareAudioSamples(long, sampleRate: 16000)
        #expect(longOut.count == maxSamples)
    }

    @Test func smartTurnPrepareAudioArrayResample() throws {
        let model = try makeTinySmartTurnModel()
        let maxSamples = model.config.processorConfig.maxAudioSeconds * model.config.processorConfig.samplingRate

        let audio8k = MLXArray.ones([8000], type: Float.self)
        let out = try model.prepareAudioSamples(audio8k, sampleRate: 8000)
        #expect(out.count == maxSamples)
    }

    @Test func smartTurnPrepareInputFeaturesShape() throws {
        let model = try makeTinySmartTurnModel()
        let audio = MLXArray.zeros([16000], type: Float.self)
        let features = try model.prepareInputFeatures(audio, sampleRate: 16000)
        eval(features)

        #expect(features.shape == [8, 800])
    }

    @Test func smartTurnPredictEndpointReturnsOutput() throws {
        let model = try makeTinySmartTurnModel()
        let audio = MLXArray.zeros([16000], type: Float.self)
        let result = try model.predictEndpoint(audio, sampleRate: 16000, threshold: 0.5)

        #expect(result.prediction == 0 || result.prediction == 1)
        #expect(result.probability >= 0.0 && result.probability <= 1.0)
    }
}

// MARK: - Smart Turn Sanitization Tests

struct SmartTurnSanitizeTests {

    @Test func smartTurnSanitizeDropsValConstants() {
        let sanitized = SmartTurnModel.sanitize([
            "val_17": MLXArray.zeros([16, 16], type: Float.self),
            "val_123": MLXArray.zeros([1], type: Float.self)
        ])
        #expect(sanitized.isEmpty)
    }

    @Test func smartTurnSanitizeRemapsPrefixes() {
        let sanitized = SmartTurnModel.sanitize([
            "inner.classifier.0.weight": MLXArray.zeros([16, 16], type: Float.self),
            "inner.pool_attention.2.bias": MLXArray.zeros([1], type: Float.self)
        ])
        #expect(sanitized["classifier_0.weight"] != nil)
        #expect(sanitized["pool_attention_2.bias"] != nil)
    }

    @Test func smartTurnSanitizeConv1dTranspose() {
        let weights: [String: MLXArray] = [
            "encoder.conv1.weight": MLXArray.zeros([16, 8, 3], type: Float.self)
        ]
        let sanitized = SmartTurnModel.sanitize(weights)
        #expect(sanitized["encoder.conv1.weight"]?.shape == [16, 3, 8])
    }

    @Test func smartTurnSanitizeFCTransposeHeuristics() {
        let weights: [String: MLXArray] = [
            "encoder.layers.0.fc1.weight": MLXArray.zeros([16, 32], type: Float.self),
            "encoder.layers.0.fc2.weight": MLXArray.zeros([32, 16], type: Float.self),
        ]
        let sanitized = SmartTurnModel.sanitize(weights)
        #expect(sanitized["encoder.layers.0.fc1.weight"]?.shape == [32, 16])
        #expect(sanitized["encoder.layers.0.fc2.weight"]?.shape == [16, 32])
    }

    @Test func smartTurnSanitizePoolTransposeHeuristics() {
        let weights: [String: MLXArray] = [
            "pool_attention.0.weight": MLXArray.zeros([16, 256], type: Float.self),
            "pool_attention.2.weight": MLXArray.zeros([256, 1], type: Float.self),
        ]
        let sanitized = SmartTurnModel.sanitize(weights)
        #expect(sanitized["pool_attention_0.weight"]?.shape == [256, 16])
        #expect(sanitized["pool_attention_2.weight"]?.shape == [1, 256])
    }
}

// MARK: - Smart Turn Network Tests

struct SmartTurnNetworkTests {

    @Test func smartTurnFromPretrainedEvaluatesConversationalAudio() async throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MLXAUDIO_ENABLE_NETWORK_TESTS"] == "1" else {
            print("Skipping network SmartTurn test. Set MLXAUDIO_ENABLE_NETWORK_TESTS=1 to enable.")
            return
        }

        let repo = env["MLXAUDIO_SMARTTURN_REPO"] ?? "mlx-community/smart-turn-v3"
        let model = try await SmartTurnModel.fromPretrained(repo)

        let audioURLTrue = Bundle.module.url(
            forResource: "conversational_a",
            withExtension: "wav",
            subdirectory: "media"
        )!
        let (_, audioTrue) = try loadAudioArray(from: audioURLTrue, sampleRate: 16000)
        let resultTrue = try model.predictEndpoint(audioTrue, sampleRate: 16000, threshold: 0.5)
        #expect(resultTrue.prediction == 1)
        #expect(resultTrue.probability >= 0.5 && resultTrue.probability <= 1.0)

        let audioURLFalse = Bundle.module.url(
            forResource: "false-turn",
            withExtension: "wav",
            subdirectory: "media"
        )!
        let (_, audioFalse) = try loadAudioArray(from: audioURLFalse, sampleRate: 16000)
        let resultFalse = try model.predictEndpoint(audioFalse, sampleRate: 16000, threshold: 0.5)
        #expect(resultFalse.prediction == 0)
        #expect(resultFalse.probability >= 0.0 && resultFalse.probability < 0.5)
    }
}

// MARK: - Silero VAD Tests

struct SileroVADConfigTests {

    @Test func branchDefaults16k() {
        let c = SileroVADBranchConfig.default16k
        #expect(c.sampleRate == 16000)
        #expect(c.filterLength == 256)
        #expect(c.hopLength == 128)
        #expect(c.pad == 64)
        #expect(c.cutoff == 129)
        #expect(c.contextSize == 64)
        #expect(c.chunkSize == 512)
    }

    @Test func branchDefaults8k() {
        let c = SileroVADBranchConfig.default8k
        #expect(c.sampleRate == 8000)
        #expect(c.filterLength == 128)
        #expect(c.hopLength == 64)
        #expect(c.pad == 32)
        #expect(c.cutoff == 65)
        #expect(c.contextSize == 32)
        #expect(c.chunkSize == 256)
    }

    @Test func modelConfigDecodesEmptyJSON() throws {
        let json = "{}".data(using: .utf8)!
        let config = try JSONDecoder().decode(SileroVADConfig.self, from: json)
        #expect(config.modelType == "silero_vad")
        #expect(config.threshold == 0.5)
        #expect(config.branch16k.chunkSize == 512)
        #expect(config.branch8k.chunkSize == 256)
    }

    @Test func modelConfigDecodesUpstreamFormat() throws {
        let json = #"""
        {
          "model_type": "silero_vad",
          "architecture": "silero_vad",
          "dtype": "float32",
          "threshold": 0.5,
          "min_speech_duration_ms": 250,
          "min_silence_duration_ms": 100,
          "speech_pad_ms": 30,
          "branch_16k": {
            "sample_rate": 16000,
            "filter_length": 256,
            "hop_length": 128,
            "pad": 64,
            "cutoff": 129,
            "context_size": 64,
            "chunk_size": 512
          }
        }
        """#.data(using: .utf8)!
        let config = try JSONDecoder().decode(SileroVADConfig.self, from: json)
        #expect(config.minSpeechDurationMs == 250)
        #expect(config.branch16k.cutoff == 129)
        #expect(config.branch8k.chunkSize == 256)
    }
}

struct SileroVADModelTests {

    @Test func initialStateShape16k() throws {
        let model = SileroVAD(SileroVADConfig())
        let st = try model.initialState(sampleRate: 16000)
        #expect(st.sampleRate == 16000)
        #expect(st.context.shape == [1, 64])
        #expect(st.lstmState == nil)
    }

    @Test func initialStateShape8k() throws {
        let model = SileroVAD(SileroVADConfig())
        let st = try model.initialState(sampleRate: 8000)
        #expect(st.sampleRate == 8000)
        #expect(st.context.shape == [1, 32])
    }

    @Test func unsupportedSampleRateThrows() {
        let model = SileroVAD(SileroVADConfig())
        #expect(throws: SileroVADError.self) {
            _ = try model.initialState(sampleRate: 22050)
        }
    }

    @Test func feedRejectsWrongChunkSize() throws {
        let model = SileroVAD(SileroVADConfig())
        let chunk = MLXArray.zeros([1, 256], type: Float.self)
        #expect(throws: SileroVADError.self) {
            _ = try model.feed(chunk: chunk, sampleRate: 16000)
        }
    }

    @Test func probsToTimestampsAllSpeech() {
        let probs = MLXArray((0 ..< 100).map { _ in Float(0.9) })
        let ts = SileroVAD.probsToTimestamps(
            probs,
            audioLen: 100 * 512,
            sampleRate: 16000,
            threshold: 0.5,
            minSpeechDurationMs: 250,
            minSilenceDurationMs: 100,
            speechPadMs: 30
        )
        #expect(ts.count == 1)
        #expect(ts.first?.start == 0)
        #expect(ts.first?.end == 100 * 512)
    }

    @Test func probsToTimestampsAllSilence() {
        let probs = MLXArray((0 ..< 100).map { _ in Float(0.01) })
        let ts = SileroVAD.probsToTimestamps(
            probs,
            audioLen: 100 * 512,
            sampleRate: 16000,
            threshold: 0.5,
            minSpeechDurationMs: 250,
            minSilenceDurationMs: 100,
            speechPadMs: 30
        )
        #expect(ts.isEmpty)
    }

    @Test func probsToTimestampsTwoSegments() {
        var values = [Float](repeating: 0.01, count: 100)
        for i in 5 ..< 30 { values[i] = 0.9 }
        for i in 50 ..< 80 { values[i] = 0.9 }
        let probs = MLXArray(values)
        let ts = SileroVAD.probsToTimestamps(
            probs,
            audioLen: 100 * 512,
            sampleRate: 16000,
            threshold: 0.5,
            minSpeechDurationMs: 250,
            minSilenceDurationMs: 100,
            speechPadMs: 30
        )
        #expect(ts.count == 2)
    }

    @Test func sanitizeDropsValPrefixAndRemapsBranchKeys() {
        let weights: [String: MLXArray] = [
            "vad_16k.lstm.Wx": MLXArray.zeros([4]),
            "vad_8k.conv1.bias": MLXArray.zeros([4]),
            "val_loss": MLXArray.zeros([1]),
            "val_acc": MLXArray.zeros([1]),
        ]
        let cleaned = SileroVAD.sanitize(weights: weights)
        #expect(cleaned.keys.contains("branch16k.lstm.Wx"))
        #expect(cleaned.keys.contains("branch8k.conv1.bias"))
        #expect(!cleaned.keys.contains("vad_16k.lstm.Wx"))
        #expect(!cleaned.keys.contains("vad_8k.conv1.bias"))
        #expect(!cleaned.keys.contains("val_loss"))
        #expect(!cleaned.keys.contains("val_acc"))
    }
}

struct SileroVADNetworkTests {

    @Test func loadV5AndPredictOnSilence() async throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MLXAUDIO_ENABLE_NETWORK_TESTS"] == "1" else {
            print("Skipping network SileroVAD test. Set MLXAUDIO_ENABLE_NETWORK_TESTS=1 to enable.")
            return
        }

        let repo = env["MLXAUDIO_SILEROVAD_REPO"] ?? "mlx-community/silero-vad"
        let model = try await SileroVAD.fromPretrained(repo)
        let silence = MLXArray.zeros([16000], type: Float.self)
        let probs = try model.predictProba(silence, sampleRate: 16000)
        eval(probs)
        let arr = probs.asArray(Float.self)
        #expect(arr.count > 0)
        #expect(arr.allSatisfy { $0 < 0.5 })
    }

    @Test func parityWithPythonReferenceOnRealAudio() async throws {
        let env = ProcessInfo.processInfo.environment
        let audioPath = env["MLXAUDIO_SILEROVAD_AUDIO"]
            ?? "/tmp/playback-eng-16k_slice.wav"
        let refPath = env["MLXAUDIO_SILEROVAD_REF"]
            ?? "/tmp/silero_vad_python_ref.json"
        let audioURL = URL(fileURLWithPath: audioPath)
        let refURL = URL(fileURLWithPath: refPath)
        guard FileManager.default.fileExists(atPath: audioURL.path),
              FileManager.default.fileExists(atPath: refURL.path) else {
            print("Skipping parity test. Audio or reference file missing at \(audioPath) / \(refPath).")
            return
        }

        let (sr, full) = try loadAudioArray(from: audioURL, sampleRate: 16000)
        #expect(sr == 16000)
        let limit = 5 * sr
        let totalSamples = full.shape[0]
        let take = min(limit, totalSamples)
        let audio5s = full[0 ..< take]
        eval(audio5s)

        let refData = try Data(contentsOf: refURL)
        struct Ref: Decodable {
            let probs: [Float]
            let n_probs: Int
            let max: Float
            let mean: Float
            let timestamps: [Stamp]
            struct Stamp: Decodable { let start: Int; let end: Int }
        }
        let ref = try JSONDecoder().decode(Ref.self, from: refData)

        let repo = env["MLXAUDIO_SILEROVAD_REPO"] ?? "mlx-community/silero-vad"
        let model = try await SileroVAD.fromPretrained(repo)
        let probsMx = try model.predictProba(audio5s, sampleRate: 16000)
        eval(probsMx)
        let probs = probsMx.asArray(Float.self)

        #expect(probs.count == ref.n_probs)
        var maxDelta: Float = 0
        for i in 0 ..< min(ref.probs.count, probs.count) {
            let d = abs(probs[i] - ref.probs[i])
            if d > maxDelta { maxDelta = d }
        }
        print("parity max|Δ| over first \(ref.probs.count) probs = \(maxDelta)")
        #expect(maxDelta < 1e-3)

        let ts = try model.getSpeechTimestamps(audio5s, sampleRate: 16000)
        #expect(ts.count == ref.timestamps.count)
        for (a, b) in zip(ts, ref.timestamps) {
            #expect(a.start == b.start)
            #expect(a.end == b.end)
        }
    }
}

// MARK: - FIFO Cap (D1) Tests

struct SortformerFifoCapTests {

    /// AOSC v2.1 / 4-spk modules config, decoded from a JSON literal (no memberwise init exists).
    private func loadSortformerModulesConfigAOSC() throws -> ModulesConfig {
        let json = """
        {
          "num_speakers": 4,
          "fc_d_model": 512,
          "tf_d_model": 192,
          "subsampling_factor": 8,
          "chunk_len": 188,
          "fifo_len": 188,
          "spkcache_len": 188,
          "spkcache_update_period": 188,
          "chunk_left_context": 1,
          "chunk_right_context": 1,
          "use_aosc": true
        }
        """
        let data = json.data(using: .utf8)!
        return try JSONDecoder().decode(ModulesConfig.self, from: data)
    }

    /// Build a StreamingState whose FIFO holds `fifoFrames` frames (spkcache empty).
    /// Uses small varied (non-zero) preds so the AOSC compression path is not degenerate.
    private func makeState(fifoFrames: Int, embDim: Int, nSpk: Int) -> StreamingState {
        let fifo = MLXRandom.uniform(low: -1.0, high: 1.0, [1, fifoFrames, embDim])
        let fifoPreds = MLXRandom.uniform(low: 0.0, high: 1.0, [1, fifoFrames, nSpk])
        return StreamingState(
            spkcache: MLXArray.zeros([1, 0, embDim]),
            spkcachePreds: MLXArray.zeros([1, 0, nSpk]),
            fifo: fifo,
            fifoPreds: fifoPreds,
            framesProcessed: 0,
            meanSilEmb: MLXArray.zeros([1, embDim]),
            nSilFrames: MLXArray.zeros([1])
        )
    }

    @Test func testFifoCappedAfterOversizedChunk() throws {
        let cfg = try loadSortformerModulesConfigAOSC()
        let embDim = cfg.fcDModel
        let nSpk = cfg.numSpeakers
        // FIFO overflow (700) >> spkcacheUpdatePeriod (188): the old code left it > fifoMax.
        let state = makeState(fifoFrames: 700, embDim: embDim, nSpk: nSpk)
        let out = SortformerModel.maybeCompressState(
            state, spkcacheMax: 188, fifoMax: 188, modulesCfg: cfg
        )
        #expect(out.fifoLen <= 188, "FIFO must be capped regardless of overflow size")
        #expect(out.spkcacheLen <= 188, "spkcache must stay capped")
    }
}

// MARK: - Bounded Window Planner (frame accounting)

/// Pure `Int` frame-accounting math for the bounded long-form streaming window.
/// No MLX/Metal required — exercises `BoundedWindowPlanner` directly.
struct BoundedWindowPlannerTests {

    /// v2.1 / 4-spk constants. `chunkLenEmb = chunkLen = 188`, halos = 16 mel frames each.
    private func makePlanner() -> BoundedWindowPlanner {
        BoundedWindowPlanner(
            hop: 160,
            nFft: 512,
            winLength: 400,
            subsamplingFactor: 8,
            chunkLenEmb: 188,
            haloLeftMel: 16,
            haloRightMel: 16
        )
    }

    /// `C` = chunkLenEmb * subsamplingFactor = NEW mel frames consumed per interior step.
    private var stepMelFrames: Int { 188 * 8 } // 1504

    /// Mel frames a full-file `extractMelFeatures` produces for `n` samples
    /// (constant pad `nFft/2` both sides; `numFrames = 1 + n/hop`).
    private func melFramesForSamples(_ n: Int) -> Int { 1 + n / 160 }

    @Test func haloInvariants() {
        let p = makePlanner()
        #expect(p.haloLeftMel % p.subsamplingFactor == 0, "H_L must align to the embedding grid")
        #expect(p.haloRightMel % p.subsamplingFactor == 0, "H_R must align to the embedding grid")
    }

    @Test func interiorStep() {
        let p = makePlanner()
        let totalSamples = 16000 * 60 // 60 s — plenty of room either side
        let g0 = stepMelFrames * 2    // an interior step, far from both ends
        let spec = p.plan(g0: g0, totalKnownSamples: totalSamples, eof: false)

        #expect(spec.rawStart % p.hop == 0, "rawStart must be hop-aligned")
        #expect(spec.discardLeftEmb == 2, "interior discards H_L/8 = 2 left-halo emb frames")
        #expect(spec.chunkEmbCount == 188, "interior chunk emits chunkLenEmb frames")
        #expect(spec.rcEmbCount == 1, "interior right-context = 1 emb frame")
        // window = left halo + chunk + right halo (all mel frames)
        #expect(spec.melWindowStart == g0 - p.haloLeftMel, "window starts one left-halo before g0")
        #expect(spec.melWindowFrameCount == p.haloLeftMel + stepMelFrames + p.haloRightMel,
                "window spans H_L + C + H_R mel frames")
        #expect(spec.newMelFrames == stepMelFrames, "interior step consumes exactly C new mel frames")
        // raw range actually contains the window's mel-frame centers
        #expect(spec.rawStart == spec.melWindowStart * p.hop, "hop-aligned to window start, no pad offset")
        #expect(spec.rawEnd <= totalSamples, "must not read past known samples")
        #expect(spec.rawEnd > spec.rawStart, "non-empty raw range")
    }

    @Test func firstStep() {
        let p = makePlanner()
        let totalSamples = 16000 * 60
        let spec = p.plan(g0: 0, totalKnownSamples: totalSamples, eof: false)

        #expect(spec.discardLeftEmb == 0, "first step has no left halo to discard")
        #expect(spec.rawStart == 0, "first step starts at sample 0")
        #expect(spec.melWindowStart == 0, "first step window begins at global mel frame 0")
        #expect(spec.chunkEmbCount == 188, "first step still emits a full chunk")
        #expect(spec.rcEmbCount == 1, "right context available")
        #expect(spec.newMelFrames == stepMelFrames)
    }

    /// No drift: walking contiguous steps over a known total must consume every NEW mel
    /// frame in `[0, totalMelFrames)` exactly once — no gap, no overlap — and the chunk
    /// emb-frame counts must sum to the full-file embedding-frame count.
    @Test func noDriftAcrossContiguousSteps() {
        let p = makePlanner()
        // A total that is NOT a multiple of C, so the final EOF step is partial.
        let totalSamples = 16000 * 91 + 137
        let totalMel = melFramesForSamples(totalSamples)

        var consumed = [Int](repeating: 0, count: totalMel)
        var sumChunkEmb = 0
        var g0 = 0
        var guardSteps = 0
        while g0 < totalMel {
            let eof = g0 + stepMelFrames >= totalMel
            let spec = p.plan(g0: g0, totalKnownSamples: totalSamples, eof: eof)
            #expect(spec.newMelFrames > 0, "every step must make progress")
            for k in g0..<(g0 + spec.newMelFrames) {
                consumed[k] += 1
            }
            sumChunkEmb += spec.chunkEmbCount
            g0 += spec.newMelFrames
            guardSteps += 1
            #expect(guardSteps < 10_000, "step loop must terminate")
        }

        #expect(consumed.allSatisfy { $0 == 1 }, "each NEW mel frame consumed exactly once")
        // full-file embedding frame count = floor((L-1)/2)+1 applied 3x
        var fullEmb = totalMel
        for _ in 0..<3 { fullEmb = (fullEmb - 1) / 2 + 1 }
        #expect(sumChunkEmb == fullEmb, "chunk emb counts telescope to the full-file emb count")
    }

    /// EOF step with no future frames: right-context shrinks to 0 and the chunk emits only
    /// the remaining new frames; nothing is read past `totalKnownSamples`.
    @Test func eofStepNoFuture() {
        let p = makePlanner()
        // total = one full step + 8 extra mel frames => second step is a tiny partial chunk.
        let extraMel = 8
        let totalSamples = (stepMelFrames + extraMel - 1) * p.hop // melFramesForSamples => C + 8
        let totalMel = melFramesForSamples(totalSamples)
        #expect(totalMel == stepMelFrames + extraMel, "fixture sets up a C+8 mel total")

        let g0 = stepMelFrames
        let spec = p.plan(g0: g0, totalKnownSamples: totalSamples, eof: true)

        #expect(spec.newMelFrames == extraMel, "chunk processes only the remaining new mel frames")
        #expect(spec.rcEmbCount == 0, "no future frames => rc shrinks to min(1, 0) = 0")
        #expect(spec.rawEnd <= totalSamples, "must not read past known samples at EOF")
        #expect(spec.melWindowStart + spec.melWindowFrameCount <= totalMel,
                "window must not extend past the known mel frames")
    }

    /// EOF-adjacent step where some future frames remain but fewer than H_R: rc must be
    /// `min(1, availableFutureEmb)` and stay 1 when at least one future emb frame exists.
    @Test func eofStepPartialFuture() {
        let p = makePlanner()
        // total = C + 4 mel frames: step 1 has a full chunk but only 4 future mel frames (< H_R=16).
        let futureMel = 4
        let totalSamples = (stepMelFrames + futureMel - 1) * p.hop
        let totalMel = melFramesForSamples(totalSamples)
        #expect(totalMel == stepMelFrames + futureMel)

        let spec = p.plan(g0: 0, totalKnownSamples: totalSamples, eof: false)
        #expect(spec.newMelFrames == stepMelFrames, "step 1 still consumes a full C")
        #expect(spec.rcEmbCount == 1, "4 future mel frames yield 1 emb frame => rc = min(1, 1) = 1")
        #expect(spec.rawEnd <= totalSamples)
    }
}

// MARK: - PCM Accumulator (bounded buffer)

struct PCMAccumulatorTests {

    @Test func sliceSpansMultipleBlocks() {
        var acc = PCMAccumulator()
        acc.append([0, 1, 2])
        acc.append([3, 4, 5])
        acc.append([6, 7, 8])
        // slice(2, 7) crosses all three blocks.
        #expect(acc.slice(from: 2, to: 7) == [2, 3, 4, 5, 6])
    }

    /// Absolute addressing survives a drop: indices stay anchored to the whole stream, so a slice
    /// of the kept tail still returns the right values. (Slicing below `k` would trap by contract —
    /// asserted in a comment rather than crash-tested.)
    @Test func absoluteAddressingSurvivesDrop() {
        var acc = PCMAccumulator()
        acc.append([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
        let k = 4
        acc.drop(beforeAbsolute: k)
        #expect(acc.firstRetainedAbsolute == k)
        // slice(4, 8) addresses the SAME absolute samples [14,15,16,17] as before the drop.
        #expect(acc.slice(from: k, to: k + 4) == [14, 15, 16, 17])
        // slice(from: k - 1, ...) would trap: samples before k are dropped. Contract-asserted only.
    }

    /// drop frees memory: retainedCount reflects only the kept tail, not the whole stream.
    @Test func dropFreesMemory() {
        var acc = PCMAccumulator()
        acc.append(Array(repeating: 0, count: 100))
        #expect(acc.retainedCount == 100)
        acc.drop(beforeAbsolute: 70)
        #expect(acc.retainedCount == 30, "only the tail [70,100) is retained")
        #expect(acc.totalAppendedCount == 100, "total seen is unaffected by drops")
    }

    /// drop is idempotent and re-droppable: dropping at k, then again below/at k, must not corrupt
    /// state; dropping nothing (index <= base) is a no-op.
    @Test func dropIsIdempotentAndReDroppable() {
        var acc = PCMAccumulator()
        acc.append(Array((0..<20).map { Float($0) }))
        acc.drop(beforeAbsolute: 8)
        #expect(acc.firstRetainedAbsolute == 8)
        #expect(acc.retainedCount == 12)

        // Re-drop at the same index: no change.
        acc.drop(beforeAbsolute: 8)
        #expect(acc.firstRetainedAbsolute == 8)
        #expect(acc.retainedCount == 12)

        // Drop below the current base (already gone): no-op, no corruption.
        acc.drop(beforeAbsolute: 3)
        #expect(acc.firstRetainedAbsolute == 8)
        #expect(acc.retainedCount == 12)

        // Drop at 0 / negative: no-op.
        acc.drop(beforeAbsolute: 0)
        #expect(acc.firstRetainedAbsolute == 8)

        // Data is still correct and absolute-addressed after the churn.
        #expect(acc.slice(from: 8, to: 12) == [8, 9, 10, 11])
    }

    /// Bounded-memory scenario: mimic the Task-5 sliding loop — each iteration appends one step of
    /// samples, slices a window (step + a left halo), then drops everything consumed below the next
    /// step's halo. retainedCount must stay <= (step + halo) and NOT grow with the iteration count.
    @Test func boundedMemoryAcrossManyIterations() {
        let step = 1504   // C = chunkLen * subsamplingFactor (~15 s of mel-frame samples, scaled down)
        let halo = 256     // left halo + STFT margin retained for the next window
        var acc = PCMAccumulator()
        var g0 = 0
        var retainedHigh = 0

        for i in 0..<50 {
            // 1. Append one step of fresh PCM (values encode absolute index for slice correctness).
            let block = (0..<step).map { Float(g0 + $0) }
            acc.append(block)

            // 2. Slice the window: left halo (clamped at stream start) + this step.
            let winStart = max(0, g0 - halo)
            let winEnd = g0 + step
            let win = acc.slice(from: winStart, to: winEnd)
            #expect(win.first == Float(winStart), "iter \(i): window starts at the right absolute sample")
            #expect(win.last == Float(winEnd - 1), "iter \(i): window ends at the right absolute sample")

            // 3. Advance, then drop everything the NEXT window won't need (keep only its left halo).
            g0 += step
            acc.drop(beforeAbsolute: max(0, g0 - halo))

            retainedHigh = max(retainedHigh, acc.retainedCount)
        }

        // Memory is flat: bounded by one step + one halo, regardless of the 50 iterations / ~75k samples.
        #expect(retainedHigh <= step + halo,
                "retained memory must stay <= one step + halo (got \(retainedHigh))")
        #expect(acc.totalAppendedCount == step * 50, "all samples were seen")
    }
}

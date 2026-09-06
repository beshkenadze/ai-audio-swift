//
//  VibeVoiceASRStreamingAudioEncoder.swift
//  MLXAudioSTT
//
//  Acoustic/Semantic tokenizer encoders: a causal depthwise-separable conv
//  stack (7 downsample stages, ratios [8,5,5,4,2,2], depths [3,3,3,3,3,3,8])
//  that turns raw 24kHz audio into a latent sequence. Ported from
//  `mlx_audio/stt/models/vibevoice_asr/audio_encoder.py` (Python MLX
//  reference), itself a port of the acoustic/semantic tokenizer used by
//  upstream `microsoft/VibeVoice`.
//
//  IMPORTANT (parity-critical): at ASR-inference time only the
//  deterministic mean latent is used -- `Model.encode_speech()` in the
//  Python reference calls `tokenizer.encode(...)` directly, never
//  `tokenizer.sample(...)` / `__call__`. `AcousticTokenizerEncoder.encode`
//  below therefore returns the mean with NO Gaussian noise added; do not
//  wire up VAE sampling on this path or CUDA parity breaks (any RNG draw
//  desyncs Swift vs. Python/CUDA outputs).
//

import Foundation
import MLX
import MLXNN

// MARK: - Causal SConv1d

/// Causal (or symmetric) 1-D convolution with manual padding, mirroring the
/// Python `SConv1d`. MLX's `Conv1d` weight layout is `[out, kernel, in/groups]`
/// (vs. PyTorch's `[out, in/groups, kernel]`) -- the transpose is handled at
/// weight-load time in `VibeVoiceASRStreamingModel.sanitize`, not here.
final class VibeVoiceSConv1d: Module {
    let inChannels: Int
    let outChannels: Int
    let kernelSize: Int
    let stride: Int
    let dilation: Int
    let groups: Int
    let causal: Bool
    let paddingTotal: Int

    @ModuleInfo var conv: Conv1d

    init(
        inChannels: Int,
        outChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        dilation: Int = 1,
        groups: Int = 1,
        bias: Bool = true,
        causal: Bool = true
    ) {
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.kernelSize = kernelSize
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.causal = causal
        self.paddingTotal = (kernelSize - 1) * dilation - (stride - 1)

        self._conv.wrappedValue = Conv1d(
            inputChannels: inChannels,
            outputChannels: outChannels,
            kernelSize: kernelSize,
            stride: stride,
            padding: 0,
            dilation: dilation,
            groups: groups,
            bias: bias
        )
    }

    /// Extra right-side padding needed so the strided output length lines up
    /// exactly (mirrors Python `_get_extra_padding_for_conv1d`).
    private func extraPadding(length: Int) -> Int {
        let nFrames = (Double(length - kernelSize + paddingTotal) / Double(stride)) + 1
        let idealLength = (Int(nFrames.rounded(.up)) - 1) * stride + (kernelSize - paddingTotal)
        return max(0, idealLength - length)
    }

    /// x: [B, T, C_in] -> [B, T', C_out]
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let length = x.dim(1)
        let extra = extraPadding(length: length)

        let padLeft: Int
        let padRight: Int
        if causal {
            padLeft = paddingTotal
            padRight = extra
        } else {
            let right = paddingTotal / 2
            padLeft = paddingTotal - right
            padRight = right + extra
        }

        var padded = x
        if padLeft > 0 || padRight > 0 {
            padded = MLX.padded(x, widths: [.init((0, 0)), .init((padLeft, padRight)), .init((0, 0))])
        }
        return conv(padded)
    }
}

// MARK: - ConvRMSNorm (channel-wise RMSNorm over [B, T, C])

/// RMSNorm over the channel (last) axis. The Python reference transposes
/// [B,C,T] <-> [B,T,C] around a standard RMSNorm; this port keeps the MLX
/// convolution's native [B,T,C] layout throughout, so no transpose is
/// needed here -- the norm is simply applied over the last axis.
final class VibeVoiceConvRMSNorm: Module {
    let eps: Float
    let elementwiseAffine: Bool

    @ParameterInfo var weight: MLXArray?

    init(dim: Int, eps: Float = 1e-5, elementwiseAffine: Bool = true) {
        self.eps = eps
        self.elementwiseAffine = elementwiseAffine
        self._weight.wrappedValue = elementwiseAffine ? MLXArray.ones([dim]) : nil
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let variance = MLX.mean(x * x, axis: -1, keepDims: true)
        var out = x * MLX.rsqrt(variance + eps)
        if let weight {
            out = out * weight
        }
        return out
    }
}

// MARK: - Depthwise-conv mixer

/// Depthwise-separable causal conv used as the token mixer inside `Block1D`.
/// `groups == channels`, i.e. one filter per channel.
final class VibeVoiceDepthwiseConv: Module {
    @ModuleInfo var conv: VibeVoiceSConv1d

    init(dim: Int, kernelSize: Int, bias: Bool = true, causal: Bool = true) {
        self._conv.wrappedValue = VibeVoiceSConv1d(
            inChannels: dim,
            outChannels: dim,
            kernelSize: kernelSize,
            stride: 1,
            groups: dim,
            bias: bias,
            causal: causal
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { conv(x) }
}

// MARK: - FFN

final class VibeVoiceTokenizerFFN: Module {
    @ModuleInfo(key: "linear1") var linear1: Linear
    @ModuleInfo(key: "linear2") var linear2: Linear

    init(dim: Int, hiddenDim: Int, bias: Bool = true) {
        self._linear1.wrappedValue = Linear(dim, hiddenDim, bias: bias)
        self._linear2.wrappedValue = Linear(hiddenDim, dim, bias: bias)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        linear2(gelu(linear1(x)))
    }
}

// MARK: - Block1D (pre-norm mixer + FFN transformer-style block)

final class VibeVoiceBlock1D: Module {
    @ModuleInfo var norm: RMSNorm
    @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm
    @ModuleInfo var mixer: VibeVoiceDepthwiseConv
    @ModuleInfo var ffn: VibeVoiceTokenizerFFN

    let hasLayerScale: Bool
    @ParameterInfo var gamma: MLXArray?
    @ParameterInfo(key: "ffn_gamma") var ffnGamma: MLXArray?

    init(
        dim: Int,
        kernelSize: Int = 7,
        eps: Float = 1e-5,
        layerScaleInitValue: Float = 1e-6,
        convBias: Bool = true,
        causal: Bool = true
    ) {
        self._norm.wrappedValue = RMSNorm(dimensions: dim, eps: eps)
        self._ffnNorm.wrappedValue = RMSNorm(dimensions: dim, eps: eps)
        self._mixer.wrappedValue = VibeVoiceDepthwiseConv(
            dim: dim, kernelSize: kernelSize, bias: convBias, causal: causal)
        self._ffn.wrappedValue = VibeVoiceTokenizerFFN(dim: dim, hiddenDim: dim * 4, bias: convBias)

        self.hasLayerScale = layerScaleInitValue > 0
        if hasLayerScale {
            self._gamma.wrappedValue = MLXArray.ones([dim]) * layerScaleInitValue
            self._ffnGamma.wrappedValue = MLXArray.ones([dim]) * layerScaleInitValue
        } else {
            self._gamma.wrappedValue = nil
            self._ffnGamma.wrappedValue = nil
        }
    }

    /// x: [B, T, C]
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = mixer(norm(x))
        if let gamma { h = h * gamma }
        var out = x + h

        h = ffn(ffnNorm(out))
        if let ffnGamma { h = h * ffnGamma }
        out = out + h
        return out
    }
}

// MARK: - TokenizerEncoder (shared backbone for acoustic + semantic)

/// Progressive-downsample conv encoder: stem (kernel=7, stride=1) + N
/// strided downsample stages, each followed by `depths[i]` transformer-style
/// `Block1D`s, then a head projection to `vaeDim`. Total downsample factor
/// == product(ratios) == the tokenizer's `hopLength` (3200 for the released
/// checkpoints).
final class VibeVoiceTokenizerEncoder: Module {
    let ratios: [Int]
    let depths: [Int]
    let nStages: Int
    let hopLength: Int

    @ModuleInfo(key: "downsample_layers") var downsampleLayers: [VibeVoiceSConv1d]
    // Type-erased so the nested array is registered for weight loading (MLXNN
    // only recurses into `[Module]`/`[[Module]]`, not `[ConcreteClass]]`);
    // mirrors the pattern used by `KokoroTextEncoder.cnn` elsewhere in this repo.
    @ModuleInfo var stages: [[Module]]
    @ModuleInfo var head: VibeVoiceSConv1d

    init(config: VibeVoiceTokenizerConfig) {
        // Encoding order reverses the config's (decoder-oriented) ratio list.
        let encodeRatios = Array(config.encoderRatios.reversed())
        let depths = config.parsedEncoderDepths
        let nFilters = config.encoderNFilters

        self.ratios = encodeRatios
        self.depths = depths
        self.nStages = depths.count
        self.hopLength = config.encoderRatios.reduce(1, *)

        var downsamples: [VibeVoiceSConv1d] = []
        downsamples.append(
            VibeVoiceSConv1d(
                inChannels: config.channels,
                outChannels: nFilters,
                kernelSize: 7,
                stride: 1,
                bias: config.convBias,
                causal: config.causal
            ))
        for i in 0..<encodeRatios.count {
            let inCh = nFilters * (1 << i)
            let outCh = nFilters * (1 << (i + 1))
            downsamples.append(
                VibeVoiceSConv1d(
                    inChannels: inCh,
                    outChannels: outCh,
                    kernelSize: encodeRatios[i] * 2,
                    stride: encodeRatios[i],
                    bias: config.convBias,
                    causal: config.causal
                ))
        }
        self._downsampleLayers.wrappedValue = downsamples

        var stages: [[Module]] = []
        for i in 0..<nStages {
            let inCh = i == 0 ? nFilters : nFilters * (1 << i)
            var blocks: [Module] = []
            for _ in 0..<depths[i] {
                blocks.append(
                    VibeVoiceBlock1D(
                        dim: inCh,
                        kernelSize: 7,
                        eps: config.layernormEps,
                        layerScaleInitValue: config.layerScaleInitValue,
                        convBias: config.convBias,
                        causal: config.causal
                    ))
            }
            stages.append(blocks)
        }
        self._stages.wrappedValue = stages

        // Channel count after the LAST strided downsample layer, i.e. after
        // `encodeRatios.count` doublings (6 for the released checkpoints) --
        // NOT `nStages` (7, the number of Block1D stage groups, one of which
        // has no preceding extra doubling). Matches the Python reference's
        // `n_filters * (2 ** len(self.ratios))` in audio_encoder.py:548.
        let finalChannels = nFilters * (1 << ratios.count)
        self._head.wrappedValue = VibeVoiceSConv1d(
            inChannels: finalChannels,
            outChannels: config.vaeDim,
            kernelSize: 7,
            stride: 1,
            bias: config.convBias,
            causal: config.causal
        )
    }

    /// audio: [B, T] or [B, T, 1] raw waveform -> [B, T', vaeDim] latent.
    func callAsFunction(_ audio: MLXArray) -> MLXArray {
        var x = audio
        if x.ndim == 2 {
            x = x[.ellipsis, .newAxis]
        }

        for i in 0..<nStages {
            x = downsampleLayers[i](x)
            for block in stages[i] {
                x = (block as! VibeVoiceBlock1D)(x)
            }
        }
        x = head(x)
        return x
    }
}

// MARK: - Acoustic Tokenizer Encoder (Gaussian VAE mean-only at inference)

final class VibeVoiceAcousticTokenizerEncoder: Module {
    @ModuleInfo var encoder: VibeVoiceTokenizerEncoder

    init(config: VibeVoiceTokenizerConfig) {
        self._encoder.wrappedValue = VibeVoiceTokenizerEncoder(config: config)
    }

    /// Deterministic mean latent -- matches the Python reference's
    /// `Model.encode_speech()` calling `tokenizer.encode(...)` directly.
    /// Never call `.sample()`/add noise on the ASR inference path: doing so
    /// would make output nondeterministic and break CUDA/Swift parity.
    func encode(_ audio: MLXArray) -> MLXArray {
        encoder(audio)
    }
}

// MARK: - Semantic Tokenizer Encoder (no sampling ever; direct encode)

final class VibeVoiceSemanticTokenizerEncoder: Module {
    @ModuleInfo var encoder: VibeVoiceTokenizerEncoder

    init(config: VibeVoiceTokenizerConfig) {
        self._encoder.wrappedValue = VibeVoiceTokenizerEncoder(config: config)
    }

    func encode(_ audio: MLXArray) -> MLXArray {
        encoder(audio)
    }
}

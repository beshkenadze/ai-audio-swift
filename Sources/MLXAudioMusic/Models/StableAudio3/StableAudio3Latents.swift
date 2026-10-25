import AVFoundation
import Foundation
import MLX

extension StableAudio3 {
    public struct LatentAudio {
        public let latents: MLXArray
        public let variant: Variant
        public let sampleCount: Int
        public var seconds: Double { Double(sampleCount) / Double(StableAudio3.sampleRate) }

        public init(latents: MLXArray, variant: Variant, sampleCount: Int) throws {
            guard sampleCount > 0, latents.dtype == .float16, latents.ndim == 3,
                  latents.dim(0) == 1, latents.dim(1) == 256, latents.dim(2) > 0,
                  sampleCount <= latents.dim(2) * StableAudio3.samplesPerLatent,
                  sampleCount >= (latents.dim(2) - 1) * StableAudio3.samplesPerLatent,
                  all(isFinite(latents)).item(Bool.self) else {
                throw CocoaError(.fileReadCorruptFile)
            }
            self.latents = latents
            self.variant = variant
            self.sampleCount = sampleCount
        }

        public init(contentsOf url: URL) throws {
            let arrays = try MLX.loadArrays(url: url)
            guard Set(arrays.keys) == ["schema", "variant", "sample_count", "latents"],
                  let schema = arrays["schema"], schema.shape == [1], schema.dtype == .int32, schema.item(Int32.self) == 1,
                  let model = arrays["variant"], model.shape == [1], model.dtype == .int32,
                  let count = arrays["sample_count"], count.shape == [1], count.dtype == .int64,
                  let latents = arrays["latents"], (0...1).contains(model.item(Int32.self)) else {
                throw CocoaError(.fileReadCorruptFile)
            }
            try self.init(latents: latents, variant: model.item(Int32.self) == 0 ? .smallMusic : .medium,
                          sampleCount: Int(count.item(Int64.self)))
        }

        public func save(to url: URL) throws {
            try MLX.save(arrays: ["schema": MLXArray([Int32(1)]),
                                  "variant": MLXArray([Int32(variant == .smallMusic ? 0 : 1)]),
                                  "sample_count": MLXArray([Int64(sampleCount)]), "latents": latents], url: url)
        }

        public func segment(length: Int, offset: Int, variant: Variant) throws -> MLXArray {
            guard variant == self.variant, length > 0, offset >= 0, offset < latents.dim(2),
                  length <= latents.dim(2) - offset else {
                throw CocoaError(.validationMissingMandatoryProperty)
            }
            return latents[.ellipsis, offset..<(offset + length)]
        }
    }

    public static func latentOffset(seconds: Double) throws -> Int {
        let position = seconds * Double(sampleRate) / Double(samplesPerLatent)
        guard position.isFinite, position >= 0, position < Double(Int32.max) else {
            throw CocoaError(.validationMissingMandatoryProperty)
        }
        return Int(position.rounded(.toNearestOrEven))
    }

    public func encodeAudioFile(_ audio: URL, seconds: Double? = nil, offset: Double = 0,
                                progress: (Stage) -> Void = { _ in }) throws -> LatentAudio {
        let duration: Double
        if let seconds {
            duration = seconds
        } else {
            let file = try AVAudioFile(forReading: audio)
            duration = Double(file.length) / file.processingFormat.sampleRate - offset
        }
        guard duration.isFinite, duration > 0, duration <= Double(Int32.max) / Double(Self.sampleRate),
              offset.isFinite, offset >= 0, Self.sampleCount(seconds: duration) > 0 else {
            throw CocoaError(.validationMissingMandatoryProperty)
        }
        progress(.encodingAudio)
        defer { Memory.clearCache() }
        let count = Self.latentLength(seconds: duration)
        let length = Self.encoderLatentLength(count, variant: variant)
        let weights = weightsDirectory.appendingPathComponent(variant.encoderWeights)
        let encoded: MLXArray
        if count <= Self.latentLength(seconds: 120) {
            let input = try Self.loadInitAudio(audio, frames: length * Self.samplesPerLatent, offset: offset)
            encoded = try Self.encodeAudio(Self.patch(input), weights: weights, variant: variant)[.ellipsis, ..<count]
                .asType(.float16)
        } else {
            encoded = try encodeLongAudio(audio, length: length, weights: weights, offset: offset)[.ellipsis, ..<count]
        }
        eval(encoded)
        return try LatentAudio(latents: encoded, variant: variant, sampleCount: Self.sampleCount(seconds: duration))
    }

    private func encodeLongAudio(_ audio: URL, length: Int, weights: URL, offset: Double) throws -> MLXArray {
        let reader = try StableAudio3AudioInput(url: audio, offset: offset)
        let small = variant == .smallMusic ? try Self.loadSmallEncoder(weights) : nil
        let medium = variant == .medium ? try Self.loadMediumEncoder(weights) : nil
        var pcm = MLXArray.zeros([1, 2, 0])
        var previousLeft = 0
        var previousRight = 0
        var pieces: [MLXArray] = []
        for start in stride(from: 0, to: length, by: 128) {
            let end = min(length, start + 128)
            let left = max(0, start - 16)
            let right = min(length, end + 16)
            let retained = pcm[.ellipsis, ((left - previousLeft) * Self.samplesPerLatent)...]
            let next = right == previousRight ? MLXArray.zeros([1, 2, 0])
                : try reader.read(frames: (right - previousRight) * Self.samplesPerLatent)
            pcm = concatenated([retained, next], axis: 2)
            eval(pcm)
            let patches = Self.patch(pcm)
            let output: MLXArray
            if let small { output = small(patches) }
            else if let (model, stream) = medium {
                output = model(patches, stream: stream, chunk: nil, positionOffset: left)
            } else { preconditionFailure("missing audio encoder") }
            let piece = output[.ellipsis, (start - left)..<(end - left)].asType(.float16)
            eval(piece)
            pieces.append(piece)
            previousLeft = left
            previousRight = right
        }
        return concatenated(pieces, axis: -1)
    }
}

import AVFoundation
import Foundation
import MLX
import Testing

@testable import MLXAudioMusic

@Suite(.serialized)
struct StableAudio3AudioInputTests {
    private func audioFile(channels: [[Float]], rate: Double = 44_100) throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("\(UUID().uuidString).caf")
        let layout = try #require(AVAudioChannelLayout(layoutTag: kAudioChannelLayoutTag_DiscreteInOrder | UInt32(channels.count)))
        let format = try #require(AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: rate,
                                               interleaved: false, channelLayout: layout))
        let buffer = try #require(AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(channels[0].count)))
        buffer.frameLength = buffer.frameCapacity
        for channel in channels.indices {
            channels[channel].withUnsafeBufferPointer {
                buffer.floatChannelData![channel].update(from: $0.baseAddress!, count: $0.count)
            }
        }
        let file = try AVAudioFile(forWriting: url, settings: format.settings)
        try file.write(from: buffer)
        return url
    }

    @Test func patchAndUnpatchPreserveEveryBit() {
        let audio = MLXArray((0..<48).map(Float.init), [2, 2, 12])
        let patches = StableAudio3.patch(audio, patchSize: 4)
        #expect(patches.shape == [2, 8, 3])
        #expect(patches[0, 0].asArray(Float.self) == [0, 4, 8])
        #expect(patches[0, 4].asArray(Float.self) == [12, 16, 20])
        #expect(bitwiseEqual(audio, StableAudio3.unpatch(patches, patchSize: 4)))
    }

    @Test func encoderRoundsOnlyTheSmallGrid() {
        #expect(StableAudio3.encoderLatentLength(109, variant: .smallMusic) == 110)
        #expect(StableAudio3.encoderLatentLength(110, variant: .smallMusic) == 110)
        #expect(StableAudio3.encoderLatentLength(109, variant: .medium) == 109)
        for variant in StableAudio3.Variant.allCases {
            #expect(!variant.requiredFiles.contains(variant.encoderWeights))
        }
    }

    @Test func sigmaOneKeepsTheTextToAudioPathBitIdentical() {
        let noise = MLXRandom.normal([1, 256, 5], dtype: .float16, key: MLXRandom.key(7))
        let initial = MLXArray.full(noise.shape, values: MLXArray(Float.nan), dtype: .float16)
        let mixed = StableAudio3.initialNoise(noise, initLatents: initial, sigma: 1)
        #expect(bitwiseEqual(noise, mixed))
        let sigmas = StableAudio3.schedule(steps: 3)
        func sample(_ x: MLXArray) -> MLXArray {
            StableAudio3.sample(noise: x, sigmas: sigmas, seed: 8) { x, _ in x * 0.2 }
        }
        #expect(bitwiseEqual(sample(noise), sample(mixed)))
    }

    @Test func noiseMixUsesHalfPrecision() {
        let noise = MLXArray([Float(1), -2]).asType(.float16)
        let initial = MLXArray([Float(-1), 4]).asType(.float16)
        let mixed = StableAudio3.initialNoise(noise, initLatents: initial, sigma: 0.5)
        #expect(mixed.dtype == .float16)
        #expect(mixed.asArray(Float.self) == [0, 1])
    }

    @Test(arguments: [Float(0.3), 0.5, 0.7, 0.9])
    func schedulePreservesTheReferenceReNoising(sigma: Float) {
        let schedule = StableAudio3.schedule(steps: 8, sigmaMax: sigma).asArray(Float.self)
        #expect(schedule[0] == sigma)
        #expect(schedule[1] > schedule[0])
        #expect(schedule.last == 0)
        let expected: [Float: [Float]] = [
            0.3: [0.300, 0.538, 0.461, 0.386, 0.316, 0.254, 0.200, 0.155, 0],
            0.5: [0.500, 0.830, 0.746, 0.637, 0.512, 0.386, 0.274, 0.184, 0],
            0.7: [0.700, 0.954, 0.909, 0.830, 0.705, 0.538, 0.362, 0.217, 0],
            0.9: [0.900, 0.989, 0.972, 0.932, 0.844, 0.683, 0.461, 0.254, 0],
        ]
        for (actual, expected) in zip(schedule, expected[sigma]!) { #expect(abs(actual - expected) < 0.0006) }
    }

    @Test func monoDuplicatesAndPadsWithZeros() throws {
        let url = try audioFile(channels: [[0.1, -0.3, 0.7]])
        let audio = try StableAudio3.loadInitAudio(url, frames: 8)
        #expect(audio.shape == [1, 2, 8])
        #expect(audio[0, 0].asArray(Float.self) == [0.1, -0.3, 0.7, 0, 0, 0, 0, 0])
        #expect(bitwiseEqual(audio[0, 0], audio[0, 1]))
    }

    @Test func pcm16UsesTheReference32767Scale() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("\(UUID().uuidString).wav")
        var data = Data()
        func append<T: FixedWidthInteger>(_ value: T) {
            withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) }
        }
        data.append(contentsOf: "RIFF".utf8); append(UInt32(44))
        data.append(contentsOf: "WAVEfmt ".utf8); append(UInt32(16)); append(UInt16(1))
        append(UInt16(2)); append(UInt32(44_100)); append(UInt32(176_400))
        append(UInt16(4)); append(UInt16(16))
        data.append(contentsOf: "data".utf8); append(UInt32(8))
        for value in [Int16(32767), -32768, 16384, -16384] { append(value) }
        try data.write(to: url)
        let audio = try StableAudio3.loadInitAudio(url, frames: 2)
        let expected = MLXArray([Float(32767) / 32767, Float(16384) / 32767,
                                 Float(-32768) / 32767, Float(-16384) / 32767], [1, 2, 2])
        #expect(abs(audio - expected).max().item(Float.self) < 2e-7)
    }

    @Test func extraChannelsAreIgnoredAndOffsetSeeks() throws {
        let url = try audioFile(channels: [[0, 0.1, 0.2, 0.3], [0, -0.1, -0.2, -0.3], [1, 1, 1, 1]])
        let audio = try StableAudio3.loadInitAudio(url, frames: 2, offset: 2.0 / 44_100)
        #expect(audio[0, 0].asArray(Float.self) == [0.2, 0.3])
        #expect(audio[0, 1].asArray(Float.self) == [-0.2, -0.3])
        let pastEnd = try StableAudio3.loadInitAudio(url, frames: 3, offset: 5)
        #expect(pastEnd.asArray(Float.self) == [Float](repeating: 0, count: 6))
    }

    @Test func resamplingPreservesToneAndStereo() throws {
        let rate = 48_000.0
        let left = (0..<48_000).map { Float(sin(2 * Double.pi * 440 * Double($0) / rate)) * 0.5 }
        let url = try audioFile(channels: [left, left.map { -$0 }], rate: rate)
        let audio = try StableAudio3.loadInitAudio(url, frames: 44_100)
        let samples = audio[0, 0].asArray(Float.self)
        let expected = (1000..<43_000).map { Float(sin(2 * Double.pi * 440 * Double($0) / 44_100)) * 0.5 }
        let rms = sqrt(zip(samples[1000..<43_000], expected).reduce(Float(0)) { $0 + pow($1.0 - $1.1, 2) } / Float(expected.count))
        #expect(rms < 0.001)
        #expect(abs(audio[0, 0] + audio[0, 1]).max().item(Float.self) < 1e-6)
    }

    @Test(arguments: [0.7, 2.3])
    func offsetDoesNotLoseOneFrameToFloatingPointRounding(offset: Double) throws {
        let frame = Int((offset * 44_100).rounded(.toNearestOrEven))
        var samples = [Float](repeating: 0, count: frame + 2)
        samples[frame - 1] = -0.5
        samples[frame] = 0.75
        let url = try audioFile(channels: [samples])
        let audio = try StableAudio3.loadInitAudio(url, frames: 1, offset: offset)
        #expect(audio[0, 0, 0].item(Float.self) == 0.75)
    }
}

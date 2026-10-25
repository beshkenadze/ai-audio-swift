import AVFoundation
import Foundation
import MLX

extension StableAudio3 {
    /// Reads only the requested segment; mono is duplicated and extra channels are ignored.
    static func loadInitAudio(_ url: URL, frames: Int, offset: Double = 0) throws -> MLXArray {
        try StableAudio3AudioInput(url: url, offset: offset).read(frames: frames)
    }
}

/// Keeps the resampler's phase and any unused output across bounded reads.
final class StableAudio3AudioInput {
    private let file: AVAudioFile
    private let source: AVAudioFormat
    private let converter: AVAudioConverter
    private let input: AVAudioPCMBuffer
    private let selected: AVAudioPCMBuffer
    private let output: AVAudioPCMBuffer
    private let scale: Float
    private var outputPosition = 0
    private var ended = false

    init(url: URL, offset: Double = 0) throws {
        guard offset.isFinite, offset >= 0 else {
            throw CocoaError(.validationMissingMandatoryProperty)
        }
        file = try AVAudioFile(forReading: url, commonFormat: .pcmFormatFloat32, interleaved: false)
        source = file.processingFormat
        guard let stereo = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                         sampleRate: source.sampleRate, channels: 2, interleaved: false),
              let target = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                         sampleRate: Double(StableAudio3.sampleRate), channels: 2, interleaved: false),
              let converter = AVAudioConverter(from: stereo, to: target),
              let input = AVAudioPCMBuffer(pcmFormat: source, frameCapacity: 8192),
              let selected = AVAudioPCMBuffer(pcmFormat: stereo, frameCapacity: 8192),
              let output = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: 4096) else {
            throw CocoaError(.fileReadCorruptFile)
        }
        let position = (offset * source.sampleRate).rounded(.toNearestOrEven)
        file.framePosition = position >= Double(file.length) ? file.length : AVAudioFramePosition(position)
        // CoreAudio maps int16 to /32768; the MLX WAV oracle uses /32767.
        self.converter = converter
        self.input = input
        self.selected = selected
        self.output = output
        output.frameLength = 0
        scale = file.fileFormat.commonFormat == .pcmFormatInt16 ? 32768.0 / 32767.0 : 1
    }

    func read(frames: Int) throws -> MLXArray {
        guard frames > 0 else { throw CocoaError(.validationMissingMandatoryProperty) }
        var samples = [Float](repeating: 0, count: frames * 2)
        var written = 0
        var readError: Error?
        while written < frames {
            if outputPosition == Int(output.frameLength), !ended {
                var error: NSError?
                let status = converter.convert(to: output, error: &error) { [self] requested, status in
                    guard file.framePosition < file.length else {
                        status.pointee = .endOfStream
                        return nil
                    }
                    do {
                        try file.read(into: input, frameCount: min(8192, requested))
                    } catch {
                        readError = error
                        status.pointee = .endOfStream
                        return nil
                    }
                    guard input.frameLength > 0 else {
                        status.pointee = .endOfStream
                        return nil
                    }
                    selected.frameLength = input.frameLength
                    for channel in 0..<2 {
                        let sourceChannel = min(channel, Int(source.channelCount) - 1)
                        for frame in 0..<Int(input.frameLength) {
                            selected.floatChannelData![channel][frame] = input.floatChannelData![sourceChannel][frame] * scale
                        }
                    }
                    status.pointee = .haveData
                    return selected
                }
                if let readError { throw readError }
                if let error { throw error }
                outputPosition = 0
                ended = status == .endOfStream
                if status == .error || (output.frameLength == 0 && !ended) { throw CocoaError(.fileReadCorruptFile) }
            }
            let count = min(Int(output.frameLength) - outputPosition, frames - written)
            for channel in 0..<2 {
                samples.withUnsafeMutableBufferPointer { destination in
                    destination.baseAddress!.advanced(by: channel * frames + written)
                        .update(from: output.floatChannelData![channel].advanced(by: outputPosition), count: count)
                }
            }
            written += count
            outputPosition += count
            if count == 0 && ended { break }
        }
        return MLXArray(samples, [1, 2, frames])
    }
}

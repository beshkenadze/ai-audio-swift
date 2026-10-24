import AVFoundation
import Foundation
import MLX

extension StableAudio3 {
    /// Reads only the requested segment; mono is duplicated and extra channels are ignored.
    static func loadInitAudio(_ url: URL, frames: Int, offset: Double = 0) throws -> MLXArray {
        guard frames > 0, offset.isFinite, offset >= 0 else {
            throw CocoaError(.validationMissingMandatoryProperty)
        }
        let file = try AVAudioFile(forReading: url, commonFormat: .pcmFormatFloat32, interleaved: false)
        let source = file.processingFormat
        guard let stereo = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                         sampleRate: source.sampleRate, channels: 2, interleaved: false),
              let target = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                         sampleRate: Double(sampleRate), channels: 2, interleaved: false),
              let converter = AVAudioConverter(from: stereo, to: target),
              let input = AVAudioPCMBuffer(pcmFormat: source, frameCapacity: 8192),
              let selected = AVAudioPCMBuffer(pcmFormat: stereo, frameCapacity: 8192),
              let output = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: 4096) else {
            throw CocoaError(.fileReadCorruptFile)
        }
        let position = (offset * source.sampleRate).rounded(.toNearestOrEven)
        file.framePosition = position >= Double(file.length) ? file.length : AVAudioFramePosition(position)
        // CoreAudio maps int16 to /32768; the MLX WAV oracle uses /32767.
        let pcm16 = file.fileFormat.commonFormat == .pcmFormatInt16
        let scale: Float = pcm16 ? 32768.0 / 32767.0 : 1
        var samples = [Float](repeating: 0, count: frames * 2)
        var written = 0
        var readError: Error?
        while written < frames {
            var error: NSError?
            let status = converter.convert(to: output, error: &error) { requested, status in
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
            let count = min(Int(output.frameLength), frames - written)
            for channel in 0..<2 {
                samples.withUnsafeMutableBufferPointer { destination in
                    destination.baseAddress!.advanced(by: channel * frames + written)
                        .update(from: output.floatChannelData![channel], count: count)
                }
            }
            written += count
            if status == .endOfStream { break }
            guard status != .error, count > 0 else { throw CocoaError(.fileReadCorruptFile) }
        }
        return MLXArray(samples, [1, 2, frames])
    }
}

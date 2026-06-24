import AVFoundation
import Foundation
@preconcurrency import MLX
import MLXAudioCore
import MLXAudioVAD

// MARK: - Errors

enum AppError: Error, LocalizedError, CustomStringConvertible {
    case inputFileNotFound(String)
    case mlxRuntimeNotConfigured(String)
    case audioConverterUnavailable(String)
    case audioReadFailed(String)

    var errorDescription: String? { description }

    var description: String {
        switch self {
        case .inputFileNotFound(let path):
            "Input audio file not found: \(path)"
        case .mlxRuntimeNotConfigured(let detail):
            "MLX command-line runtime is not configured: \(detail)"
        case .audioConverterUnavailable(let detail):
            "Could not create the AVAudioConverter: \(detail)"
        case .audioReadFailed(let detail):
            "Failed to read audio: \(detail)"
        }
    }
}

enum CLIError: Error, CustomStringConvertible {
    case missingValue(String)
    case unknownOption(String)
    case invalidValue(String, String)

    var description: String {
        switch self {
        case .missingValue(let key):
            "Missing value for \(key)"
        case .unknownOption(let key):
            "Unknown option \(key)"
        case .invalidValue(let key, let value):
            "Invalid value for \(key): \(value)"
        }
    }
}

// MARK: - CLI

struct CLI {
    let inputPath: String
    let repo: String
    let chunkDuration: Float
    let threshold: Float
    let rttmOut: String?
    let maxSeconds: Double?
    let selfCheck: Bool
    let verbose: Bool

    static func parse() throws -> CLI {
        try parse(Array(CommandLine.arguments.dropFirst()))
    }

    static func parse(_ arguments: [String]) throws -> CLI {
        var inputPath: String?
        var repo = "mlx-community/diar_streaming_sortformer_4spk-v2.1-fp16"
        var chunkDuration: Float = 15.0
        var threshold: Float = 0.5
        var rttmOut: String?
        var maxSeconds: Double?
        var selfCheck = false
        var verbose = false

        var iterator = arguments.makeIterator()
        while let arg = iterator.next() {
            switch arg {
            case "--input", "-i":
                guard let value = iterator.next() else { throw CLIError.missingValue(arg) }
                inputPath = value
            case "--repo":
                guard let value = iterator.next() else { throw CLIError.missingValue(arg) }
                repo = value
            case "--chunk-duration":
                guard let value = iterator.next() else { throw CLIError.missingValue(arg) }
                guard let parsed = Float(value), parsed > 0 else { throw CLIError.invalidValue(arg, value) }
                chunkDuration = parsed
            case "--threshold":
                guard let value = iterator.next() else { throw CLIError.missingValue(arg) }
                guard let parsed = Float(value), parsed >= 0, parsed <= 1 else { throw CLIError.invalidValue(arg, value) }
                threshold = parsed
            case "--rttm-out":
                guard let value = iterator.next() else { throw CLIError.missingValue(arg) }
                rttmOut = value
            case "--max-seconds":
                guard let value = iterator.next() else { throw CLIError.missingValue(arg) }
                guard let parsed = Double(value), parsed > 0 else { throw CLIError.invalidValue(arg, value) }
                maxSeconds = parsed
            case "--self-check":
                selfCheck = true
            case "--verbose", "-v":
                verbose = true
            case "--help", "-h":
                printUsage()
                exit(0)
            default:
                if inputPath == nil, !arg.hasPrefix("-") {
                    inputPath = arg
                } else {
                    throw CLIError.unknownOption(arg)
                }
            }
        }

        guard let finalInput = inputPath, !finalInput.isEmpty else {
            throw CLIError.missingValue("--input")
        }

        return CLI(
            inputPath: finalInput,
            repo: repo,
            chunkDuration: chunkDuration,
            threshold: threshold,
            rttmOut: rttmOut,
            maxSeconds: maxSeconds,
            selfCheck: selfCheck,
            verbose: verbose
        )
    }

    static func printUsage() {
        let executable = (CommandLine.arguments.first as NSString?)?.lastPathComponent ?? "mlx-audio-swift-diar"
        print(
            """
            Usage:
              \(executable) --input <path> [options]

            Description:
              Memory-bounded long-form streaming speaker diarization using the Sortformer
              model. Decodes/resamples the input incrementally (AVAudioFile + AVAudioConverter
              to 16 kHz mono Float32) and drives SortformerModel.generateStreamBounded, so peak
              memory and per-chunk latency stay flat regardless of file duration.

            Options:
              --input, -i <path>         Input audio file (required, any format AVFoundation reads)
              --repo <id>                Hugging Face repo id.
                                         Default: mlx-community/diar_streaming_sortformer_4spk-v2.1-fp16
              --chunk-duration <sec>     Streaming step size in seconds. Default: 15
              --threshold <f>            Speaker activity threshold in [0,1]. Default: 0.5
              --rttm-out <path>          Optional path to write RTTM segments
              --max-seconds <n>          Optional: stop feeding after N seconds of audio (smoke runs)
              --self-check               Regression guard: load a SHORT clip fully and compare the
                                         precompute path (generateStream) against the bounded path
                                         (generateStreamBounded) on identical samples; print per-slot
                                         and overall frame-agreement. Use with --max-seconds to bound
                                         the clip (e.g. --max-seconds 150 --chunk-duration 15).
              --verbose, -v              Print per-chunk progress
              --help, -h                 Show this help

            Examples:
              \(executable) --input recording.flac --rttm-out out.rttm
              \(executable) --input recording.flac --max-seconds 60 --chunk-duration 15 --verbose
              \(executable) --self-check --input clip.flac --max-seconds 150 --chunk-duration 15 -v
            """
        )
    }
}

// MARK: - Windowed audio reader (pull closure)

/// Decodes an audio file incrementally into fixed-size 16 kHz mono Float32 blocks via a
/// persistent `AVAudioConverter`. The converter is created once and fed through its input
/// block so its resampling state stays continuous across pulls; only one native window plus
/// one output block live in memory at a time.
final class WindowedAudioReader: @unchecked Sendable {
    private let file: AVAudioFile
    private let converter: AVAudioConverter
    private let inputBuffer: AVAudioPCMBuffer
    private let outputBuffer: AVAudioPCMBuffer
    private let outputBlockFrames: AVAudioFrameCount
    private let nativeWindowFrames: AVAudioFrameCount

    private var fileExhausted = false   // no more native frames left to read from the file
    private var converterDrained = false // converter reported endOfStream and produced nothing more

    /// - Parameters:
    ///   - url: input file
    ///   - targetSampleRate: output sample rate (16 kHz for Sortformer)
    ///   - outputBlockSeconds: size of each pulled output block (default 1.0 s)
    init(url: URL, targetSampleRate: Double = 16000, outputBlockSeconds: Double = 1.0) throws {
        let file = try AVAudioFile(forReading: url)
        self.file = file
        let inputFormat = file.processingFormat

        guard let outputFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: targetSampleRate,
            channels: 1,
            interleaved: false
        ) else {
            throw AppError.audioConverterUnavailable("could not build 16 kHz mono Float32 output format")
        }

        guard let converter = AVAudioConverter(from: inputFormat, to: outputFormat) else {
            throw AppError.audioConverterUnavailable(
                "AVAudioConverter(from: \(inputFormat), to: \(outputFormat)) returned nil"
            )
        }
        self.converter = converter

        self.outputBlockFrames = AVAudioFrameCount(max(1.0, outputBlockSeconds * targetSampleRate))
        guard let outputBuffer = AVAudioPCMBuffer(pcmFormat: outputFormat, frameCapacity: outputBlockFrames) else {
            throw AppError.audioConverterUnavailable("could not allocate output PCM buffer")
        }
        self.outputBuffer = outputBuffer

        // Native window sized to roughly match each output block (in input sample-rate frames),
        // so the converter rarely starves mid-block. The converter pulls more via its input
        // block as needed; this is just a buffering granularity.
        let ratio = inputFormat.sampleRate / targetSampleRate
        self.nativeWindowFrames = AVAudioFrameCount(max(1.0, Double(outputBlockFrames) * ratio))
        guard let inputBuffer = AVAudioPCMBuffer(pcmFormat: inputFormat, frameCapacity: nativeWindowFrames) else {
            throw AppError.audioConverterUnavailable("could not allocate input PCM buffer")
        }
        self.inputBuffer = inputBuffer
    }

    /// Pull the next 16 kHz mono block. Returns nil at end-of-stream, and NEVER a non-nil
    /// empty block: a zero-produce-but-not-drained transient (e.g. the converter needs more
    /// input) is retried internally. Each retry's input block reads another native window,
    /// advancing the file, so the loop is bounded — it always reaches frames or EOF.
    func next() throws -> [Float]? {
        while true {
            if converterDrained { return nil }

            outputBuffer.frameLength = 0
            var conversionError: NSError?

            let status = converter.convert(to: outputBuffer, error: &conversionError) { [self] _, outStatus in
                // Input block: read the next native window from the file. Signals endOfStream once
                // the file is exhausted, so the converter flushes any buffered tail.
                if fileExhausted {
                    outStatus.pointee = .endOfStream
                    return nil
                }
                do {
                    inputBuffer.frameLength = 0
                    try file.read(into: inputBuffer, frameCount: nativeWindowFrames)
                } catch {
                    // A genuine mid-file decode error truncates the stream. Warn on stderr so a
                    // corrupt run is not mistaken for a clean one, then end the stream gracefully.
                    fputs("warning: audio read failed mid-file, ending stream early: \(error.localizedDescription)\n", stderr)
                    fileExhausted = true
                    outStatus.pointee = .endOfStream
                    return nil
                }
                if inputBuffer.frameLength == 0 {
                    fileExhausted = true
                    outStatus.pointee = .endOfStream
                    return nil
                }
                outStatus.pointee = .haveData
                return inputBuffer
            }

            if let conversionError {
                throw AppError.audioReadFailed(conversionError.localizedDescription)
            }

            if status == .endOfStream {
                converterDrained = true
            }

            let produced = Int(outputBuffer.frameLength)
            if produced > 0 {
                guard let channel = outputBuffer.floatChannelData?[0] else {
                    throw AppError.audioReadFailed("converter produced frames but floatChannelData was nil")
                }
                return Array(UnsafeBufferPointer(start: channel, count: produced))
            }

            // produced == 0: drained -> EOF; otherwise loop and pull more input (file advances).
            if converterDrained { return nil }
        }
    }
}

// MARK: - Metrics helpers

func currentResidentBytes() -> UInt64 {
    var info = mach_task_basic_info()
    var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size / MemoryLayout<natural_t>.size)
    let kr = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
        }
    }
    return kr == KERN_SUCCESS ? info.resident_size : 0
}

func formatGB(_ bytes: UInt64) -> String {
    String(format: "%.3f GB", Double(bytes) / 1_073_741_824.0)
}

func formatGB(_ bytes: Int) -> String {
    String(format: "%.3f GB", Double(bytes) / 1_073_741_824.0)
}

func median(_ values: [Double]) -> Double {
    guard !values.isEmpty else { return 0 }
    let sorted = values.sorted()
    let mid = sorted.count / 2
    if sorted.count % 2 == 0 {
        return (sorted[mid - 1] + sorted[mid]) / 2
    }
    return sorted[mid]
}

// MARK: - App

@main
enum App {
    static func main() async {
        // CRITICAL: cap MLX/Metal memory FIRST, before any MLXArray allocation. Uncapped MLX can
        // OOM-reboot the machine. 18 GiB hard cap (project policy). See note below on the symbol.
        capMLXMemory(bytes: 18 * 1024 * 1024 * 1024)

        do {
            let args = try CLI.parse()
            try await run(args)
        } catch {
            fputs("Error: \(error)\n", stderr)
            CLI.printUsage()
            exit(1)
        }
    }

    /// 18 GiB MLX/Metal hard cap.
    ///
    /// Symbol note: the installed mlx-swift (0.30.x) marks `GPU.set(memoryLimit:relaxed:)` as
    /// *deprecated* (renamed to the `Memory.memoryLimit` property, which has no `relaxed:`
    /// parameter — the limit is always strict / wait-on-malloc). We set the non-deprecated
    /// `Memory.memoryLimit` to keep the build warning-clean; this is the strict cap the project
    /// memory note's `relaxed: false` intended.
    static func capMLXMemory(bytes: Int) {
        MLX.Memory.memoryLimit = bytes
    }

    static func run(_ args: CLI) async throws {
        let inputURL = resolveURL(path: args.inputPath)
        guard FileManager.default.fileExists(atPath: inputURL.path) else {
            throw AppError.inputFileNotFound(inputURL.path)
        }

        try ensureMLXRuntimeReadyForShell()

        if args.selfCheck {
            try await runSelfCheck(args, inputURL: inputURL)
            return
        }

        // Audio duration (for RTF) from the source file, capped by --max-seconds if set.
        let probe = try AVAudioFile(forReading: inputURL)
        let fileDurationSec = Double(probe.length) / probe.processingFormat.sampleRate
        let audioDurationSec = args.maxSeconds.map { min($0, fileDurationSec) } ?? fileDurationSec

        print("Loading Sortformer model (\(args.repo))")
        let model = try await SortformerModel.fromPretrained(args.repo)

        print("Opening audio (\(inputURL.path)) — \(String(format: "%.1f", fileDurationSec))s @ \(Int(probe.processingFormat.sampleRate)) Hz, \(probe.processingFormat.channelCount) ch")
        let reader = try WindowedAudioReader(url: inputURL, targetSampleRate: 16000, outputBlockSeconds: 1.0)

        // Bound the producer by --max-seconds (count of 16 kHz mono samples already pulled).
        let sampleBudget: Int? = args.maxSeconds.map { Int($0 * 16000.0) }
        let pulledSamples = Counter()

        let audioSource: () async throws -> [Float]? = {
            if let budget = sampleBudget, pulledSamples.value >= budget {
                return nil
            }
            guard var block = try reader.next() else { return nil }
            if let budget = sampleBudget {
                let remaining = budget - pulledSamples.value
                if remaining <= 0 { return nil }
                if block.count > remaining {
                    block = Array(block.prefix(remaining))
                }
            }
            pulledSamples.add(block.count)
            return block
        }

        print("Running bounded streaming diarization (chunk=\(args.chunkDuration)s, threshold=\(args.threshold))")

        var allSegments: [DiarizationSegment] = []
        var perChunkLatency: [Double] = []
        var peakRSS: UInt64 = currentResidentBytes()

        let wallStart = Date()
        var lastTick = wallStart

        for try await out in model.generateStreamBounded(
            audioSource: audioSource,
            sampleRate: 16000,
            chunkDuration: args.chunkDuration,
            threshold: args.threshold,
            verbose: args.verbose
        ) {
            let now = Date()
            perChunkLatency.append(now.timeIntervalSince(lastTick))
            lastTick = now
            allSegments.append(contentsOf: out.segments)
            let rss = currentResidentBytes()
            if rss > peakRSS { peakRSS = rss }
        }

        let wallSeconds = Date().timeIntervalSince(wallStart)
        let mlxPeak = MLX.Memory.peakMemory

        // RTTM out (concatenate per-chunk RTTM lines; reuse DiarizationOutput.text formatting).
        if let rttmOut = args.rttmOut, !rttmOut.isEmpty {
            let rttm = DiarizationOutput(segments: allSegments).text
            let rttmURL = resolveURL(path: rttmOut)
            try (rttm + "\n").write(to: rttmURL, atomically: true, encoding: .utf8)
            print("Wrote RTTM (\(allSegments.count) segments) to \(rttmURL.path)")
        }

        printMetrics(
            wallSeconds: wallSeconds,
            audioDurationSec: audioDurationSec,
            perChunkLatency: perChunkLatency,
            segments: allSegments,
            peakRSS: peakRSS,
            mlxPeak: mlxPeak
        )
    }

    // MARK: - Self-consistency gate (precompute vs bounded)

    /// Frame grid resolution for the agreement matrix: hop * subsamplingFactor / sr = 160*8/16000.
    static let selfCheckFrameDuration = 0.08

    /// Loads a SHORT clip fully (16 kHz mono) and compares the precompute streaming path
    /// (`generateStream`, full-file mel + preEncode) against the memory-bounded path
    /// (`generateStreamBounded`, sliding window) on the SAME samples. Renders both segment lists
    /// to per-frame per-speaker-slot activity matrices and reports agreement. Speaker slots are
    /// directly comparable (the model assigns slots deterministically — no Hungarian matching).
    static func runSelfCheck(_ args: CLI, inputURL: URL) async throws {
        print("Loading Sortformer model (\(args.repo))")
        let model = try await SortformerModel.fromPretrained(args.repo)

        // Load the SHORT clip FULLY as one mono [Float] @ 16 kHz, then slice to --max-seconds.
        let (sr, audioArray) = try loadAudioArray(from: inputURL, sampleRate: 16000)
        var samples = audioArray.asArray(Float.self)
        if let maxSeconds = args.maxSeconds {
            let budget = Int(maxSeconds * Double(sr))
            if samples.count > budget { samples = Array(samples.prefix(budget)) }
        }
        let clipDurationSec = Double(samples.count) / Double(sr)
        print(String(format: "Self-check clip: %.2f s @ %d Hz (%d samples)", clipDurationSec, sr, samples.count))
        print(String(format: "Running BOTH paths with chunk-duration=%.1fs, threshold=%.2f", args.chunkDuration, args.threshold))

        // --- Precompute path: generateStream over the full MLXArray of the SAME samples. ---
        print("  [1/2] precompute path (generateStream)...")
        var precomputeSegments: [DiarizationSegment] = []
        let preStart = Date()
        for try await out in model.generateStream(
            audio: MLXArray(samples),
            sampleRate: 16000,
            chunkDuration: args.chunkDuration,
            threshold: args.threshold,
            verbose: args.verbose
        ) {
            precomputeSegments.append(contentsOf: out.segments)
        }
        let preWall = Date().timeIntervalSince(preStart)

        // --- Bounded path: generateStreamBounded over a closure yielding the SAME samples
        //     in ~1 s blocks, then nil. ---
        print("  [2/2] bounded path (generateStreamBounded)...")
        let blockSize = 16000 // ~1 s blocks
        let cursor = Counter()
        let sendableSamples = samples
        let audioSource: () async throws -> [Float]? = {
            let start = cursor.value
            if start >= sendableSamples.count { return nil }
            let end = min(start + blockSize, sendableSamples.count)
            cursor.add(end - start)
            return Array(sendableSamples[start..<end])
        }
        var boundedSegments: [DiarizationSegment] = []
        let bndStart = Date()
        for try await out in model.generateStreamBounded(
            audioSource: audioSource,
            sampleRate: 16000,
            chunkDuration: args.chunkDuration,
            threshold: args.threshold,
            verbose: args.verbose
        ) {
            boundedSegments.append(contentsOf: out.segments)
        }
        let bndWall = Date().timeIntervalSince(bndStart)

        // --- Render to per-frame per-slot activity matrices over [0, clipDuration]. ---
        let frameDur = selfCheckFrameDuration
        let nFrames = max(1, Int((clipDurationSec / frameDur).rounded(.up)))
        let nSlots = (precomputeSegments + boundedSegments).map { $0.speaker }.max().map { $0 + 1 } ?? 0

        let preMatrix = activityMatrix(precomputeSegments, nFrames: nFrames, nSlots: nSlots, frameDur: frameDur)
        let bndMatrix = activityMatrix(boundedSegments, nFrames: nFrames, nSlots: nSlots, frameDur: frameDur)

        // Per-slot frame agreement + per-slot speech time; overall frame agreement.
        var totalAgree = 0
        let totalCells = nFrames * max(nSlots, 1)
        print("")
        print("============= Self-Consistency (precompute vs bounded) =============")
        print(String(format: "Clip duration:         %.2f s  (%d frames @ %.2fs)", clipDurationSec, nFrames, frameDur))
        print("Speaker slots:         \(nSlots)")
        print(String(format: "Precompute wall:       %.2f s   segments: %d", preWall, precomputeSegments.count))
        print(String(format: "Bounded wall:          %.2f s   segments: %d", bndWall, boundedSegments.count))
        print("--------------------------------------------------------------------")
        print("Per-slot frame agreement %  |  speech-time precompute / bounded (s)")
        for slot in 0..<max(nSlots, 1) {
            var agree = 0
            var prePos = 0
            var bndPos = 0
            for f in 0..<nFrames {
                let p = nSlots > 0 ? preMatrix[slot * nFrames + f] : false
                let b = nSlots > 0 ? bndMatrix[slot * nFrames + f] : false
                if p == b { agree += 1 }
                if p { prePos += 1 }
                if b { bndPos += 1 }
            }
            totalAgree += agree
            let pct = nFrames > 0 ? 100.0 * Double(agree) / Double(nFrames) : 100.0
            let preSec = Double(prePos) * frameDur
            let bndSec = Double(bndPos) * frameDur
            print(String(format: "  slot %d:  %6.2f %%        |  %7.2f / %7.2f", slot, pct, preSec, bndSec))
        }
        let overallPct = totalCells > 0 ? 100.0 * Double(totalAgree) / Double(totalCells) : 100.0
        print("--------------------------------------------------------------------")
        print(String(format: "OVERALL frame agreement:  %.2f %%  (%d / %d cells)", overallPct, totalAgree, totalCells))
        let preTotalSec = Double(preMatrix.lazy.filter { $0 }.count) * frameDur
        let bndTotalSec = Double(bndMatrix.lazy.filter { $0 }.count) * frameDur
        print(String(format: "Total speech-time:        precompute %.2f s  /  bounded %.2f s", preTotalSec, bndTotalSec))
        print("====================================================================")

        // Verdict thresholds (per Task 7): >=~90% expected; <80% signals a frame-accounting bug.
        if overallPct < 80.0 {
            print("")
            print("FAIL: overall frame agreement \(String(format: "%.2f", overallPct))% < 80% — likely a frame-accounting/dtype bug.")
            print("      Do NOT commit. Inspect window planning (BoundedWindowPlanner), dtype, and MLX.eval discipline.")
            exit(2)
        } else if overallPct < 90.0 {
            print("")
            print("WARN: overall frame agreement \(String(format: "%.2f", overallPct))% in [80,90) — passable but lower than expected (~>=90%).")
        } else {
            print("")
            print("PASS: bounded ≈ precompute (overall frame agreement \(String(format: "%.2f", overallPct))% >= 90%).")
        }
    }

    /// Flatten segments into a row-major [slot * nFrames + frame] boolean activity grid over
    /// `[0, nFrames * frameDur)`. Each segment marks frames whose centre falls in [start, end).
    static func activityMatrix(
        _ segments: [DiarizationSegment],
        nFrames: Int,
        nSlots: Int,
        frameDur: Double
    ) -> [Bool] {
        guard nSlots > 0, nFrames > 0 else { return [] }
        var grid = [Bool](repeating: false, count: nSlots * nFrames)
        for seg in segments {
            let slot = seg.speaker
            guard slot >= 0, slot < nSlots else { continue }
            let f0 = max(0, Int((Double(seg.start) / frameDur).rounded(.down)))
            let f1 = min(nFrames, Int((Double(seg.end) / frameDur).rounded(.up)))
            if f1 <= f0 { continue }
            for f in f0..<f1 { grid[slot * nFrames + f] = true }
        }
        return grid
    }

    static func printMetrics(
        wallSeconds: Double,
        audioDurationSec: Double,
        perChunkLatency: [Double],
        segments: [DiarizationSegment],
        peakRSS: UInt64,
        mlxPeak: Int
    ) {
        let speakers = Set(segments.map { $0.speaker })
        let minLat = perChunkLatency.min() ?? 0
        let maxLat = perChunkLatency.max() ?? 0
        let medLat = median(perChunkLatency)
        let first = perChunkLatency.first ?? 0
        let last = perChunkLatency.last ?? 0
        let trend: String
        if first > 0 {
            trend = String(format: "%.2fx (last/first)", last / first)
        } else {
            trend = "n/a"
        }
        let rtf = audioDurationSec > 0 ? wallSeconds / audioDurationSec : 0
        let mlxPeakStr = mlxPeak > 0 ? formatGB(mlxPeak) : "n/a"

        print("")
        print("==================== Diarization Metrics ====================")
        print(String(format: "Audio duration:        %.2f s", audioDurationSec))
        print(String(format: "Total wall time:       %.2f s", wallSeconds))
        print(String(format: "RTF (wall/audio):      %.3f", rtf))
        print("Chunks processed:      \(perChunkLatency.count)")
        print(String(format: "Per-chunk latency:     min %.3fs / median %.3fs / max %.3fs", minLat, medLat, maxLat))
        print(String(format: "Latency trend:         first %.3fs -> last %.3fs  (%@)", first, last, trend))
        print("Total segments:        \(segments.count)")
        print("Distinct speakers:     \(speakers.count)  \(speakers.sorted())")
        print("Peak process RSS:      \(formatGB(peakRSS))")
        print("MLX peak memory:       \(mlxPeakStr)")
        print("=============================================================")
    }

    private static func resolveURL(path: String) -> URL {
        if path.hasPrefix("/") {
            return URL(fileURLWithPath: path)
        }
        return URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent(path)
    }

    // MARK: - MLX metallib discovery (mirrors mlx-audio-swift-lid)

    static func ensureMLXRuntimeReadyForShell(
        executableURL: URL? = CommandLine.arguments.first.map { URL(fileURLWithPath: $0) },
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws {
        let searchRoots = runtimeSearchRoots(executableURL: executableURL, environment: environment)
        let candidatePaths = metallibCandidates(searchRoots: searchRoots)

        if candidatePaths.contains(where: { FileManager.default.fileExists(atPath: $0.path) }) {
            return
        }

        let searchedPaths = candidatePaths.map(\.path).joined(separator: ", ")
        throw AppError.mlxRuntimeNotConfigured(
            """
            could not find MLX metal resources near the executable or on DYLD_FRAMEWORK_PATH. \
            Searched: \(searchedPaths). Run the tool from Xcode, or export DYLD_FRAMEWORK_PATH \
            to the SwiftPM build directory before invoking the CLI.
            """
        )
    }

    private static func runtimeSearchRoots(
        executableURL: URL?,
        environment: [String: String]
    ) -> [URL] {
        var roots: [URL] = []

        if let executableURL {
            roots.append(executableURL.deletingLastPathComponent())
        }

        if let frameworkPath = environment["DYLD_FRAMEWORK_PATH"] {
            for rawPath in frameworkPath.split(separator: ":") where !rawPath.isEmpty {
                roots.append(URL(fileURLWithPath: String(rawPath)))
            }
        }

        return roots
    }

    private static func metallibCandidates(searchRoots: [URL]) -> [URL] {
        let suffixes = [
            "default.metallib",
            "mlx.metallib",
            "Resources/default.metallib",
            "Resources/mlx.metallib",
            "mlx-swift_Cmlx.bundle/default.metallib",
            "mlx-swift_Cmlx.bundle/mlx.metallib",
            "mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib",
            "mlx-swift_Cmlx.bundle/Contents/Resources/mlx.metallib",
        ]

        return searchRoots.flatMap { root in
            suffixes.map { root.appendingPathComponent($0) }
        }
    }
}

/// Tiny mutable sample counter captured by reference into the pull closure.
/// `generateStreamBounded` runs the closure serially on a single detached task, so no locking
/// is required for correctness.
final class Counter {
    private var _value = 0
    var value: Int { _value }
    func add(_ n: Int) { _value += n }
}

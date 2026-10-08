import Foundation
@preconcurrency import MLX
import MLXAudioMusic

enum CLIError: Error, CustomStringConvertible {
    case missingValue(String)
    case unknownOption(String)
    case invalidValue(String, String)

    var description: String {
        switch self {
        case .missingValue(let key): "Missing value for \(key)"
        case .unknownOption(let key): "Unknown option \(key)"
        case .invalidValue(let key, let value): "Invalid value for \(key): \(value)"
        }
    }
}

struct CLI {
    var weights: URL?
    var variant = StableAudio3.Variant.smallMusic
    var prompt: String?
    var seconds = 30.0
    var seed: UInt64 = 0
    var steps = 8
    var output: URL?

    static func parse(_ arguments: [String]) throws -> CLI {
        var cli = CLI()
        var iterator = arguments.makeIterator()
        func value(_ key: String) throws -> String {
            guard let value = iterator.next() else { throw CLIError.missingValue(key) }
            return value
        }
        while let arg = iterator.next() {
            switch arg {
            case "--weights": cli.weights = URL(fileURLWithPath: try value(arg), isDirectory: true)
            case "--dit":
                let raw = try value(arg)
                guard let variant = StableAudio3.Variant(rawValue: raw) else { throw CLIError.invalidValue(arg, raw) }
                cli.variant = variant
            case "--prompt": cli.prompt = try value(arg)
            case "--seconds":
                let raw = try value(arg)
                guard let seconds = Double(raw), seconds > 0 else { throw CLIError.invalidValue(arg, raw) }
                cli.seconds = seconds
            case "--seed":
                let raw = try value(arg)
                guard let seed = UInt64(raw) else { throw CLIError.invalidValue(arg, raw) }
                cli.seed = seed
            case "--steps":
                let raw = try value(arg)
                guard let steps = Int(raw), steps > 0 else { throw CLIError.invalidValue(arg, raw) }
                cli.steps = steps
            case "--out", "-o": cli.output = URL(fileURLWithPath: try value(arg))
            case "--help", "-h":
                printUsage()
                exit(0)
            default: throw CLIError.unknownOption(arg)
            }
        }
        guard cli.weights != nil else { throw CLIError.missingValue("--weights") }
        guard cli.prompt != nil else { throw CLIError.missingValue("--prompt") }
        guard cli.output != nil else { throw CLIError.missingValue("--out") }
        return cli
    }

    static func printUsage() {
        print(
            """
            Usage:
              mlx-audio-swift-music --weights <dir> --prompt <text> --out <file.wav> [options]

            Generates music with Stable Audio 3 from the MLX weights in
            stabilityai/stable-audio-3-optimized (the MLX/ folder).

            Options:
              --weights <dir>     Directory with t5gemma_f16.npz, dit_<variant>_f16.npz and the decoder
              --dit <variant>     sm-music (default) or medium
              --prompt <text>     Text prompt
              --seconds <s>       Length in seconds. Default: 30
              --seed <n>          Random seed. Default: 0
              --steps <n>         Sampler steps. Default: 8
              --out, -o <path>    Output WAV, 16-bit stereo 44.1 kHz
            """
        )
    }
}

/// 16-bit PCM WAV: clipped to [-1, 1] and truncated, like the reference writer.
func writeWAV(_ audio: MLXArray, sampleRate: Int, to url: URL) throws {
    let channels = audio.dim(0)
    let interleaved = clip(audio, min: -1, max: 1).transposed(1, 0).flattened() * 32767
    let samples = interleaved.asType(.int16).asArray(Int16.self)
    var data = Data()
    func append<T: FixedWidthInteger>(_ value: T) {
        withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) }
    }
    let bytes = samples.count * 2
    data.append(contentsOf: "RIFF".utf8); append(UInt32(36 + bytes))
    data.append(contentsOf: "WAVEfmt ".utf8); append(UInt32(16)); append(UInt16(1))
    append(UInt16(channels)); append(UInt32(sampleRate))
    append(UInt32(sampleRate * channels * 2)); append(UInt16(channels * 2)); append(UInt16(16))
    data.append(contentsOf: "data".utf8); append(UInt32(bytes))
    samples.withUnsafeBufferPointer { data.append(UnsafeBufferPointer(start: $0.baseAddress, count: $0.count)) }
    try data.write(to: url)
}

@main
enum App {
    static func main() {
        do {
            let cli = try CLI.parse(Array(CommandLine.arguments.dropFirst()))
            let model = StableAudio3(variant: cli.variant, weightsDirectory: cli.weights!)
            let started = Date()
            let audio = try model.generate(
                .init(prompt: cli.prompt!, seconds: cli.seconds, seed: cli.seed, steps: cli.steps)
            ) { stage in
                switch stage {
                case .encodingText: print("text encoder")
                case .sampling(let step, let total): print("sample \(step)/\(total)")
                case .decoding: print("decoder")
                }
            }
            try writeWAV(audio, sampleRate: StableAudio3.sampleRate, to: cli.output!)
            let wall = Date().timeIntervalSince(started)
            let peak = Double(Memory.peakMemory) / 1_073_741_824
            print(String(format: "done %.2fs wall  %.1fs audio  peak %.2f GB  seed %llu",
                         wall, Double(audio.dim(1)) / Double(StableAudio3.sampleRate), peak, cli.seed))
        } catch {
            FileHandle.standardError.write(Data("error: \(error)\n".utf8))
            exit(1)
        }
    }
}

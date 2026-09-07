//
//  main.swift
//
//  Runs the MLX arm of the VibeVoice ASR benchmark and writes an omni-bench
//  Run Artifact. The Python CLI's `run` command is not usable here because the
//  adapter is Swift, so this drives `AudioTranscriptionProducer` directly --
//  the same producer, the same measurement, the same artifact bytes.
//
//  Identity is supplied as a file rather than assembled here on purpose: the
//  Registry digests it carries are resolved by
//  `tools/omni_bench_host_vibevoice/emit_swift_identity.py` through
//  omni-bench's own resolver, so they cannot drift out of sync with the
//  Registry the way hand-copied constants would.
//
//  Scoring stays in Python; this only produces Evidence.
//

import Foundation
import MLXAudioSTT
import OmniBench
import OmniBenchVibeVoiceHost

struct Arguments {
    var manifest: String?
    var identity: String?
    var modelPath: String?
    var out: String?
    var chunkDuration = 2.0
    var textAudioDelay = 0.5
    var maxNewTokens = 256
    var temperature: Float = 0.0
}

func parseArguments() -> Arguments {
    var args = Arguments()
    var iterator = CommandLine.arguments.dropFirst().makeIterator()
    while let flag = iterator.next() {
        switch flag {
        case "--manifest": args.manifest = iterator.next()
        case "--identity": args.identity = iterator.next()
        case "--model": args.modelPath = iterator.next()
        case "--out": args.out = iterator.next()
        case "--chunk-duration": args.chunkDuration = Double(iterator.next() ?? "") ?? 2.0
        case "--text-audio-delay": args.textAudioDelay = Double(iterator.next() ?? "") ?? 0.5
        case "--max-new-tokens": args.maxNewTokens = Int(iterator.next() ?? "") ?? 256
        case "--temperature": args.temperature = Float(iterator.next() ?? "") ?? 0.0
        case "--help", "-h":
            print("""
            vibevoice-omni-bench-run \\
              --manifest <prepared manifest.json> \\
              --identity <identity.json from emit_swift_identity.py> \\
              --model <VibeVoice checkpoint directory> \\
              --out <artifact.jsonl> \\
              [--chunk-duration 2.0] [--text-audio-delay 0.5] \\
              [--max-new-tokens 256] [--temperature 0.0]
            """)
            exit(0)
        default:
            FileHandle.standardError.write(Data("unknown flag \(flag)\n".utf8))
            exit(2)
        }
    }
    return args
}

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data("error: \(message)\n".utf8))
    exit(1)
}

let args = parseArguments()
guard let manifestPath = args.manifest, let identityPath = args.identity,
      let modelPath = args.modelPath, let outPath = args.out
else {
    fail("--manifest, --identity, --model and --out are all required (see --help)")
}

guard let identityData = try? Data(contentsOf: URL(fileURLWithPath: identityPath)),
      let identity = try? JSONSerialization.jsonObject(with: identityData) as? [String: Any]
else {
    fail("could not read identity JSON at \(identityPath)")
}

// `construct` is the Task's own construct block; the producer echoes it into
// the artifact and the adapter can read `language` off it.
let construct = (identity["construct"] as? [String: Any]) ?? [:]

let run: PreparedAudioTranscriptionRun
do {
    run = try PreparedAudioTranscriptionRun(
        manifestURL: URL(fileURLWithPath: manifestPath),
        identity: identity,
        construct: construct)
} catch {
    fail("prepared run rejected: \(error)")
}

print("loading \(modelPath) ...")
let model: VibeVoiceASRStreamingModel
do {
    model = try await VibeVoiceASRStreamingModel.fromModelDirectory(
        URL(fileURLWithPath: modelPath))
} catch {
    fail("could not load model: \(error)")
}

var parameters = VibeVoiceStreamingParameters.default
parameters.chunkDuration = args.chunkDuration
parameters.textAudioDelay = args.textAudioDelay
parameters.maxNewTokensPerChunk = args.maxNewTokens
// Greedy on both arms: a sampled decode would make the parity delta noise.
parameters.temperature = args.temperature

let host = VibeVoiceOmniBenchHost(model: model, parameters: parameters)
do {
    try host.warmup()
} catch {
    fail("warmup failed: \(error)")
}

print("running \(run.samples.count) samples ...")
let producer = AudioTranscriptionProducer(run: run)
do {
    let result = try producer.run(adapter: host, outputURL: URL(fileURLWithPath: outPath))
    print("artifact    : \(result.artifactURL.path)")
    print("identity_key: \(result.identityKey)")
    print("sha256      : \(result.artifactSha256)")
} catch {
    fail("producer failed: \(error)")
}

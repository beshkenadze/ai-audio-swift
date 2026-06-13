#if canImport(CoreML)
import CoreML
import Foundation
import HuggingFace
import MLX

// Nemotron streaming session whose FastConformer ENCODER runs on the ANE (CoreML),
// while the prompt MLP + RNN-T decode stay in MLX. The encoder is the bulk of the
// compute, so this offloads it to the ANE (÷~5.8 power) and frees the GPU for the
// accurate Voxtral lane — the ANE∥GPU half of the two-tier.
//
// The encoder is fed the validated uniform-F recipe (each window = [preFrames prev-mel
// ++ newFrames new-mel], stride newFrames, zero-padded at start and final tail). This
// mirrors the offline `cacheAwareStreamEncodeCoreML` loop, driven incrementally as
// audio arrives. Decode reuses `streamRNNTDecode` — token-identical to the MLX path.

public final class NemotronCoreMLStreamSession: @unchecked Sendable {
    private let model: NemotronASRModel
    private let encoder: NemotronCoreMLStreamingEncoder
    private let rnntState: NemotronASRStreamRNNTState
    private let frameSeconds: Double
    private let pre: Int
    private let new: Int
    private let sf: Int
    private let featIn: Int

    private var audio: [Float] = []
    private var processed = 0          // mel frames already consumed
    private var emittedText = ""

    public var text: String { emittedText }

    /// HF repo with the prebuilt cache-aware streaming CoreML/ANE encoder.
    public static let defaultRepo = "beshkenadze/nemotron-3.5-asr-streaming-0.6b-coreml-ane-stream"

    public static func create(
        model: NemotronASRModel,
        repo: String = defaultRepo,
        preFrames: Int = 9,
        newFrames: Int = 112
    ) async throws -> NemotronCoreMLStreamSession {
        let url = try await downloadPackage(repo: repo)
        let enc = try NemotronCoreMLStreamingEncoder(
            modelURL: url,
            featIn: model.encoderConfig.featIn,
            dModel: model.encoderConfig.dModel,
            subsamplingFactor: model.encoderConfig.subsamplingFactor,
            preFrames: preFrames,
            newFrames: newFrames,
            layers: model.encoderConfig.nLayers,
            attnCache: model.defaultAttContextSize.first ?? 70,
            convCache: model.encoderConfig.convKernelSize - 1)
        return NemotronCoreMLStreamSession(model: model, encoder: enc)
    }

    private init(model: NemotronASRModel, encoder: NemotronCoreMLStreamingEncoder) {
        self.model = model
        self.encoder = encoder
        self.rnntState = NemotronASRStreamRNNTState(blankToken: model.blankTokenID)
        self.frameSeconds = Double(model.encoderConfig.subsamplingFactor * model.preprocessConfig.hopLength)
            / Double(model.preprocessConfig.sampleRate)
        self.pre = encoder.preFrames
        self.new = encoder.newFrames
        self.sf = model.encoderConfig.subsamplingFactor
        self.featIn = encoder.featIn
        encoder.reset()
    }

    @discardableResult
    public func step(_ samples: [Float]) -> String {
        audio.append(contentsOf: samples)
        drain(final: false)
        return emittedText
    }

    @discardableResult
    public func finish() -> String {
        drain(final: true)
        return emittedText
    }

    private func drain(final: Bool) {
        guard !audio.isEmpty else { return }
        var mel = NemotronASRAudio.logMelSpectrogram(MLXArray(audio), config: model.preprocessConfig)
        if mel.ndim == 2 { mel = mel.expandedDimensions(axis: 0) }
        mel = mel.asType(.float32)
        let total = mel.shape[1]

        while processed < total {
            let realNew = min(new, total - processed)
            if realNew < new && !final { break }   // wait for a full chunk unless flushing
            let p = processed
            // window = [pre prev-mel ++ new new-mel], zeros at first prepend and last tail.
            let avail = min(pre, p)
            var parts: [MLXArray] = []
            if pre - avail > 0 { parts.append(MLXArray.zeros([1, pre - avail, featIn], dtype: .float32)) }
            if avail > 0 { parts.append(mel[0..., (p - avail)..<p, 0...]) }
            parts.append(mel[0..., p..<(p + realNew), 0...])
            if new - realNew > 0 { parts.append(MLXArray.zeros([1, new - realNew, featIn], dtype: .float32)) }
            let window = MLX.concatenated(parts, axis: 1)

            guard let encoded = try? encoder.step(window) else { break }
            let isFinal = final && (p + realNew >= total)
            let keep = isFinal ? encoded.shape[2] : min(encoded.shape[2], max(1, (realNew + sf - 1) / sf))
            let h = encoded[0..., 0..., 0..<keep].transposed(0, 2, 1).asType(model.computeDType)
            model.streamRNNTDecode(model.applyPrompt(h, language: nil), state: rnntState, frameSeconds: frameSeconds)
            processed += realNew
        }
        emittedText = NemoAlignment.sentencesToResult(NemoAlignment.tokensToSentences(rnntState.results)).text
    }

    // Minimal HF snapshot download of the encoder package (mirrors ParakeetModel's).
    private static func downloadPackage(repo: String, cache: HubCache = .default) async throws -> URL {
        guard let repoID = Repo.ID(rawValue: repo) else {
            throw NSError(domain: "NemotronCoreML", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "Invalid ANE encoder repo: \(repo)"])
        }
        let hfToken = ProcessInfo.processInfo.environment["HF_TOKEN"]
        let client = (hfToken?.isEmpty == false)
            ? HubClient(host: HubClient.defaultHost, bearerToken: hfToken!, cache: cache)
            : HubClient(cache: cache)
        let dir = (client.cache ?? cache).cacheDirectory
            .appendingPathComponent("mlx-audio")
            .appendingPathComponent(repo.replacingOccurrences(of: "/", with: "_"))
        func findPkg() -> URL? {
            (try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil))?
                .first { ["mlpackage", "mlmodelc"].contains($0.pathExtension) }
        }
        if let p = findPkg() { return p }
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        _ = try await client.downloadSnapshot(
            of: repoID, kind: .model, to: dir, revision: "main",
            matching: ["*.json", "*.mlmodel", "*.bin", "*.weights", "*.mil", "*.espresso.*"],
            progressHandler: { _ in })
        guard let p = findPkg() else {
            throw NSError(domain: "NemotronCoreML", code: 3,
                          userInfo: [NSLocalizedDescriptionKey: "No .mlpackage/.mlmodelc in \(repo)"])
        }
        return p
    }
}
#endif

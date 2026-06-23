import Foundation
import MLX

/// Public façade for the Nemotron + Voxtral two-tier streaming pipeline.
///
/// The per-model `fromPretrained` loaders for Nemotron and Voxtral are
/// module-internal (unlike the other ASR models), so external consumers can't
/// assemble a `TwoTierSession` themselves. This loads both models once and vends
/// a fresh `TwoTierSession` per utterance (a session accumulates text, so each
/// new dictation needs a clean one).
public final class TwoTierEngine {
    public static let defaultNemotronRepo = "mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit"
    public static let defaultVoxtralRepo = "mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit"

    private let nemotron: NemotronASRModel
    private let voxtral: VoxtralRealtimeModel

    private init(nemotron: NemotronASRModel, voxtral: VoxtralRealtimeModel) {
        self.nemotron = nemotron
        self.voxtral = voxtral
    }

    /// Download (first run) and load both models. Caps Metal memory so an
    /// unbounded MLX run can't OOM-reboot the machine.
    public static func load(
        nemotronRepo: String = defaultNemotronRepo,
        voxtralRepo: String = defaultVoxtralRepo,
        memoryLimitBytes: Int = 18 * 1024 * 1024 * 1024
    ) async throws -> TwoTierEngine {
        GPU.set(memoryLimit: memoryLimitBytes, relaxed: false)
        let nemotron = try await NemotronASRModel.fromPretrained(nemotronRepo)
        let voxtral = try await VoxtralRealtimeModel.fromPretrained(voxtralRepo)
        return TwoTierEngine(nemotron: nemotron, voxtral: voxtral)
    }

    /// A fresh session for one utterance. `confirmed` = Voxtral finals,
    /// `partial` = Nemotron tail beyond Voxtral's coverage.
    public func makeSession(
        language: String? = nil,
        fastChunkMs: Int = 80,
        voxtralDelayMs: Int = 960
    ) -> TwoTierSession {
        TwoTierSession(
            nemotron: nemotron, voxtral: voxtral,
            language: language, fastChunkMs: fastChunkMs, voxtralDelayMs: voxtralDelayMs
        )
    }
}

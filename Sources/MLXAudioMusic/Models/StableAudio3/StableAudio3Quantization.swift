import Foundation
import MLX
import MLXNN

/// The archive describes packed storage; activations keep the stage's compute dtype.
struct StableAudio3Quantization: Decodable {
    let formatVersion: Int
    let bits: Int
    let groupSize: Int
    let mode: String

    enum CodingKeys: String, CodingKey {
        case formatVersion = "format_version", bits, groupSize = "group_size", mode
    }

    static func prepare(_ model: Module, archive: NPZArchive, weights: [String: MLXArray]) throws {
        try prepare(model, archive: archive, keys: Set(weights.keys))
    }

    /// `keys` are the module paths the weights will be loaded under.
    static func prepare(_ model: Module, archive: NPZArchive, keys: Set<String>) throws {
        guard archive.contains("QUANTIZATION") else {
            guard !keys.contains(where: { $0.hasSuffix(".scales") }) else {
                throw NPZArchiveError.missingEntry("QUANTIZATION")
            }
            return
        }
        let config = try JSONDecoder().decode(Self.self, from: archive.bytes("QUANTIZATION"))
        guard config.formatVersion == 1, [4, 8].contains(config.bits),
              [32, 64, 128].contains(config.groupSize), config.mode == "affine" else {
            throw NPZArchiveError.malformedHeader("QUANTIZATION")
        }
        quantize(model: model, groupSize: config.groupSize, bits: config.bits,
                 filter: { path, _ in keys.contains("\(path).scales") })
    }

    static func cast(_ weights: [String: MLXArray], to dtype: DType) -> [String: MLXArray] {
        weights.mapValues { $0.dtype == .uint32 ? $0 : $0.asType(dtype) }
    }
}

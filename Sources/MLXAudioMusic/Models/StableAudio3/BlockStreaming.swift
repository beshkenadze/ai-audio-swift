import Foundation
import MLX
import MLXNN

/// A stack of identically shaped blocks in an `.npz`, grouped by block.
///
/// Streaming keeps one module for the whole stack and reads each block's weights from
/// the archive as it is about to run. The arrays it replaces are released
/// once nothing computes with them, so a block or two is resident instead of all.
struct NPZBlockEntries {
    let archive: NPZArchive
    private let reader: NPZEntryReader
    /// Per block: module key relative to the block, and its entry.
    private let blocks: [[(key: String, entry: NPZArchive.Entry)]]

    /// `moduleKey` maps an archive name to its module path, or nil for names to skip.
    init(archive: NPZArchive, prefix: String, count: Int, moduleKey: (String) -> String? = { $0 }) throws {
        var blocks = Array(repeating: [(key: String, entry: NPZArchive.Entry)](), count: count)
        for entry in try archive.entries(where: { $0 != "QUANTIZATION" }) {
            guard let key = moduleKey(entry.name), key.hasPrefix(prefix) else { continue }
            let path = key.dropFirst(prefix.count)
            guard let dot = path.firstIndex(of: "."), let index = Int(path[..<dot]), blocks.indices.contains(index)
            else { throw NPZArchiveError.malformedHeader(entry.name) }
            blocks[index].append((String(path[path.index(after: dot)...]), entry))
        }
        self.archive = archive
        self.blocks = blocks
        reader = NPZEntryReader(archive)
    }

    var keys: Set<String> { Set(blocks.first?.map(\.key) ?? []) }

    /// Every block must fill `module` exactly, so swapping one in later cannot fail.
    func validate(against module: Module) throws {
        let expected = Dictionary(uniqueKeysWithValues: module.parameters().flattened().map { ($0.0, $0.1.shape) })
        for (index, block) in blocks.enumerated() {
            let shapes = Dictionary(uniqueKeysWithValues: block.map { ($0.key, $0.entry.shape) })
            guard shapes == expected else {
                throw NPZArchiveError.missingEntry("block \(index) of \(archive.url.lastPathComponent)")
            }
        }
    }

    func parameters(_ index: Int, dtype: DType) -> ModuleParameters {
        let arrays = Dictionary(uniqueKeysWithValues: blocks[index].map { ($0.key, reader.array($0.entry)) })
        return ModuleParameters.unflattened(StableAudio3Quantization.cast(arrays, to: dtype))
    }
}

/// The DiT's transformer blocks, streamed. With the next block copied in behind the
/// one running, two of Medium's 24 blocks are resident instead of 2.9 GB of weights.
final class StableAudio3BlockStream {
    let block: StableAudio3DiTBlock
    private let entries: NPZBlockEntries

    init(archive: NPZArchive, config: StableAudio3DiTConfig) throws {
        entries = try NPZBlockEntries(archive: archive, prefix: "transformer.layers.", count: config.depth,
                                      moduleKey: StableAudio3DiT.moduleKey)
        block = StableAudio3DiTBlock(config)
        try StableAudio3Quantization.prepare(block, archive: archive, keys: entries.keys)
        try entries.validate(against: block)
    }

    /// Puts block `index` into ``block``.
    func load(_ index: Int) {
        block.update(parameters: entries.parameters(index, dtype: .float16))
    }
}

/// SAME-L's twelve blocks, streamed: one 140 MB block resident instead of 1.7 GB.
final class SAMELBlockStream {
    private let block = SAMELBlock(dimensions: SAMELDecoder.dimensions, sineGate: false)
    private let entries: NPZBlockEntries
    private let sineGate: (Int) -> Bool

    init(archive: NPZArchive, sineGate: @escaping (Int) -> Bool = SAMELDecoder.sineGate) throws {
        self.sineGate = sineGate
        entries = try NPZBlockEntries(archive: archive, prefix: "blocks.", count: SAMELDecoder.depth)
        try StableAudio3Quantization.prepare(block, archive: archive, keys: entries.keys)
        try entries.validate(against: block)
    }

    /// The one block module, holding block `index`.
    func load(_ index: Int) -> SAMELBlock {
        block.update(parameters: entries.parameters(index, dtype: .float32))
        block.ff.sineGate = sineGate(index)
        return block
    }
}

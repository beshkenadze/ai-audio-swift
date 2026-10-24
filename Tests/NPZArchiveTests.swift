import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXAudioMusic

struct NPZArchiveTests {
    private let values: [Float] = [1, -2, 3.5, 4, 0.25, -6]
    private var floats: Data { values.withUnsafeBufferPointer { Data(buffer: $0) } }

    @Test func readsStoredArrays() throws {
        let url = try NPZFixture.write(NPZFixture.zip([("w.npy", NPZFixture.npy(descr: "<f4", shape: [2, 3], body: floats))]))
        defer { try? FileManager.default.removeItem(at: url) }
        let archive = try NPZArchive(url: url)
        #expect(archive.names == ["w"])
        let w = try archive.array("w")
        #expect(w.shape == [2, 3])
        #expect(w.asArray(Float.self) == values)
    }

    @Test func entriesComeInFileOrder() throws {
        let url = try NPZFixture.write(NPZFixture.zip([
            ("b.npy", NPZFixture.npy(descr: "<f4", shape: [6], body: floats)),
            ("a.npy", NPZFixture.npy(descr: "<f4", shape: [6], body: floats)),
        ]))
        defer { try? FileManager.default.removeItem(at: url) }
        let archive = try NPZArchive(url: url)
        #expect(try archive.entries().map(\.name) == ["b", "a"])
        #expect(archive.names == ["a", "b"])
    }

    @Test func readsPackedUInt32WithoutLosingBits() throws {
        let values: [UInt32] = [0, 0x8000_0001, 0xFFFF_FFFF, 0x1234_ABCD]
        let body = values.withUnsafeBufferPointer { Data(buffer: $0) }
        let url = try NPZFixture.write(NPZFixture.zip([("w.npy", NPZFixture.npy(descr: "<u4", shape: [1, 4], body: body))]))
        defer { try? FileManager.default.removeItem(at: url) }
        let weights = try NPZArchive(url: url).arrays()
        let cast = StableAudio3Quantization.cast(weights, to: .float16)
        #expect(cast["w"]?.dtype == .uint32)
        #expect(cast["w"]?.asArray(UInt32.self) == values)
    }

    @Test func loadsAnAffineProjectionFromTheArchive() throws {
        let dense = MLXArray((0..<128).map { Float($0 % 19) / 20 }, [2, 64])
        let (packed, scales, biases) = quantized(dense, groupSize: 64, bits: 8)
        let packedData = packed.asArray(UInt32.self).withUnsafeBufferPointer { Data(buffer: $0) }
        let scaleData = scales.asArray(Float.self).withUnsafeBufferPointer { Data(buffer: $0) }
        let biasData = biases!.asArray(Float.self).withUnsafeBufferPointer { Data(buffer: $0) }
        let metadata = Data(#"{"format_version":1,"bits":8,"group_size":64,"mode":"affine"}"#.utf8)
        let url = try NPZFixture.write(NPZFixture.zip([
            ("projection.weight.npy", NPZFixture.npy(descr: "<u4", shape: packed.shape, body: packedData)),
            ("projection.scales.npy", NPZFixture.npy(descr: "<f4", shape: scales.shape, body: scaleData)),
            ("projection.biases.npy", NPZFixture.npy(descr: "<f4", shape: biases!.shape, body: biasData)),
            ("QUANTIZATION.npy", NPZFixture.npy(descr: "|u1", shape: [metadata.count], body: metadata)),
        ]))
        defer { try? FileManager.default.removeItem(at: url) }
        let archive = try NPZArchive(url: url)
        let weights = try archive.arrays { $0 != "QUANTIZATION" }
        let model = QuantizationTestProjection()
        try StableAudio3Quantization.prepare(model, archive: archive, weights: weights)
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        #expect(model.projection is QuantizedLinear)
        let input = MLXArray.ones([1, 64])
        let expected = matmul(input, dequantized(packed, scales: scales, biases: biases,
                                               groupSize: 64, bits: 8).T)
        let error = abs(model.projection(input) - expected).max().item(Float.self)
        #expect(error < 1e-4)
    }

    @Test func rejectsPackedWeightsWithoutTheirMetadata() throws {
        let url = try NPZFixture.write(NPZFixture.zip([("w.npy", NPZFixture.npy(descr: "<f4", shape: [2, 3], body: floats))]))
        defer { try? FileManager.default.removeItem(at: url) }
        let archive = try NPZArchive(url: url)
        #expect(throws: NPZArchiveError.self) {
            try StableAudio3Quantization.prepare(QuantizationTestProjection(), archive: archive,
                                                 weights: ["projection.scales": MLXArray.ones([2, 1])])
        }
    }

    @Test func rejectsAShapeLargerThanThePayload() throws {
        let url = try NPZFixture.write(NPZFixture.zip([("w.npy", NPZFixture.npy(descr: "<f4", shape: [3, 3], body: floats))]))
        defer { try? FileManager.default.removeItem(at: url) }
        #expect(throws: NPZArchiveError.self) { try NPZArchive(url: url).array("w") }
    }

    @Test func rejectsADimensionMLXCannotIndex() throws {
        // Valid for NumPy: an empty array whose second dimension exceeds Int32.
        let url = try NPZFixture.write(NPZFixture.zip([("w.npy", NPZFixture.npy(descr: "<f4", shape: [0, 2_147_483_648], body: Data()))]))
        defer { try? FileManager.default.removeItem(at: url) }
        #expect(throws: NPZArchiveError.self) { try NPZArchive(url: url).array("w") }
    }

    @Test func rejectsATruncatedArchive() throws {
        let archive = NPZFixture.zip([("w.npy", NPZFixture.npy(descr: "<f4", shape: [2, 3], body: floats))])
        // Keep the end record but cut the data it points into.
        let tail = archive.suffix(22)
        let url = try NPZFixture.write(archive.prefix(40) + tail)
        defer { try? FileManager.default.removeItem(at: url) }
        #expect(throws: NPZArchiveError.self) { try NPZArchive(url: url) }
    }

    @Test func rejectsADirectoryBeyondTheFile() throws {
        var archive = NPZFixture.zip([("w.npy", NPZFixture.npy(descr: "<f4", shape: [2, 3], body: floats))])
        let offsetField = archive.count - 6
        archive.replaceSubrange(offsetField..<(offsetField + 4), with: Data([0xFF, 0xFF, 0xFF, 0x7F]))
        let url = try NPZFixture.write(archive)
        defer { try? FileManager.default.removeItem(at: url) }
        #expect(throws: NPZArchiveError.self) { try NPZArchive(url: url) }
    }
}

private final class QuantizationTestProjection: Module {
    @ModuleInfo var projection: Linear

    override init() {
        _projection.wrappedValue = Linear(64, 2, bias: false)
        super.init()
    }
}

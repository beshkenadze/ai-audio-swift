import Foundation
import MLX
import Testing

@testable import MLXAudioMusic

struct NPZArchiveTests {
    /// A `.npy` payload: version 1.0 header, then raw little-endian elements.
    private func npy(descr: String, shape: [Int], body: Data) -> Data {
        let dims = shape.map(String.init).joined(separator: ", ") + (shape.count == 1 ? "," : "")
        var header = "{'descr': '\(descr)', 'fortran_order': False, 'shape': (\(dims)), }"
        while (10 + header.utf8.count + 1) % 64 != 0 { header += " " }
        header += "\n"
        var data = Data([0x93]) + Data("NUMPY".utf8) + Data([1, 0])
        data += withUnsafeBytes(of: UInt16(header.utf8.count).littleEndian) { Data($0) }
        return data + Data(header.utf8) + body
    }

    /// An uncompressed zip, as `np.savez` writes one.
    private func zip(_ entries: [(String, Data)]) -> Data {
        func le<T: FixedWidthInteger>(_ value: T) -> Data { withUnsafeBytes(of: value.littleEndian) { Data($0) } }
        var archive = Data()
        var directory = Data()
        for (name, payload) in entries {
            let offset = UInt32(archive.count)
            let fields = le(UInt16(20)) + le(UInt16(0)) + le(UInt16(0)) + le(UInt32(0)) + le(UInt32(0))
                + le(UInt32(payload.count)) + le(UInt32(payload.count))
                + le(UInt16(name.utf8.count)) + le(UInt16(0))
            archive += le(UInt32(0x0403_4B50)) + fields + Data(name.utf8) + payload
            directory += le(UInt32(0x0201_4B50)) + le(UInt16(20)) + fields
                + le(UInt16(0)) + le(UInt16(0)) + le(UInt16(0)) + le(UInt32(0)) + le(offset) + Data(name.utf8)
        }
        let directoryOffset = UInt32(archive.count)
        archive += directory
        archive += le(UInt32(0x0605_4B50)) + le(UInt16(0)) + le(UInt16(0))
            + le(UInt16(entries.count)) + le(UInt16(entries.count))
            + le(UInt32(directory.count)) + le(directoryOffset) + le(UInt16(0))
        return archive
    }

    private func write(_ data: Data) throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("\(UUID().uuidString).npz")
        try data.write(to: url)
        return url
    }

    private let values: [Float] = [1, -2, 3.5, 4, 0.25, -6]
    private var floats: Data { values.withUnsafeBufferPointer { Data(buffer: $0) } }

    @Test func readsStoredArrays() throws {
        let url = try write(zip([("w.npy", npy(descr: "<f4", shape: [2, 3], body: floats))]))
        defer { try? FileManager.default.removeItem(at: url) }
        let archive = try NPZArchive(url: url)
        #expect(archive.names == ["w"])
        let w = try archive.array("w")
        #expect(w.shape == [2, 3])
        #expect(w.asArray(Float.self) == values)
    }

    @Test func rejectsAShapeLargerThanThePayload() throws {
        let url = try write(zip([("w.npy", npy(descr: "<f4", shape: [3, 3], body: floats))]))
        defer { try? FileManager.default.removeItem(at: url) }
        #expect(throws: NPZArchiveError.self) { try NPZArchive(url: url).array("w") }
    }

    @Test func rejectsADimensionMLXCannotIndex() throws {
        // Valid for NumPy: an empty array whose second dimension exceeds Int32.
        let url = try write(zip([("w.npy", npy(descr: "<f4", shape: [0, 2_147_483_648], body: Data()))]))
        defer { try? FileManager.default.removeItem(at: url) }
        #expect(throws: NPZArchiveError.self) { try NPZArchive(url: url).array("w") }
    }

    @Test func rejectsATruncatedArchive() throws {
        let archive = zip([("w.npy", npy(descr: "<f4", shape: [2, 3], body: floats))])
        // Keep the end record but cut the data it points into.
        let tail = archive.suffix(22)
        let url = try write(archive.prefix(40) + tail)
        defer { try? FileManager.default.removeItem(at: url) }
        #expect(throws: NPZArchiveError.self) { try NPZArchive(url: url) }
    }

    @Test func rejectsADirectoryBeyondTheFile() throws {
        var archive = zip([("w.npy", npy(descr: "<f4", shape: [2, 3], body: floats))])
        let offsetField = archive.count - 6
        archive.replaceSubrange(offsetField..<(offsetField + 4), with: Data([0xFF, 0xFF, 0xFF, 0x7F]))
        let url = try write(archive)
        defer { try? FileManager.default.removeItem(at: url) }
        #expect(throws: NPZArchiveError.self) { try NPZArchive(url: url) }
    }
}

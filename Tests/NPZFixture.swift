import Foundation
import MLX
import MLXNN

/// `.npz` archives written the way `np.savez` writes them, for tests.
enum NPZFixture {
    /// A `.npy` payload: version 1.0 header, then raw little-endian elements.
    static func npy(descr: String, shape: [Int], body: Data) -> Data {
        let dims = shape.map(String.init).joined(separator: ", ") + (shape.count == 1 ? "," : "")
        var header = "{'descr': '\(descr)', 'fortran_order': False, 'shape': (\(dims)), }"
        while (10 + header.utf8.count + 1) % 64 != 0 { header += " " }
        header += "\n"
        var data = Data([0x93]) + Data("NUMPY".utf8) + Data([1, 0])
        data += withUnsafeBytes(of: UInt16(header.utf8.count).littleEndian) { Data($0) }
        return data + Data(header.utf8) + body
    }

    /// An uncompressed zip, as `np.savez` writes one.
    static func zip(_ entries: [(String, Data)]) -> Data {
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

    static func write(_ data: Data) throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("\(UUID().uuidString).npz")
        try data.write(to: url)
        return url
    }

    /// Every parameter of `module` under its module path, plus the archive metadata
    /// a quantized module needs, with the blocks interleaved as `np.savez` leaves them.
    static func write(
        _ module: Module, quantization: (groupSize: Int, bits: Int)? = nil, excluding: Set<String> = []
    ) throws -> URL {
        let descr: [DType: String] = [.float16: "<f2", .float32: "<f4", .uint32: "<u4"]
        var entries = module.parameters().flattened().filter { !excluding.contains($0.0) }.map { key, value in
            ("\(key).npy", npy(descr: descr[value.dtype]!, shape: value.shape, body: value.asData(access: .copy).data))
        }
        if let quantization {
            let metadata = Data(#"{"format_version":1,"bits":\#(quantization.bits),"group_size":\#(quantization.groupSize),"mode":"affine"}"#.utf8)
            entries.append(("QUANTIZATION.npy", npy(descr: "|u1", shape: [metadata.count], body: metadata)))
        }
        entries.sort { String($0.0.reversed()) < String($1.0.reversed()) }
        return try write(zip(entries))
    }
}

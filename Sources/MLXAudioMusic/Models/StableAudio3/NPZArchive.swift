import Foundation
import MLX

public enum NPZArchiveError: LocalizedError {
    case notAZipArchive(URL)
    case compressedEntry(String)
    case missingEntry(String)
    case malformedHeader(String)
    case unsupportedDType(String, String)
    case corrupt(URL)

    public var errorDescription: String? {
        switch self {
        case .notAZipArchive(let url): "Not an .npz archive: \(url.path)"
        case .compressedEntry(let name): "Compressed .npz entry is not supported: \(name)"
        case .missingEntry(let name): "Missing .npz entry: \(name)"
        case .malformedHeader(let name): "Malformed .npy header: \(name)"
        case .unsupportedDType(let name, let descr): "Unsupported .npy dtype \(descr) in \(name)"
        case .corrupt(let url): "Corrupt .npz archive: \(url.path)"
        }
    }
}

/// Reads arrays straight out of an `.npz` written by `np.savez`.
///
/// `np.savez` stores entries uncompressed, so every array is a byte range of the
/// memory-mapped file: nothing is decompressed and the mapping costs no dirty memory.
/// Stable Audio 3 publishes its MLX weights in this format.
public struct NPZArchive {
    public let url: URL
    private let data: Data
    private let payloads: [String: Range<Int>]

    public init(url: URL) throws {
        self.url = url
        data = try Data(contentsOf: url, options: .alwaysMapped)
        payloads = try Self.readCentralDirectory(data, url: url)
    }

    /// Entry names without the `.npy` suffix.
    public var names: [String] { payloads.keys.sorted() }

    public func contains(_ name: String) -> Bool { payloads[name] != nil }

    public func array(_ name: String) throws -> MLXArray {
        let (descr, shape, body) = try header(name)
        let itemSize: Int
        switch descr {
        case "<f2": itemSize = 2
        case "<f4", "<i4": itemSize = 4
        case "<i8", "<f8": itemSize = 8
        case "|u1", "|b1": itemSize = 1
        default: throw NPZArchiveError.unsupportedDType(name, descr)
        }
        let (count, overflow) = shape.reduce((1, false)) { partial, dim in
            let product = partial.0.multipliedReportingOverflow(by: dim)
            return (product.partialValue, partial.1 || product.overflow)
        }
        guard !overflow, count.multipliedReportingOverflow(by: itemSize) == (body.count, false) else {
            throw NPZArchiveError.malformedHeader(name)
        }
        func make<T: HasDType>(_ type: T.Type) -> MLXArray {
            MLXArray(data[body], shape, type: type)
        }
        switch descr {
        case "<f2": return make(Float16.self)
        case "<f4": return make(Float32.self)
        case "<i4": return make(Int32.self)
        case "<i8": return make(Int64.self)
        case "|u1": return make(UInt8.self)
        case "|b1": return make(Bool.self)
        default:
            // float64: MLX has no float64 on the GPU; every consumer here wants float32.
            let values = data[body].withUnsafeBytes { Array($0.bindMemory(to: Double.self)) }
            return MLXArray(values.map(Float.init), shape)
        }
    }

    /// The raw element bytes of an entry, for byte blobs such as an embedded tokenizer.
    public func bytes(_ name: String) throws -> Data {
        Data(data[try header(name).body])
    }

    public func arrays(where include: (String) -> Bool = { _ in true }) throws -> [String: MLXArray] {
        var result: [String: MLXArray] = [:]
        for name in payloads.keys where include(name) {
            result[name] = try array(name)
        }
        return result
    }

    // MARK: - .npy

    private func header(_ name: String) throws -> (descr: String, shape: [Int], body: Range<Int>) {
        guard let payload = payloads[name] else { throw NPZArchiveError.missingEntry(name) }
        let start = payload.lowerBound
        let malformed = NPZArchiveError.malformedHeader(name)
        guard payload.count >= 12,
              data[start] == 0x93, data[start + 1..<start + 6].elementsEqual("NUMPY".utf8)
        else { throw malformed }
        let major = data[start + 6]
        let headerLength: Int
        let headerStart: Int
        if major == 1 {
            headerLength = Int(try data.readLE(UInt16.self, at: start + 8, or: malformed))
            headerStart = start + 10
        } else {
            headerLength = Int(try data.readLE(UInt32.self, at: start + 8, or: malformed))
            headerStart = start + 12
        }
        guard headerStart + headerLength <= payload.upperBound,
              let text = String(data: data[headerStart..<headerStart + headerLength], encoding: .ascii)
        else { throw NPZArchiveError.malformedHeader(name) }

        guard let descr = text.firstCapture(#"'descr':\s*'([^']+)'"#),
              text.firstCapture(#"'fortran_order':\s*(\w+)"#) == "False",
              let shapeText = text.firstCapture(#"'shape':\s*\(([^)]*)\)"#)
        else { throw NPZArchiveError.malformedHeader(name) }
        let dimensions = shapeText.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
        let shape = dimensions.compactMap { Int($0) }
        // MLX indexes with Int32, even when another dimension makes the array empty.
        guard shape.count == dimensions.count, shape.allSatisfy({ (0...Int(Int32.max)).contains($0) })
        else { throw malformed }
        return (descr, shape, (headerStart + headerLength)..<payload.upperBound)
    }

    // MARK: - Zip

    private static func readCentralDirectory(_ data: Data, url: URL) throws -> [String: Range<Int>] {
        let corrupt = NPZArchiveError.corrupt(url)
        func read<T: FixedWidthInteger>(_ type: T.Type, _ offset: Int) throws -> T {
            try data.readLE(type, at: offset, or: corrupt)
        }
        func checked(_ start: Int, _ length: Int) throws -> Range<Int> {
            guard start >= 0, length >= 0, start <= data.count - length else { throw corrupt }
            return start..<(start + length)
        }

        let endSignature: UInt32 = 0x0605_4B50
        let searchStart = max(0, data.count - 22 - 0xFFFF)
        guard data.count >= 22,
              let end = try stride(from: data.count - 22, through: searchStart, by: -1)
                  .first(where: { try read(UInt32.self, $0) == endSignature })
        else { throw NPZArchiveError.notAZipArchive(url) }

        var entryCount = Int(try read(UInt16.self, end + 10))
        var directoryOffset = Int(try read(UInt32.self, end + 16))
        // Python's zipfile writes a zip64 end record once the archive passes 2 GiB.
        let locator = end - 20
        if locator >= 0, try read(UInt32.self, locator) == 0x0706_4B50 {
            let record = try read(UInt64.self, locator + 8)
            guard record < UInt64(data.count), try read(UInt32.self, Int(record)) == 0x0606_4B50 else {
                throw corrupt
            }
            let count = try read(UInt64.self, Int(record) + 32)
            let offset = try read(UInt64.self, Int(record) + 48)
            guard count < UInt64(data.count), offset < UInt64(data.count) else { throw corrupt }
            entryCount = Int(count)
            directoryOffset = Int(offset)
        }

        var payloads: [String: Range<Int>] = [:]
        var cursor = directoryOffset
        for _ in 0..<entryCount {
            guard try read(UInt32.self, cursor) == 0x0201_4B50 else { throw corrupt }
            let method = try read(UInt16.self, cursor + 10)
            var size = UInt64(try read(UInt32.self, cursor + 20))
            let nameLength = Int(try read(UInt16.self, cursor + 28))
            let extraLength = Int(try read(UInt16.self, cursor + 30))
            let commentLength = Int(try read(UInt16.self, cursor + 32))
            var localOffset = UInt64(try read(UInt32.self, cursor + 42))
            let nameRange = try checked(cursor + 46, nameLength)
            let name = String(decoding: data[nameRange], as: UTF8.self)

            // Zip64 extra field: 64-bit values replace exactly the 32-bit fields
            // stored as 0xFFFFFFFF, in the order uncompressed, compressed, offset.
            if size == 0xFFFF_FFFF || localOffset == 0xFFFF_FFFF {
                let uncompressedIsWide = try read(UInt32.self, cursor + 24) == 0xFFFF_FFFF
                let extraRange = try checked(nameRange.upperBound, extraLength)
                var extra = extraRange.lowerBound
                while extra + 4 <= extraRange.upperBound {
                    let id = try read(UInt16.self, extra)
                    let length = Int(try read(UInt16.self, extra + 2))
                    if id == 0x0001 {
                        var field = extra + 4
                        if uncompressedIsWide { field += 8 }
                        if size == 0xFFFF_FFFF {
                            size = try read(UInt64.self, field)
                            field += 8
                        }
                        if localOffset == 0xFFFF_FFFF {
                            localOffset = try read(UInt64.self, field)
                        }
                    }
                    extra += 4 + length
                }
            }

            guard method == 0 else { throw NPZArchiveError.compressedEntry(name) }
            guard localOffset < UInt64(data.count), size < UInt64(data.count) else { throw corrupt }
            let local = Int(localOffset)
            guard try read(UInt32.self, local) == 0x0403_4B50 else { throw corrupt }
            let localName = Int(try read(UInt16.self, local + 26))
            let localExtra = Int(try read(UInt16.self, local + 28))
            let key = name.hasSuffix(".npy") ? String(name.dropLast(4)) : name
            payloads[key] = try checked(local + 30 + localName + localExtra, Int(size))
            cursor = nameRange.upperBound + extraLength + commentLength
        }
        return payloads
    }
}

private extension Data {
    /// A little-endian integer at `offset`, or `failure` when it does not fit in the data.
    func readLE<T: FixedWidthInteger>(_ type: T.Type, at offset: Int, or failure: Error) throws -> T {
        guard offset >= 0, offset <= count - MemoryLayout<T>.size else { throw failure }
        var value = T.zero
        Swift.withUnsafeMutableBytes(of: &value) { target in
            self.copyBytes(to: target, from: offset..<(offset + MemoryLayout<T>.size))
        }
        return T(littleEndian: value)
    }
}

private extension String {
    func firstCapture(_ pattern: String) -> String? {
        guard let regex = try? NSRegularExpression(pattern: pattern),
              let match = regex.firstMatch(in: self, range: NSRange(startIndex..., in: self)),
              let range = Range(match.range(at: 1), in: self)
        else { return nil }
        return String(self[range])
    }
}

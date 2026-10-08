import Foundation
import MLX

public enum NPZArchiveError: LocalizedError {
    case notAZipArchive(URL)
    case compressedEntry(String)
    case missingEntry(String)
    case malformedHeader(String)
    case unsupportedDType(String, String)

    public var errorDescription: String? {
        switch self {
        case .notAZipArchive(let url): "Not an .npz archive: \(url.path)"
        case .compressedEntry(let name): "Compressed .npz entry is not supported: \(name)"
        case .missingEntry(let name): "Missing .npz entry: \(name)"
        case .malformedHeader(let name): "Malformed .npy header: \(name)"
        case .unsupportedDType(let name, let descr): "Unsupported .npy dtype \(descr) in \(name)"
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
        case "<f8":
            // MLX has no float64 on the GPU; every consumer here wants float32.
            let values = data[body].withUnsafeBytes { Array($0.bindMemory(to: Double.self)) }
            return MLXArray(values.map(Float.init), shape)
        default: throw NPZArchiveError.unsupportedDType(name, descr)
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
        guard payload.count >= 10,
              data[start] == 0x93, data[start + 1..<start + 6].elementsEqual("NUMPY".utf8)
        else { throw NPZArchiveError.malformedHeader(name) }
        let major = data[start + 6]
        let headerLength: Int
        let headerStart: Int
        if major == 1 {
            headerLength = Int(data.readLE(UInt16.self, at: start + 8))
            headerStart = start + 10
        } else {
            headerLength = Int(data.readLE(UInt32.self, at: start + 8))
            headerStart = start + 12
        }
        guard headerStart + headerLength <= payload.upperBound,
              let text = String(data: data[headerStart..<headerStart + headerLength], encoding: .ascii)
        else { throw NPZArchiveError.malformedHeader(name) }

        guard let descr = text.firstCapture(#"'descr':\s*'([^']+)'"#),
              text.firstCapture(#"'fortran_order':\s*(\w+)"#) == "False",
              let shapeText = text.firstCapture(#"'shape':\s*\(([^)]*)\)"#)
        else { throw NPZArchiveError.malformedHeader(name) }
        let shape = shapeText.split(separator: ",").compactMap {
            Int($0.trimmingCharacters(in: .whitespaces))
        }
        return (descr, shape, (headerStart + headerLength)..<payload.upperBound)
    }

    // MARK: - Zip

    private static func readCentralDirectory(_ data: Data, url: URL) throws -> [String: Range<Int>] {
        let endSignature: UInt32 = 0x0605_4B50
        let searchStart = max(0, data.count - 22 - 0xFFFF)
        guard data.count >= 22,
              let end = stride(from: data.count - 22, through: searchStart, by: -1)
                  .first(where: { data.readLE(UInt32.self, at: $0) == endSignature })
        else { throw NPZArchiveError.notAZipArchive(url) }

        var entryCount = Int(data.readLE(UInt16.self, at: end + 10))
        var directoryOffset = Int(data.readLE(UInt32.self, at: end + 16))
        // Python's zipfile writes a zip64 end record once the archive passes 2 GiB.
        let locator = end - 20
        if locator >= 0, data.readLE(UInt32.self, at: locator) == 0x0706_4B50 {
            let record = Int(data.readLE(UInt64.self, at: locator + 8))
            guard data.readLE(UInt32.self, at: record) == 0x0606_4B50 else {
                throw NPZArchiveError.notAZipArchive(url)
            }
            entryCount = Int(data.readLE(UInt64.self, at: record + 32))
            directoryOffset = Int(data.readLE(UInt64.self, at: record + 48))
        }

        var payloads: [String: Range<Int>] = [:]
        var cursor = directoryOffset
        for _ in 0..<entryCount {
            guard data.readLE(UInt32.self, at: cursor) == 0x0201_4B50 else {
                throw NPZArchiveError.notAZipArchive(url)
            }
            let method = data.readLE(UInt16.self, at: cursor + 10)
            var size = Int(data.readLE(UInt32.self, at: cursor + 20))
            let nameLength = Int(data.readLE(UInt16.self, at: cursor + 28))
            let extraLength = Int(data.readLE(UInt16.self, at: cursor + 30))
            let commentLength = Int(data.readLE(UInt16.self, at: cursor + 32))
            var localOffset = Int(data.readLE(UInt32.self, at: cursor + 42))
            let nameStart = cursor + 46
            let name = String(decoding: data[nameStart..<nameStart + nameLength], as: UTF8.self)

            // Zip64 extra field: 64-bit values replace exactly the 32-bit fields
            // stored as 0xFFFFFFFF, in the order uncompressed, compressed, offset.
            if size == 0xFFFF_FFFF || localOffset == 0xFFFF_FFFF {
                let uncompressedIsWide = Int(data.readLE(UInt32.self, at: cursor + 24)) == 0xFFFF_FFFF
                var extra = nameStart + nameLength
                let extraEnd = extra + extraLength
                while extra + 4 <= extraEnd {
                    let id = data.readLE(UInt16.self, at: extra)
                    let length = Int(data.readLE(UInt16.self, at: extra + 2))
                    if id == 0x0001 {
                        var field = extra + 4
                        if uncompressedIsWide { field += 8 }
                        if size == 0xFFFF_FFFF {
                            size = Int(data.readLE(UInt64.self, at: field))
                            field += 8
                        }
                        if localOffset == 0xFFFF_FFFF {
                            localOffset = Int(data.readLE(UInt64.self, at: field))
                        }
                    }
                    extra += 4 + length
                }
            }

            guard method == 0 else { throw NPZArchiveError.compressedEntry(name) }
            guard data.readLE(UInt32.self, at: localOffset) == 0x0403_4B50 else {
                throw NPZArchiveError.notAZipArchive(url)
            }
            let localName = Int(data.readLE(UInt16.self, at: localOffset + 26))
            let localExtra = Int(data.readLE(UInt16.self, at: localOffset + 28))
            let start = localOffset + 30 + localName + localExtra
            let key = name.hasSuffix(".npy") ? String(name.dropLast(4)) : name
            payloads[key] = start..<(start + size)
            cursor = nameStart + nameLength + extraLength + commentLength
        }
        return payloads
    }
}

private extension Data {
    func readLE<T: FixedWidthInteger>(_ type: T.Type, at offset: Int) -> T {
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

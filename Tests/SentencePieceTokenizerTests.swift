import Foundation
import Testing

@testable import MLXAudioCore

struct SentencePieceDummyPrefixTests {
    /// `<unk>`, `▁`, `a`, `▁a`, with an optional `normalizer_spec.add_dummy_prefix`.
    private func model(addDummyPrefix: Bool?) -> Data {
        func varint(_ value: UInt64) -> Data {
            var value = value
            var bytes = Data()
            repeat {
                var byte = UInt8(value & 0x7F)
                value >>= 7
                if value != 0 { byte |= 0x80 }
                bytes.append(byte)
            } while value != 0
            return bytes
        }
        func field(_ number: Int, _ payload: Data) -> Data {
            varint(UInt64(number << 3 | 2)) + varint(UInt64(payload.count)) + payload
        }
        func piece(_ token: String, score: Float, type: UInt64 = 1) -> Data {
            var body = field(1, Data(token.utf8))
            body += varint(UInt64(2 << 3 | 5))
            body += withUnsafeBytes(of: score.bitPattern.littleEndian) { Data($0) }
            body += varint(UInt64(3 << 3)) + varint(type)
            return field(1, body)
        }
        var data = piece("<unk>", score: 0, type: 2)
            + piece("▁", score: -1) + piece("a", score: -2) + piece("▁a", score: -1.5)
        if let addDummyPrefix {
            data += field(3, varint(UInt64(3 << 3)) + varint(addDummyPrefix ? 1 : 0))
        }
        return data
    }

    @Test func dummyPrefixIsAddedByDefault() throws {
        let tokenizer = try SentencePieceTokenizer(sentencePieceModelData: model(addDummyPrefix: nil))
        #expect(tokenizer.encodeWithByteFallback("a") == [3])
    }

    @Test func modelCanTurnTheDummyPrefixOff() throws {
        // Gemma's tokenizer (used by T5Gemma) sets add_dummy_prefix = false.
        let tokenizer = try SentencePieceTokenizer(sentencePieceModelData: model(addDummyPrefix: false))
        #expect(tokenizer.encodeWithByteFallback("a") == [2])
    }
}

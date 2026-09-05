import Foundation

/// Splits text into sentence-level chunks for incremental synthesis with
/// non-autoregressive models (Kokoro, KittenTTS): each chunk is synthesized
/// independently and streamed as soon as it is ready, so first-audio latency
/// is one sentence instead of the whole input.
enum SentenceChunker {
    private static let terminators: Set<Character> = [".", "!", "?", "…", ";", "\n"]

    /// - Parameters:
    ///   - minLength: pieces shorter than this merge with the following piece
    ///     (avoids synthesizing fragments like "Dr." or list numbers alone).
    ///   - maxLength: pieces longer than this are split again on commas, then
    ///     hard-wrapped on whitespace (guards model token limits).
    static func chunks(from text: String, minLength: Int = 4, maxLength: Int? = nil) -> [String] {
        var pieces: [String] = []
        var current = ""

        for character in text {
            current.append(character)
            if terminators.contains(character) {
                appendTrimmed(current, to: &pieces)
                current = ""
            }
        }
        appendTrimmed(current, to: &pieces)

        var merged: [String] = []
        for piece in pieces {
            if let last = merged.last, last.count < minLength {
                merged[merged.count - 1] = last + " " + piece
            } else {
                merged.append(piece)
            }
        }
        // A trailing fragment below minLength joins the previous sentence.
        if merged.count >= 2, let last = merged.last, last.count < minLength {
            merged.removeLast()
            merged[merged.count - 1] += " " + last
        }

        guard let maxLength else { return merged }
        return merged.flatMap { split($0, by: ",", maxLength: maxLength) }
    }

    private static func appendTrimmed(_ piece: String, to pieces: inout [String]) {
        let trimmed = piece.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmed.isEmpty {
            pieces.append(trimmed)
        }
    }

    private static func split(_ piece: String, by separator: Character, maxLength: Int) -> [String] {
        guard piece.count > maxLength else { return [piece] }

        let parts = piece.split(separator: separator).map {
            $0.trimmingCharacters(in: .whitespaces)
        }.filter { !$0.isEmpty }

        // No separator to split on: hard-wrap on whitespace.
        guard parts.count > 1 else { return hardWrap(piece, maxLength: maxLength) }

        // Re-pack comma parts greedily up to maxLength.
        var result: [String] = []
        var current = ""
        for part in parts {
            if current.isEmpty {
                current = part
            } else if current.count + part.count + 2 <= maxLength {
                current += ", " + part
            } else {
                result.append(current)
                current = part
            }
        }
        if !current.isEmpty {
            result.append(current)
        }
        return result.flatMap { $0.count > maxLength ? hardWrap($0, maxLength: maxLength) : [$0] }
    }

    private static func hardWrap(_ piece: String, maxLength: Int) -> [String] {
        var result: [String] = []
        var current = ""
        for word in piece.split(whereSeparator: { $0.isWhitespace }) {
            if current.isEmpty {
                current = String(word)
            } else if current.count + word.count + 1 <= maxLength {
                current += " " + word
            } else {
                result.append(current)
                current = String(word)
            }
        }
        if !current.isEmpty {
            result.append(current)
        }
        return result
    }
}

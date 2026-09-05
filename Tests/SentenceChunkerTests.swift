import Testing
@testable import MLXAudioTTS

struct SentenceChunkerTests {
    @Test func singleSentencePassesThrough() {
        #expect(SentenceChunker.chunks(from: "Hello there, how are you?") == ["Hello there, how are you?"])
    }

    @Test func splitsOnSentenceBoundaries() {
        let chunks = SentenceChunker.chunks(from: "First sentence. Second one! And a third?")
        #expect(chunks == ["First sentence.", "Second one!", "And a third?"])
    }

    @Test func mergesShortFragmentsForward() {
        let chunks = SentenceChunker.chunks(from: "No. This continues the thought. Done now.", minLength: 4)
        #expect(chunks == ["No. This continues the thought.", "Done now."])
    }

    @Test func trailingShortFragmentJoinsPrevious() {
        let chunks = SentenceChunker.chunks(from: "A complete first sentence here. Ok.", minLength: 4)
        #expect(chunks == ["A complete first sentence here. Ok."])
    }

    @Test func emptyAndWhitespaceInputYieldsNothing() {
        #expect(SentenceChunker.chunks(from: "").isEmpty)
        #expect(SentenceChunker.chunks(from: "  \n  ").isEmpty)
    }

    @Test func longSentenceSplitsOnCommasUnderMaxLength() {
        let long = (1...20).map { "clause number \($0) keeps going" }.joined(separator: ", ") + "."
        let chunks = SentenceChunker.chunks(from: long, maxLength: 120)
        #expect(chunks.count > 1)
        #expect(chunks.allSatisfy { $0.count <= 120 })
        let words = chunks.joined(separator: " ").split(separator: " ").filter { $0 != "" }
        #expect(words.filter { $0.hasPrefix("clause") }.count == 20)
    }

    @Test func commalessLongSentenceHardWrapsOnWords() {
        let long = Array(repeating: "word", count: 200).joined(separator: " ") + "."
        let chunks = SentenceChunker.chunks(from: long, maxLength: 100)
        #expect(chunks.count > 1)
        #expect(chunks.allSatisfy { $0.count <= 100 })
    }

    @Test func newlinesActAsBoundaries() {
        let chunks = SentenceChunker.chunks(from: "Line one without period\nLine two here")
        #expect(chunks == ["Line one without period", "Line two here"])
    }
}

import Foundation
import AVFoundation
import Speech

// Apple's on-device SpeechAnalyzer (macOS 26 / iOS 26) as a TEST-ONLY reference.
// It natively emits volatile (interim) → finalized results on the ANE — i.e. the
// DeepGram-style interim→final pattern, for free. We use it ONLY to measure
// whether the native ANE path beats our own stack; it is NOT a production
// dependency (Apple-platform-locked, black box, API churn). Production lives in
// the vendor-independent Local-Agreement-over-Voxtral path.

@available(macOS 26.0, iOS 26.0, *)
final class SpeechAnalyzerASR: LiveASR, @unchecked Sendable {  // all mutable state is NSLock-guarded
    let label = "APPLE SpeechAnalyzer ·reference"
    let isLocal = true

    private let lock = NSLock()
    private var snap: Snap

    private let transcriber: SpeechTranscriber
    private let analyzer: SpeechAnalyzer
    private let analyzerFormat: AVAudioFormat
    private let inputBuilder: AsyncStream<AnalyzerInput>.Continuation
    private let srcFormat: AVAudioFormat
    private var converter: AVAudioConverter?

    private var finalized = ""
    private var volatileText = ""
    private var speechWall = -1.0, firstText = -1.0, lastWallNow = 0.0

    enum Err: Error { case noFormat, noConverter, convertFailed(NSError?) }

    /// Async factory: installs the locale model asset, negotiates the audio
    /// format, and starts the analyzer before returning a ready provider.
    static func create(locale localeId: String) async throws -> SpeechAnalyzerASR {
        let transcriber = SpeechTranscriber(
            locale: Locale(identifier: localeId),
            transcriptionOptions: [],
            reportingOptions: [.volatileResults],
            attributeOptions: [])
        if let req = try await AssetInventory.assetInstallationRequest(supporting: [transcriber]) {
            FileHandle.standardError.write(Data("Apple: downloading \(localeId) model asset (one-time)...\n".utf8))
            try await req.downloadAndInstall()
        }
        let analyzer = SpeechAnalyzer(modules: [transcriber])
        guard let fmt = await SpeechAnalyzer.bestAvailableAudioFormat(compatibleWith: [transcriber]) else {
            throw Err.noFormat
        }
        let (seq, builder) = AsyncStream<AnalyzerInput>.makeStream()
        try await analyzer.start(inputSequence: seq)
        return SpeechAnalyzerASR(transcriber: transcriber, analyzer: analyzer, format: fmt, builder: builder)
    }

    private init(transcriber: SpeechTranscriber, analyzer: SpeechAnalyzer,
                 format: AVAudioFormat, builder: AsyncStream<AnalyzerInput>.Continuation) {
        self.transcriber = transcriber
        self.analyzer = analyzer
        self.analyzerFormat = format
        self.inputBuilder = builder
        self.srcFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16000, channels: 1, interleaved: false)!
        self.snap = Snap(label: label, note: "reference · on-device ANE")
        // Drain results: volatile guesses + finalized corrections, like DeepGram.
        Task { [weak self] in
            guard let self else { return }
            do {
                for try await result in transcriber.results {
                    self.ingest(String(result.text.characters), isFinal: result.isFinal)
                }
            } catch {
                self.markErr()
            }
        }
    }

    // Synchronous (NSLock can't be used from the async results loop directly).
    private func ingest(_ text: String, isFinal: Bool) {
        lock.lock(); defer { lock.unlock() }
        if isFinal { finalized += text; volatileText = "" } else { volatileText = text }
        let joined = finalized + volatileText
        if firstText < 0, !joined.trimmingCharacters(in: .whitespaces).isEmpty, speechWall >= 0 {
            firstText = lastWallNow - speechWall
        }
        snap.text = joined
        snap.ttft = firstText
        snap.perf = "ANE"
    }

    private func markErr() { lock.lock(); defer { lock.unlock() }; snap.note = "reference · err" }

    func feed(_ f: AudioFrame) {
        lock.lock()
        lastWallNow = f.wallNow
        if f.speechActive, speechWall < 0 { speechWall = f.wallNow }
        lock.unlock()
        guard !f.samples.isEmpty,
              let inBuf = AVAudioPCMBuffer(pcmFormat: srcFormat, frameCapacity: AVAudioFrameCount(f.samples.count))
        else { return }
        inBuf.frameLength = AVAudioFrameCount(f.samples.count)
        f.samples.withUnsafeBufferPointer { src in
            inBuf.floatChannelData![0].update(from: src.baseAddress!, count: src.count)
        }
        if let out = try? convert(inBuf, to: analyzerFormat) {
            inputBuilder.yield(AnalyzerInput(buffer: out))
        }
    }

    // Resample/repack to the analyzer's negotiated format. The AnalyzerInput
    // buffer MUST match bestAvailableAudioFormat() exactly or start() throws.
    private func convert(_ buffer: AVAudioPCMBuffer, to format: AVAudioFormat) throws -> AVAudioPCMBuffer {
        if buffer.format == format { return buffer }
        if converter == nil || converter?.outputFormat != format {
            converter = AVAudioConverter(from: buffer.format, to: format)
            converter?.primeMethod = .none
        }
        guard let converter else { throw Err.noConverter }
        let ratio = converter.outputFormat.sampleRate / converter.inputFormat.sampleRate
        let cap = AVAudioFrameCount((Double(buffer.frameLength) * ratio).rounded(.up))
        guard let outBuf = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: cap) else { throw Err.convertFailed(nil) }
        var nsErr: NSError?
        nonisolated(unsafe) var done = false   // block runs synchronously; safe
        let status = converter.convert(to: outBuf, error: &nsErr) { _, statusPtr in
            defer { done = true }
            statusPtr.pointee = done ? .noDataNow : .haveData
            return done ? nil : buffer
        }
        if status == .error { throw Err.convertFailed(nsErr) }
        return outBuf
    }

    func finish() {
        inputBuilder.finish()
        // Drain so trailing finalized results land before the snapshot is read.
        let sem = DispatchSemaphore(value: 0)
        Task { [analyzer] in
            try? await analyzer.finalizeAndFinishThroughEndOfInput()
            try? await Task.sleep(nanoseconds: 250_000_000)
            sem.signal()
        }
        sem.wait()
    }

    func snapshot() -> Snap { lock.lock(); defer { lock.unlock() }; return snap }
}

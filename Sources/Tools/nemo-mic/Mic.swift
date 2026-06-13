@preconcurrency import AVFoundation
import Darwin
import Foundation
import MLX
import MLXAudioCore
import MLXAudioSTT

// Live-mic demo for NemotronASRStreamSession with a pinned stats line.
//
// Captures the default input via AVAudioEngine, resamples to 16 kHz mono with a
// single stateful AVAudioConverter, and feeds fixed chunks into session.step().
// Decoded text scrolls in the terminal; a status line is pinned to the bottom
// row (TTFT, per-step compute, RTF, counts) via an ANSI scroll region.
//
// All audio wiring lives in MicRunner (nonisolated @unchecked Sendable) so the
// input-tap closure does NOT inherit main()'s actor isolation — otherwise
// AVAudioEngine invoking it from the realtime audio thread trips a Swift
// concurrency executor assertion (EXC_BREAKPOINT).
//
// macOS mic permission: allow the terminal under
// System Settings → Privacy & Security → Microphone (first run may prompt).

private final class Flag: @unchecked Sendable { var done = false }

/// Minimal ANSI pinned-bottom-line helper. No-op when stdout is not a TTY.
private struct StatusBar {
    let rows: Int
    let isTTY: Bool

    init() {
        isTTY = isatty(STDOUT_FILENO) != 0
        var w = winsize()
        let ok = ioctl(STDOUT_FILENO, TIOCGWINSZ, &w) == 0 && w.ws_row > 2
        rows = ok ? Int(w.ws_row) : 24
    }

    private func write(_ s: String) { FileHandle.standardOutput.write(Data(s.utf8)) }

    /// Clear screen, reserve the bottom row, put the cursor at the top.
    func begin() {
        guard isTTY else { return }
        write("\u{1B}[2J\u{1B}[1;\(rows - 1)r\u{1B}[1;1H")
    }

    /// Redraw the pinned line without disturbing the scrolling text cursor.
    func render(_ line: String) {
        guard isTTY else { return }
        write("\u{1B}7\u{1B}[\(rows);1H\u{1B}[2K\u{1B}[7m \(line) \u{1B}[0m\u{1B}8")
    }

    /// Release the scroll region and drop to the bottom.
    func end() {
        guard isTTY else { return }
        write("\u{1B}[r\u{1B}[\(rows);1H\n")
    }
}

/// Owns the engine + converter + session; session access and all counters are
/// serialized on one queue. Nonisolated, so the tap closure has no isolation.
private final class MicRunner: @unchecked Sendable {
    private let session: NemotronASRStreamSession
    private let feedSamples: Int
    private let chunkMsLabel: String
    private let queue = DispatchQueue(label: "nemo.mic.feed")
    private var pending: [Float] = []

    private let engine = AVAudioEngine()
    private let inFmt: AVAudioFormat
    private let outFmt: AVAudioFormat
    private let converter: AVAudioConverter
    private let bar = StatusBar()
    private var ticker: DispatchSourceTimer?
    private var dumpFile: AVAudioFile?

    // Metrics (mutated only on `queue`).
    private var startTime: CFAbsoluteTime = 0
    private var speechStartWall: CFAbsoluteTime = 0
    private var speechDetected = false
    private var firstTextDelay: Double = -1
    private var audioSamplesFed = 0
    private var stepCount = 0
    private var lastStepMs = 0.0
    private var stepMsTotal = 0.0

    init(session: NemotronASRStreamSession, feedSamples: Int, chunkMsLabel: String,
         inputDevice: AudioDeviceID?, dumpURL: URL?) {
        self.session = session
        self.feedSamples = feedSamples
        self.chunkMsLabel = chunkMsLabel

        // Point the input node at the chosen device BEFORE reading its format.
        let devID = inputDevice ?? AudioDevices.defaultInput()
        if let devID {
            try? AudioDevices.setInput(devID, on: engine)
            let fmt = engine.inputNode.outputFormat(forBus: 0)
            let info = "INPUT: \(AudioDevices.name(of: devID)) | uid=\(AudioDevices.uid(of: devID)) | "
                + "\(Int(fmt.sampleRate))Hz \(fmt.channelCount)ch\n"
            FileHandle.standardError.write(Data(info.utf8))
        }

        self.inFmt = engine.inputNode.outputFormat(forBus: 0)
        guard let out = AVAudioFormat(
            commonFormat: .pcmFormatFloat32, sampleRate: 16000, channels: 1, interleaved: false
        ), let conv = AVAudioConverter(from: inFmt, to: out) else {
            fatalError("could not build 16 kHz mono converter from \(inFmt)")
        }
        self.outFmt = out
        self.converter = conv
        if let dumpURL {
            dumpFile = try? AVAudioFile(forWriting: dumpURL, settings: out.settings)
        }
    }

    func start() throws {
        startTime = CFAbsoluteTimeGetCurrent()
        bar.begin()
        engine.inputNode.installTap(onBus: 0, bufferSize: 4096, format: inFmt) { [self] buffer, _ in
            let floats = convert(buffer)
            if !floats.isEmpty { feed(floats) }
        }
        engine.prepare()
        try engine.start()
        // Refresh the pinned line ~4×/s so wall/RTF tick even between chunks.
        let t = DispatchSource.makeTimerSource(queue: queue)
        t.schedule(deadline: .now() + 0.25, repeating: 0.25)
        t.setEventHandler { [self] in renderStatus() }
        t.resume()
        ticker = t
    }

    func stop() -> String {
        ticker?.cancel()
        engine.stop()
        engine.inputNode.removeTap(onBus: 0)
        queue.sync {
            if !pending.isEmpty {
                emit(session.step(pending))
                pending = []
            }
            emit(session.finish())
            renderStatus()
        }
        bar.end()
        let audioS = Double(audioSamplesFed) / 16000.0
        let avgMs = stepCount > 0 ? stepMsTotal / Double(stepCount) : 0
        let rtf = audioS > 0 ? (stepMsTotal / 1000.0) / audioS : 0
        let ttft = firstTextDelay >= 0 ? String(format: "%.2fs", firstTextDelay) : "—"
        let lag = max(0, (CFAbsoluteTimeGetCurrent() - startTime) - audioS)
        FileHandle.standardError.write(Data(String(
            format: "STATS audio=%.1fs chunks=%d avg_step=%.0fms RTF=%.2f lag=%.2fs TTFT=%@\n",
            audioS, stepCount, avgMs, rtf, lag, ttft).utf8))
        return session.text
    }

    private func convert(_ buffer: AVAudioPCMBuffer) -> [Float] {
        let ratio = outFmt.sampleRate / inFmt.sampleRate
        let cap = AVAudioFrameCount(Double(buffer.frameLength) * ratio) + 16
        guard let out = AVAudioPCMBuffer(pcmFormat: outFmt, frameCapacity: cap) else { return [] }
        let flag = Flag()
        var err: NSError?
        converter.convert(to: out, error: &err) { _, status in
            if flag.done { status.pointee = .noDataNow; return nil }
            flag.done = true
            status.pointee = .haveData
            return buffer
        }
        guard err == nil, out.frameLength > 0, let ch = out.floatChannelData else { return [] }
        if let dumpFile { try? dumpFile.write(from: out) }
        return Array(UnsafeBufferPointer(start: ch[0], count: Int(out.frameLength)))
    }

    private func feed(_ floats: [Float]) {
        queue.async { [self] in
            pending.append(contentsOf: floats)
            while pending.count >= feedSamples {
                let chunk = Array(pending.prefix(feedSamples))
                pending.removeFirst(feedSamples)
                // Speech onset (RMS gate ~ −40 dB) so TTFT measures from when you
                // start talking, not from launch (which counts pre-speech silence).
                if !speechDetected {
                    var sum: Float = 0
                    for v in chunk { sum += v * v }
                    if (sum / Float(chunk.count)).squareRoot() > 0.01 {
                        speechDetected = true
                        speechStartWall = CFAbsoluteTimeGetCurrent()
                    }
                }
                let t0 = CFAbsoluteTimeGetCurrent()
                let delta = session.step(chunk)
                lastStepMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000
                stepMsTotal += lastStepMs
                stepCount += 1
                audioSamplesFed += chunk.count
                if firstTextDelay < 0, !delta.text.trimmingCharacters(in: .whitespaces).isEmpty {
                    firstTextDelay = CFAbsoluteTimeGetCurrent() - (speechDetected ? speechStartWall : startTime)
                }
                emit(delta)
            }
            renderStatus()
        }
    }

    private func emit(_ delta: NemotronASRStreamSession.Delta) {
        guard !delta.text.isEmpty else { return }
        FileHandle.standardOutput.write(Data(delta.text.utf8))
    }

    private func renderStatus() {
        let wall = CFAbsoluteTimeGetCurrent() - startTime
        let audioS = Double(audioSamplesFed) / 16000.0
        let avgMs = stepCount > 0 ? stepMsTotal / Double(stepCount) : 0
        let rtf = audioS > 0 ? (stepMsTotal / 1000.0) / audioS : 0
        let words = session.text.split(whereSeparator: \.isWhitespace).count
        let ttft = firstTextDelay >= 0 ? String(format: "%.2fs", firstTextDelay) : "—"
        // Honest responsiveness: how far the processed audio trails real time
        // (capture → text). Unlike TTFT, not inflated by pre-speech silence.
        let lag = max(0, wall - audioS)
        let line = String(
            format: "nemo %@ │ wall %.1fs │ audio %.1fs │ lag %.2fs │ words %d │ step %.0f/%.0fms │ RTF %.2f │ TTFT %@",
            chunkMsLabel, wall, audioS, lag, words, lastStepMs, avgMs, rtf, ttft
        )
        bar.render(line)
    }
}

@main
struct NemoMic {
    static func main() async throws {
        GPU.set(memoryLimit: 18 * 1024 * 1024 * 1024, relaxed: false)

        var repo = "mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit"
        var language: String? = "ru"
        var chunkMs: Int? = nil
        var seconds: Double? = nil
        var feedMs = 480
        var inputQuery: String? = nil
        var dumpPath: String? = nil

        var it = CommandLine.arguments.dropFirst().makeIterator()
        while let a = it.next() {
            switch a {
            case "--model": repo = it.next() ?? repo
            case "--language": language = it.next()
            case "--chunk-ms": chunkMs = Int(it.next() ?? "")
            case "--seconds": seconds = Double(it.next() ?? "")
            case "--feed-ms": feedMs = Int(it.next() ?? "") ?? feedMs
            case "--input": inputQuery = it.next()
            case "--dump": dumpPath = it.next()
            case "--list-devices":
                let def = AudioDevices.defaultInput()
                for d in AudioDevices.inputs() {
                    let mark = d.id == def ? " (default)" : ""
                    print("\(d.name)\(mark)\n    uid: \(d.uid)")
                }
                return
            default: fatalError("unknown arg \(a)")
            }
        }

        var inputDevice: AudioDeviceID? = nil
        if let inputQuery {
            guard let found = AudioDevices.find(inputQuery) else {
                FileHandle.standardError.write(Data("no input device matching '\(inputQuery)' (try --list-devices)\n".utf8))
                exit(1)
            }
            inputDevice = found.id
        }

        FileHandle.standardError.write(Data("loading \(repo)...\n".utf8))
        let model = try await NemotronASRModel.fromPretrained(repo)

        // Warm Metal kernels on a throwaway session so the first LIVE chunk isn't
        // cold (otherwise the first step pays ~1 s of kernel JIT, inflating TTFT).
        let warm = model.makeStreamSession(language: language, chunkMs: chunkMs)
        _ = warm.step([Float](repeating: 0, count: 16000 * 2))
        _ = warm.finish()

        let session = model.makeStreamSession(language: language, chunkMs: chunkMs)
        let runner = MicRunner(
            session: session,
            feedSamples: max(1, 16000 * feedMs / 1000),
            chunkMsLabel: chunkMs.map { "\($0)ms" } ?? "native",
            inputDevice: inputDevice,
            dumpURL: dumpPath.map { URL(fileURLWithPath: $0) }
        )

        try runner.start()
        let prompt = seconds == nil
            ? "READY: speak now (press Enter to stop)\n"
            : "READY: speak now (\(Int(seconds!)) s)\n"
        FileHandle.standardError.write(Data(prompt.utf8))

        if let seconds {
            try await Task.sleep(nanoseconds: UInt64(seconds * 1_000_000_000))
        } else {
            _ = readLine()
        }

        let final = runner.stop()
        FileHandle.standardError.write(Data("FINAL: \(final)\n".utf8))
    }
}

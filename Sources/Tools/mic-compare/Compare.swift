@preconcurrency import AVFoundation
import Darwin
import Foundation
import MLX
import MLXAudioCore
import MLXAudioSTT

// Side-by-side live ASR: feeds one mic stream into BOTH Nemotron and Voxtral
// streaming sessions and shows their transcripts + metrics stacked, redrawn in
// place. Lets you judge which model reads your speech better, live.
//
// One tap → one 16 kHz mono convert → both sessions stepped on a single serial
// queue (combined RTF stays < 1, so realtime is fine). Audio wiring lives in a
// nonisolated class so the tap closure has no actor isolation.

private final class Flag: @unchecked Sendable { var done = false }

private struct Stats {
    var chunks = 0
    var lastStepMs = 0.0
    var stepMsTotal = 0.0
    var audioSamples = 0
    var firstTextDelay = -1.0
    var label = ""
    var transcript = ""
}

/// Full-screen two-block HUD, redrawn in place (no-op when stdout isn't a TTY).
private struct HUD {
    let isTTY = isatty(STDOUT_FILENO) != 0
    var cols: Int { size().0 }
    var rows: Int { size().1 }

    private func size() -> (Int, Int) {
        var w = winsize()
        if ioctl(STDOUT_FILENO, TIOCGWINSZ, &w) == 0, w.ws_col > 0 {
            return (Int(w.ws_col), Int(w.ws_row))
        }
        return (80, 24)
    }

    private func write(_ s: String) { FileHandle.standardOutput.write(Data(s.utf8)) }

    private func wrap(_ text: String, width: Int, maxLines: Int) -> [String] {
        var lines: [String] = []
        var cur = ""
        for word in text.split(separator: " ", omittingEmptySubsequences: false) {
            if cur.isEmpty { cur = String(word) }
            else if cur.count + 1 + word.count <= width { cur += " " + word }
            else { lines.append(cur); cur = String(word) }
        }
        if !cur.isEmpty { lines.append(cur) }
        return Array(lines.suffix(maxLines))  // keep the most recent lines
    }

    func render(_ a: Stats, _ b: Stats, wall: Double) {
        guard isTTY else { return }
        let (cols, rows) = (cols, rows)
        let bodyLines = max(2, (rows - 6) / 2)
        var out = "\u{1B}[H\u{1B}[2J"
        for s in [a, b] {
            let audioS = Double(s.audioSamples) / 16000.0
            let avg = s.chunks > 0 ? s.stepMsTotal / Double(s.chunks) : 0
            let rtf = audioS > 0 ? (s.stepMsTotal / 1000) / audioS : 0
            let lag = max(0, wall - audioS)
            let ttft = s.firstTextDelay >= 0 ? String(format: "%.2fs", s.firstTextDelay) : "—"
            let words = s.transcript.split(whereSeparator: \.isWhitespace).count
            let head = String(
                format: " %@  lag %.2fs · words %d · step %.0f/%.0fms · RTF %.2f · TTFT %@",
                s.label, lag, words, s.lastStepMs, avg, rtf, ttft)
            out += "\u{1B}[7m" + head.padding(toLength: min(cols, max(head.count, cols)), withPad: " ", startingAt: 0)
                + "\u{1B}[0m\n"
            for l in wrap(s.transcript.isEmpty ? "…" : s.transcript, width: cols - 1, maxLines: bodyLines) {
                out += l + "\n"
            }
            out += "\n"
        }
        write(out)
    }

    func begin() { guard isTTY else { return }; write("\u{1B}[2J\u{1B}[H") }
    func end() { guard isTTY else { return }; write("\u{1B}[\(rows);1H\n") }
}

private final class CompareRunner: @unchecked Sendable {
    private let nemo: NemotronASRStreamSession
    private let vox: VoxtralRealtimeStreamSession
    private var nemoStats: Stats
    private var voxStats: Stats
    private let feedSamples: Int
    private let queue = DispatchQueue(label: "mic.compare.feed")
    private var pending: [Float] = []
    private var startTime: CFAbsoluteTime = 0
    private var speechWall: CFAbsoluteTime = 0
    private var speech = false

    private let engine = AVAudioEngine()
    private let inFmt: AVAudioFormat
    private let outFmt: AVAudioFormat
    private let converter: AVAudioConverter
    private let hud = HUD()
    private var ticker: DispatchSourceTimer?

    init(nemo: NemotronASRStreamSession, nemoLabel: String,
         vox: VoxtralRealtimeStreamSession, voxLabel: String,
         feedSamples: Int, inputDevice: AudioDeviceID?) {
        self.nemo = nemo
        self.vox = vox
        self.feedSamples = feedSamples
        self.nemoStats = Stats(label: nemoLabel)
        self.voxStats = Stats(label: voxLabel)

        if let dev = inputDevice ?? AudioDevices.defaultInput() {
            try? AudioDevices.setInput(dev, on: engine)
            let f = engine.inputNode.outputFormat(forBus: 0)
            FileHandle.standardError.write(Data(
                "INPUT: \(AudioDevices.name(of: dev)) | \(Int(f.sampleRate))Hz \(f.channelCount)ch\n".utf8))
        }
        self.inFmt = engine.inputNode.outputFormat(forBus: 0)
        guard let out = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16000, channels: 1, interleaved: false),
              let conv = AVAudioConverter(from: inFmt, to: out) else {
            fatalError("converter setup failed for \(inFmt)")
        }
        self.outFmt = out
        self.converter = conv
    }

    func start() throws {
        startTime = CFAbsoluteTimeGetCurrent()
        hud.begin()
        engine.inputNode.installTap(onBus: 0, bufferSize: 4096, format: inFmt) { [self] buf, _ in
            let f = convert(buf)
            if !f.isEmpty { feed(f) }
        }
        engine.prepare()
        try engine.start()
        let t = DispatchSource.makeTimerSource(queue: queue)
        t.schedule(deadline: .now() + 0.2, repeating: 0.2)
        t.setEventHandler { [self] in hud.render(nemoStats, voxStats, wall: CFAbsoluteTimeGetCurrent() - startTime) }
        t.resume()
        ticker = t
    }

    func stop() {
        ticker?.cancel()
        engine.stop()
        engine.inputNode.removeTap(onBus: 0)
        queue.sync {
            if !pending.isEmpty { stepBoth(pending); pending = [] }
            _ = nemo.finish(); nemoStats.transcript = nemo.text
            _ = vox.finish(); voxStats.transcript = vox.text
            hud.render(nemoStats, voxStats, wall: CFAbsoluteTimeGetCurrent() - startTime)
        }
        hud.end()
        printSummary()
    }

    private func printSummary() {
        func metrics(_ s: Stats) -> (words: Int, avg: Double, rtf: Double, ttft: String, lag: Double) {
            let audioS = Double(s.audioSamples) / 16000.0
            let avg = s.chunks > 0 ? s.stepMsTotal / Double(s.chunks) : 0
            let rtf = audioS > 0 ? (s.stepMsTotal / 1000) / audioS : 0
            let wall = CFAbsoluteTimeGetCurrent() - startTime
            let ttft = s.firstTextDelay >= 0 ? String(format: "%.2fs", s.firstTextDelay) : "—"
            return (s.transcript.split(whereSeparator: \.isWhitespace).count, avg, rtf, ttft, max(0, wall - audioS))
        }
        let n = metrics(nemoStats), v = metrics(voxStats)
        let audioS = Double(nemoStats.audioSamples) / 16000.0
        func row(_ k: String, _ a: String, _ b: String) -> String {
            k.padding(toLength: 12, withPad: " ", startingAt: 0)
                + a.padding(toLength: 30, withPad: " ", startingAt: 0) + b
        }
        var out = "\n══════════════ COMPARISON (audio \(String(format: "%.1f", audioS))s) ══════════════\n"
        out += row("", nemoStats.label, voxStats.label) + "\n"
        out += row("words", "\(n.words)", "\(v.words)") + "\n"
        out += row("avg step", String(format: "%.0fms", n.avg), String(format: "%.0fms", v.avg)) + "\n"
        out += row("RTF", String(format: "%.2f", n.rtf), String(format: "%.2f", v.rtf)) + "\n"
        out += row("TTFT", n.ttft, v.ttft) + "\n"
        out += row("lag", String(format: "%.2fs", n.lag), String(format: "%.2fs", v.lag)) + "\n"
        out += "───────────────────────────────────────────────────────\n"
        out += "NEMOTRON: \(nemoStats.transcript)\n\nVOXTRAL:  \(voxStats.transcript)\n"
        FileHandle.standardError.write(Data(out.utf8))
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
        return Array(UnsafeBufferPointer(start: ch[0], count: Int(out.frameLength)))
    }

    private func feed(_ floats: [Float]) {
        queue.async { [self] in
            pending.append(contentsOf: floats)
            while pending.count >= feedSamples {
                let chunk = Array(pending.prefix(feedSamples))
                pending.removeFirst(feedSamples)
                if !speech {
                    var s: Float = 0; for v in chunk { s += v * v }
                    if (s / Float(chunk.count)).squareRoot() > 0.01 { speech = true; speechWall = CFAbsoluteTimeGetCurrent() }
                }
                stepBoth(chunk)
            }
        }
    }

    private func stepBoth(_ chunk: [Float]) {
        let origin = speech ? speechWall : startTime
        var t = CFAbsoluteTimeGetCurrent()
        let nd = nemo.step(chunk)
        nemoStats.lastStepMs = (CFAbsoluteTimeGetCurrent() - t) * 1000
        nemoStats.stepMsTotal += nemoStats.lastStepMs
        nemoStats.chunks += 1
        nemoStats.audioSamples += chunk.count
        nemoStats.transcript = nemo.text
        if nemoStats.firstTextDelay < 0, !nd.text.trimmingCharacters(in: .whitespaces).isEmpty {
            nemoStats.firstTextDelay = CFAbsoluteTimeGetCurrent() - origin
        }

        t = CFAbsoluteTimeGetCurrent()
        let vd = vox.step(chunk)
        voxStats.lastStepMs = (CFAbsoluteTimeGetCurrent() - t) * 1000
        voxStats.stepMsTotal += voxStats.lastStepMs
        voxStats.chunks += 1
        voxStats.audioSamples += chunk.count
        voxStats.transcript = vox.text
        if voxStats.firstTextDelay < 0, !vd.text.trimmingCharacters(in: .whitespaces).isEmpty {
            voxStats.firstTextDelay = CFAbsoluteTimeGetCurrent() - origin
        }
    }
}

@main
struct MicCompare {
    static func main() async throws {
        GPU.set(memoryLimit: 18 * 1024 * 1024 * 1024, relaxed: false)

        var nemoRepo = "mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit"
        var voxRepo = "mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit"
        var language: String? = "ru"
        var chunkMs: Int? = nil
        var seconds: Double? = nil
        var feedMs = 480
        var inputQuery: String? = nil

        var it = CommandLine.arguments.dropFirst().makeIterator()
        while let a = it.next() {
            switch a {
            case "--nemo": nemoRepo = it.next() ?? nemoRepo
            case "--vox": voxRepo = it.next() ?? voxRepo
            case "--language": language = it.next()
            case "--chunk-ms": chunkMs = Int(it.next() ?? "")
            case "--seconds": seconds = Double(it.next() ?? "")
            case "--feed-ms": feedMs = Int(it.next() ?? "") ?? feedMs
            case "--input": inputQuery = it.next()
            case "--list-devices":
                let def = AudioDevices.defaultInput()
                for d in AudioDevices.inputs() { print("\(d.name)\(d.id == def ? " (default)" : "")\n    uid: \(d.uid)") }
                return
            default: fatalError("unknown arg \(a)")
            }
        }

        var inputDevice: AudioDeviceID? = nil
        if let inputQuery {
            guard let found = AudioDevices.find(inputQuery) else {
                FileHandle.standardError.write(Data("no input matching '\(inputQuery)'\n".utf8)); exit(1)
            }
            inputDevice = found.id
        }

        FileHandle.standardError.write(Data("loading nemotron + voxtral...\n".utf8))
        let nemoModel = try await NemotronASRModel.fromPretrained(nemoRepo)
        let voxModel = try await VoxtralRealtimeModel.fromPretrained(voxRepo)

        // Warm both so first live chunks aren't cold.
        let wn = nemoModel.makeStreamSession(language: language, chunkMs: chunkMs)
        _ = wn.step([Float](repeating: 0, count: 16000 * 2)); _ = wn.finish()
        let wv = voxModel.makeStreamSession()
        _ = wv.step([Float](repeating: 0, count: 16000 * 2)); _ = wv.finish()

        func quant(_ r: String) -> String {
            for q in ["8bit", "4bit", "6bit", "bf16", "fp16"] where r.lowercased().contains(q) { return q }
            return "?"
        }
        let runner = CompareRunner(
            nemo: nemoModel.makeStreamSession(language: language, chunkMs: chunkMs),
            nemoLabel: "NEMOTRON 0.6b \(quant(nemoRepo)) (\(chunkMs.map { "\($0)ms" } ?? "native"))",
            vox: voxModel.makeStreamSession(),
            voxLabel: "VOXTRAL 4B \(quant(voxRepo)) (480ms)",
            feedSamples: max(1, 16000 * feedMs / 1000),
            inputDevice: inputDevice
        )

        try runner.start()
        FileHandle.standardError.write(Data(
            (seconds == nil ? "READY: speak now (Enter to stop)\n" : "READY: speak now (\(Int(seconds!)) s)\n").utf8))
        if let seconds { try await Task.sleep(nanoseconds: UInt64(seconds * 1_000_000_000)) } else { _ = readLine() }
        runner.stop()
    }
}

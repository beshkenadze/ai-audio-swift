@preconcurrency import AVFoundation
import Darwin
import Foundation
import MLX
import MLXAudioCore
import MLXAudioSTT
import MLXAudioVAD

// Side-by-side live ASR: one mic feeds N providers (local Nemotron/Voxtral +
// optional cloud DeepGram/Gemini) and their transcripts + metrics are stacked,
// redrawn in place. See which engine reads your speech best, live.

private final class Flag: @unchecked Sendable { var done = false }

/// Full-screen N-block HUD, redrawn in place (no-op when stdout isn't a TTY).
private struct HUD {
    let isTTY = isatty(STDOUT_FILENO) != 0
    private func size() -> (Int, Int) {
        var w = winsize()
        if ioctl(STDOUT_FILENO, TIOCGWINSZ, &w) == 0, w.ws_col > 0 { return (Int(w.ws_col), Int(w.ws_row)) }
        return (100, 30)
    }
    private func write(_ s: String) { FileHandle.standardOutput.write(Data(s.utf8)) }
    func begin() { guard isTTY else { return }; write("\u{1B}[2J\u{1B}[H") }
    func end() { guard isTTY else { return }; write("\u{1B}[\(size().1);1H\n") }

    private func wrap(_ text: String, width: Int, maxLines: Int) -> [String] {
        var lines: [String] = []; var cur = ""
        for word in text.split(separator: " ", omittingEmptySubsequences: false) {
            if cur.isEmpty { cur = String(word) }
            else if cur.count + 1 + word.count <= width { cur += " " + word }
            else { lines.append(cur); cur = String(word) }
        }
        if !cur.isEmpty { lines.append(cur) }
        return Array(lines.suffix(maxLines))
    }

    func render(_ snaps: [Snap], vad: (on: Bool, active: Bool, prob: Float)?) {
        guard isTTY else { return }
        let (cols, rows) = size()
        var out = "\u{1B}[H\u{1B}[2J"
        var budget = rows - 1
        if let vad, vad.on {
            out += String(format: " VAD %@  p=%.2f\n", vad.active ? "● SPEECH" : "○ silence", vad.prob)
            budget -= 1
        }
        let body = max(1, (budget - snaps.count) / max(1, snaps.count))
        for s in snaps {
            let ttft = s.ttft >= 0 ? String(format: "%.2fs", s.ttft) : "—"
            let lag = s.lag >= 0 ? String(format: "%.2fs", s.lag) : "—"
            let words = s.text.split(whereSeparator: \.isWhitespace).count
            var head = " \(s.label)  ttft \(ttft) · lag \(lag) · words \(words)"
            if !s.perf.isEmpty { head += " · \(s.perf)" }
            if !s.note.isEmpty { head += " · \(s.note)" }
            out += "\u{1B}[7m" + head.padding(toLength: min(cols, max(head.count, cols)), withPad: " ", startingAt: 0) + "\u{1B}[0m\n"
            for l in wrap(s.text.isEmpty ? "…" : s.text, width: cols - 1, maxLines: body) { out += l + "\n" }
        }
        write(out)
    }
}

private final class CompareRunner: @unchecked Sendable {
    private let providers: [LiveASR]
    private let feedSamples: Int
    private let queue = DispatchQueue(label: "mic.compare.feed")
    private var pending: [Float] = []
    private var startTime: CFAbsoluteTime = 0
    private var totalSamples = 0

    private let vad: SileroVAD?
    private var vadState: SileroVADStreamingState?
    private var vadBuffer: [Float] = []
    private var speechActive = false
    private var silenceWindows = 0
    private var lastVadProb: Float = 0
    private let hangoverWindows = 8

    private let engine = AVAudioEngine()
    private let inFmt: AVAudioFormat
    private let outFmt: AVAudioFormat
    private let converter: AVAudioConverter
    private let hud = HUD()
    private var ticker: DispatchSourceTimer?

    init(providers: [LiveASR], vad: SileroVAD?, feedSamples: Int, inputDevice: AudioDeviceID?) {
        self.providers = providers
        self.vad = vad
        self.feedSamples = feedSamples
        if let dev = inputDevice ?? AudioDevices.defaultInput() {
            try? AudioDevices.setInput(dev, on: engine)
            let f = engine.inputNode.outputFormat(forBus: 0)
            FileHandle.standardError.write(Data("INPUT: \(AudioDevices.name(of: dev)) | \(Int(f.sampleRate))Hz \(f.channelCount)ch\n".utf8))
        }
        self.inFmt = engine.inputNode.outputFormat(forBus: 0)
        guard let out = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16000, channels: 1, interleaved: false),
              let conv = AVAudioConverter(from: inFmt, to: out) else { fatalError("converter setup failed for \(inFmt)") }
        self.outFmt = out
        self.converter = conv
    }

    func start(mic: Bool = true) throws {
        startTime = CFAbsoluteTimeGetCurrent()
        hud.begin()
        if mic {
            engine.inputNode.installTap(onBus: 0, bufferSize: 4096, format: inFmt) { [self] buf, _ in
                let f = convert(buf); if !f.isEmpty { feed(f) }
            }
            engine.prepare(); try engine.start()
        }
        let t = DispatchSource.makeTimerSource(queue: queue)
        t.schedule(deadline: .now() + 0.2, repeating: 0.2)
        t.setEventHandler { [self] in
            hud.render(providers.map { $0.snapshot() }, vad: (on: vad != nil, active: speechActive, prob: lastVadProb))
        }
        t.resume(); ticker = t
    }

    /// Drive from a 16 kHz mono file at real-time pace (no mic) — for verifying
    /// providers (incl. cloud) on a known clip without speaking.
    func feedFileRealtime(_ samples: [Float]) {
        let cs = feedSamples
        var i = 0
        while i < samples.count {
            let e = min(i + cs, samples.count)
            feed(Array(samples[i..<e]))
            i = e
            Thread.sleep(forTimeInterval: Double(cs) / 16000.0)  // ~real-time pacing
        }
    }

    func stop() {
        ticker?.cancel()
        engine.stop(); engine.inputNode.removeTap(onBus: 0)
        queue.sync {
            if !pending.isEmpty { emitFrame(pending); pending = [] }
            for p in providers { p.finish() }
        }
        // give cloud sockets a moment to flush final transcripts
        Thread.sleep(forTimeInterval: 0.6)
        hud.render(providers.map { $0.snapshot() }, vad: (on: vad != nil, active: speechActive, prob: lastVadProb))
        hud.end()
        printSummary()
    }

    private func convert(_ buffer: AVAudioPCMBuffer) -> [Float] {
        let ratio = outFmt.sampleRate / inFmt.sampleRate
        let cap = AVAudioFrameCount(Double(buffer.frameLength) * ratio) + 16
        guard let out = AVAudioPCMBuffer(pcmFormat: outFmt, frameCapacity: cap) else { return [] }
        let flag = Flag(); var err: NSError?
        converter.convert(to: out, error: &err) { _, status in
            if flag.done { status.pointee = .noDataNow; return nil }
            flag.done = true; status.pointee = .haveData; return buffer
        }
        guard err == nil, out.frameLength > 0, let ch = out.floatChannelData else { return [] }
        return Array(UnsafeBufferPointer(start: ch[0], count: Int(out.frameLength)))
    }

    private func feed(_ floats: [Float]) {
        queue.async { [self] in
            if let vad {
                vadBuffer.append(contentsOf: floats)
                while vadBuffer.count >= 512 {
                    let w = Array(vadBuffer.prefix(512)); vadBuffer.removeFirst(512)
                    if let (prob, st) = try? vad.feed(chunk: MLXArray(w), state: vadState) {
                        vadState = st; lastVadProb = prob.asArray(Float.self).max() ?? 0
                        if lastVadProb > 0.5 { speechActive = true; silenceWindows = 0 }
                        else { silenceWindows += 1; if silenceWindows > hangoverWindows { speechActive = false } }
                    }
                }
            }
            pending.append(contentsOf: floats)
            while pending.count >= feedSamples {
                let chunk = Array(pending.prefix(feedSamples)); pending.removeFirst(feedSamples)
                emitFrame(chunk)
            }
        }
    }

    private func emitFrame(_ chunk: [Float]) {
        totalSamples += chunk.count
        let frame = AudioFrame(
            samples: chunk, pcm16le: Self.pcm16le(chunk), speechActive: speechActive,
            wallNow: CFAbsoluteTimeGetCurrent() - startTime, audioEndS: Double(totalSamples) / 16000.0)
        for p in providers { p.feed(frame) }  // local steps sync (serialized here); cloud sends non-blocking
    }

    private static func pcm16le(_ samples: [Float]) -> Data {
        var d = Data(capacity: samples.count * 2)
        for s in samples {
            let v = Int16(max(-32767, min(32767, (s * 32767).rounded())))
            withUnsafeBytes(of: v.littleEndian) { d.append(contentsOf: $0) }
        }
        return d
    }

    private func printSummary() {
        let snaps = providers.map { $0.snapshot() }
        var out = "\n══════════════ COMPARISON ══════════════\n"
        for s in snaps {
            let ttft = s.ttft >= 0 ? String(format: "%.2fs", s.ttft) : "—"
            let lag = s.lag >= 0 ? String(format: "%.2fs", s.lag) : "—"
            out += "\n=== \(s.label) ===  ttft \(ttft) · lag \(lag)\(s.perf.isEmpty ? "" : " · \(s.perf)")\n\(s.text)\n"
        }
        FileHandle.standardError.write(Data(out.utf8))
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
        var useVad = false
        var vadRepo = "mlx-community/silero-vad"
        var cloud = false
        var geminiModel = "gemini-2.5-flash-native-audio-latest"
        var dgModel = "nova-2"
        var wavPath: String? = nil

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
            case "--vad": useVad = true
            case "--vad-repo": vadRepo = it.next() ?? vadRepo
            case "--cloud": cloud = true
            case "--gemini-model": geminiModel = it.next() ?? geminiModel
            case "--dg-model": dgModel = it.next() ?? dgModel
            case "--wav": wavPath = it.next()
            case "--list-devices":
                let def = AudioDevices.defaultInput()
                for d in AudioDevices.inputs() { print("\(d.name)\(d.id == def ? " (default)" : "")\n    uid: \(d.uid)") }
                return
            default: fatalError("unknown arg \(a)")
            }
        }

        var inputDevice: AudioDeviceID? = nil
        if let inputQuery {
            guard let f = AudioDevices.find(inputQuery) else {
                FileHandle.standardError.write(Data("no input matching '\(inputQuery)'\n".utf8)); exit(1)
            }
            inputDevice = f.id
        }

        func quant(_ r: String) -> String {
            for q in ["8bit", "4bit", "6bit", "bf16", "fp16"] where r.lowercased().contains(q) { return q }
            return "?"
        }

        FileHandle.standardError.write(Data("loading models\(useVad ? " + silero-vad" : "")\(cloud ? " + cloud" : "")...\n".utf8))
        let nemoModel = try await NemotronASRModel.fromPretrained(nemoRepo)
        let voxModel = try await VoxtralRealtimeModel.fromPretrained(voxRepo)
        let vad = useVad ? try await SileroVAD.fromPretrained(vadRepo) : nil

        // Warm Metal kernels so the first live chunks aren't cold.
        let wn = nemoModel.makeStreamSession(language: language, chunkMs: chunkMs); _ = wn.step([Float](repeating: 0, count: 16000 * 2)); _ = wn.finish()
        let wv = voxModel.makeStreamSession(); _ = wv.step([Float](repeating: 0, count: 16000 * 2)); _ = wv.finish()

        let nemoSession = nemoModel.makeStreamSession(language: language, chunkMs: chunkMs)
        let voxSession = voxModel.makeStreamSession()

        var providers: [LiveASR] = [
            LocalASR(label: "NEMOTRON 0.6b \(quant(nemoRepo)) (\(chunkMs.map { "\($0)ms" } ?? "native"))",
                     gated: false,
                     step: { nemoSession.step($0); return nemoSession.text },
                     finish: { _ = nemoSession.finish(); return nemoSession.text }),
            LocalASR(label: "VOXTRAL 4B \(quant(voxRepo)) (480ms)",
                     gated: vad != nil,
                     step: { voxSession.step($0); return voxSession.text },
                     finish: { _ = voxSession.finish(); return voxSession.text }),
        ]

        if cloud {
            if let key = Env.value("DEEPGRAM_API_KEY") {
                providers.append(DeepgramASR(key: key, language: language ?? "ru", model: dgModel))
            } else {
                FileHandle.standardError.write(Data("(no DEEPGRAM_API_KEY in .env — skipping DeepGram)\n".utf8))
            }
            if let key = Env.value("GEMINI_API_KEY") {
                providers.append(GeminiASR(key: key, model: geminiModel))
            } else {
                FileHandle.standardError.write(Data("(no GEMINI_API_KEY in .env — skipping Gemini)\n".utf8))
            }
        }

        let runner = CompareRunner(
            providers: providers, vad: vad,
            feedSamples: max(1, 16000 * feedMs / 1000), inputDevice: inputDevice)

        if let wavPath {
            let (sr, raw) = try loadAudioArray(from: URL(fileURLWithPath: wavPath), sampleRate: 16000)
            precondition(sr == 16000, "expected 16k, got \(sr)")
            let samples = (raw.ndim > 1 ? raw.mean(axis: -1) : raw).asType(.float32).asArray(Float.self)
            try runner.start(mic: false)
            FileHandle.standardError.write(Data("feeding \(wavPath) (\(String(format: "%.1f", Double(samples.count) / 16000))s) at real-time pace...\n".utf8))
            runner.feedFileRealtime(samples)
            runner.stop()
            return
        }

        try runner.start()
        FileHandle.standardError.write(Data((seconds == nil ? "READY: speak now (Enter to stop)\n" : "READY: speak now (\(Int(seconds!)) s)\n").utf8))
        if let seconds { try await Task.sleep(nanoseconds: UInt64(seconds * 1_000_000_000)) } else { _ = readLine() }
        runner.stop()
    }
}

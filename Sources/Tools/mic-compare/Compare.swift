@preconcurrency import AVFoundation
import Darwin
import Foundation
import MLX
import MLXAudioCore
import MLXAudioSTT
import MLXAudioVAD
import AudioEnhanceKitCore
import AudioEnhanceKitDFN
import AudioEnhanceKitDPDFNet

/// Common surface over AudioEnhanceKit's two ANE streaming denoisers so the tap
/// can use either DeepFilterNet or DPDFNet interchangeably.
protocol LiveDenoiser: AnyObject {
    func enqueueInput(_ samples: UnsafePointer<Float>, count: Int) -> Bool
    func dequeueOutput(into dst: UnsafeMutablePointer<Float>, maxCount: Int) -> Int
    func requestFlush()
    func stop()
}
extension AsyncStreamingDenoiser: LiveDenoiser {}
extension AsyncDPDFNetEnhancer: LiveDenoiser {}

// Side-by-side live ASR: one mic feeds N providers (local Nemotron/Voxtral +
// optional cloud DeepGram/Gemini) and their transcripts + metrics are stacked,
// redrawn in place. See which engine reads your speech best, live.

private final class Flag: @unchecked Sendable { var done = false }

/// Write mono Int16 WAV.
private func writeWavMono(_ samples: [Float], sampleRate: Int, to url: URL) {
    let sr = UInt32(sampleRate); let ch: UInt16 = 1; let bits: UInt16 = 16
    let dataSize = UInt32(samples.count * 2)
    var d = Data()
    func u32(_ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { d.append(contentsOf: $0) } }
    func u16(_ v: UInt16) { withUnsafeBytes(of: v.littleEndian) { d.append(contentsOf: $0) } }
    d.append("RIFF".data(using: .ascii)!); u32(36 + dataSize); d.append("WAVE".data(using: .ascii)!)
    d.append("fmt ".data(using: .ascii)!); u32(16); u16(1); u16(ch); u32(sr)
    u32(sr * UInt32(ch) * UInt32(bits) / 8); u16(ch * bits / 8); u16(bits)
    d.append("data".data(using: .ascii)!); u32(dataSize)
    for s in samples {
        let v = Int16(max(-32767, min(32767, (s * 32767).rounded())))
        withUnsafeBytes(of: v.littleEndian) { d.append(contentsOf: $0) }
    }
    try? d.write(to: url)
}

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

    /// Pad/truncate to an exact display width (plain text → char count == columns).
    private func cell(_ s: String, _ w: Int) -> String {
        let t = s.count > w ? String(s.prefix(w)) : s
        return t + String(repeating: " ", count: max(0, w - t.count))
    }

    /// Side-by-side columns (one per engine), each a fixed width so growth in one
    /// column never reflows the others — no more stacked rows jittering.
    func render(_ snaps: [Snap], vad: (on: Bool, active: Bool, prob: Float)?) {
        guard isTTY else { return }
        let (cols, rows) = size()
        let eol = "\u{1B}[K\n"   // erase-to-EOL: redraw in place without a flickery full clear
        var out = "\u{1B}[H"
        var top = rows - 1
        if let vad, vad.on {
            out += String(format: " VAD %@  p=%.2f", vad.active ? "● SPEECH" : "○ silence", vad.prob) + eol
            top -= 1
        }
        let n = max(1, snaps.count)
        let sep = " │ "
        let colW = max(12, (cols - (n - 1) * sep.count) / n)
        let bodyH = max(1, top - 2)   // 2 header rows (label + metrics)

        var labelRow: [String] = [], metaRow: [String] = [], bodies: [[String]] = []
        for s in snaps {
            labelRow.append(cell(s.label, colW))
            let ttft = s.ttft >= 0 ? String(format: "%.2fs", s.ttft) : "—"
            let lag = s.lag >= 0 ? String(format: "%.2fs", s.lag) : "—"
            let words = s.text.split(whereSeparator: \.isWhitespace).count
            var meta = "ttft \(ttft)·lag \(lag)·\(words)w"
            if !s.perf.isEmpty { meta += "·\(s.perf)" }
            if !s.note.isEmpty { meta += "·\(s.note)" }
            metaRow.append(cell(meta, colW))
            var lines = wrap(s.text.isEmpty ? "…" : s.text, width: colW, maxLines: bodyH)
            while lines.count < bodyH { lines.append("") }
            bodies.append(lines.map { cell($0, colW) })
        }
        out += "\u{1B}[7m" + labelRow.joined(separator: sep) + "\u{1B}[0m" + eol   // inverted header
        out += "\u{1B}[2m" + metaRow.joined(separator: sep) + "\u{1B}[0m" + eol    // dim metrics
        for r in 0..<bodyH { out += bodies.map { $0[r] }.joined(separator: sep) + eol }
        out += "\u{1B}[J"   // clear anything below the grid (e.g. after a resize)
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
    private let hud = HUD()
    private var ticker: DispatchSourceTimer?

    private let recordURL: URL?
    private var recordBuffer: [Float] = []

    // Single audio path (mic OR file): source -> 48k mono -> optional ANE denoise
    // (DeepFilterNet/DPDFNet) -> 16k -> ASR. GPU stays free for the models.
    private let denoiser: (any LiveDenoiser)?
    private let to48: AVAudioConverter    // mic input format -> 48k mono
    private let down16: AVAudioConverter  // 48k mono -> 16k mono (the one downsample)
    private var dnOut = [Float](repeating: 0, count: 16384)

    init(providers: [LiveASR], vad: SileroVAD?, feedSamples: Int, inputDevice: AudioDeviceID?,
         recordURL: URL? = nil, denoiser: (any LiveDenoiser)? = nil) {
        self.providers = providers
        self.vad = vad
        self.feedSamples = feedSamples
        self.recordURL = recordURL
        self.denoiser = denoiser
        if let dev = inputDevice ?? AudioDevices.defaultInput() {
            try? AudioDevices.setInput(dev, on: engine)
            let f = engine.inputNode.outputFormat(forBus: 0)
            FileHandle.standardError.write(Data("INPUT: \(AudioDevices.name(of: dev)) | \(Int(f.sampleRate))Hz \(f.channelCount)ch\n".utf8))
        }
        self.inFmt = engine.inputNode.outputFormat(forBus: 0)
        guard let out = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16000, channels: 1, interleaved: false),
              let mono48 = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 48000, channels: 1, interleaved: false),
              let c48 = AVAudioConverter(from: inFmt, to: mono48),
              let c16 = AVAudioConverter(from: mono48, to: out) else { fatalError("converter setup failed for \(inFmt)") }
        self.outFmt = out
        self.to48 = c48
        self.down16 = c16
    }

    func start(mic: Bool = true) throws {
        startTime = CFAbsoluteTimeGetCurrent()
        hud.begin()
        if mic {
            engine.inputNode.installTap(onBus: 0, bufferSize: 4096, format: inFmt) { [self] buf, _ in
                let mono48 = resampleMono(buf, via: to48)  // mic format -> 48k mono
                if !mono48.isEmpty { feedSource48(mono48) }
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

    /// Emulate the mic from a 48 kHz mono file at real-time pace — same single path
    /// as the live tap (denoise included), so a recorded clip is deterministic.
    func feedFile48Realtime(_ mono48: [Float]) {
        let cs = 4800  // 0.1s @48k
        var i = 0
        while i < mono48.count {
            let e = min(i + cs, mono48.count)
            feedSource48(Array(mono48[i..<e]))
            i = e
            Thread.sleep(forTimeInterval: Double(cs) / 48000.0)  // ~real-time pacing
        }
        if let denoiser {  // flush the denoiser tail
            denoiser.requestFlush()
            for _ in 0..<20 {
                let n = dnOut.withUnsafeMutableBufferPointer { denoiser.dequeueOutput(into: $0.baseAddress!, maxCount: $0.count) }
                if n > 0 { let s16 = resampleMono(Array(dnOut[0..<n]), via: down16); if !s16.isEmpty { feed(s16) } }
                Thread.sleep(forTimeInterval: 0.03)
            }
        }
    }

    func stop() {
        ticker?.cancel()
        engine.stop(); engine.inputNode.removeTap(onBus: 0)
        denoiser?.stop()
        queue.sync {
            if !pending.isEmpty { emitFrame(pending); pending = [] }
            for p in providers { p.finish() }
        }
        // give cloud sockets a moment to flush final transcripts
        Thread.sleep(forTimeInterval: 0.6)
        hud.render(providers.map { $0.snapshot() }, vad: (on: vad != nil, active: speechActive, prob: lastVadProb))
        hud.end()
        if let recordURL {
            writeWavMono(recordBuffer, sampleRate: 16000, to: recordURL)
            FileHandle.standardError.write(Data("recorded \(String(format: "%.1f", Double(recordBuffer.count) / 16000))s -> \(recordURL.path) (replay with --wav)\n".utf8))
        }
        printSummary()
    }

    /// THE single audio path: 48k mono in -> optional ANE denoise -> 16k -> ASR.
    /// Both the mic tap and the file feeder call this, so behaviour is identical.
    private func feedSource48(_ mono48: [Float]) {
        var s48 = mono48
        if let denoiser {
            mono48.withUnsafeBufferPointer { _ = denoiser.enqueueInput($0.baseAddress!, count: $0.count) }
            let n = dnOut.withUnsafeMutableBufferPointer { denoiser.dequeueOutput(into: $0.baseAddress!, maxCount: $0.count) }
            s48 = n > 0 ? Array(dnOut[0..<n]) : []
        }
        if !s48.isEmpty {
            let s16 = resampleMono(s48, via: down16)
            if !s16.isEmpty { feed(s16) }
        }
    }

    /// Convert a PCM buffer through a stateful converter -> [Float] (its output fmt).
    private func resampleMono(_ buffer: AVAudioPCMBuffer, via conv: AVAudioConverter) -> [Float] {
        let outF = conv.outputFormat
        let cap = AVAudioFrameCount(Double(buffer.frameLength) * outF.sampleRate / conv.inputFormat.sampleRate) + 64
        guard let out = AVAudioPCMBuffer(pcmFormat: outF, frameCapacity: cap) else { return [] }
        let flag = Flag(); var err: NSError?
        conv.convert(to: out, error: &err) { _, status in
            if flag.done { status.pointee = .noDataNow; return nil }
            flag.done = true; status.pointee = .haveData; return buffer
        }
        guard err == nil, out.frameLength > 0, let ch = out.floatChannelData else { return [] }
        return Array(UnsafeBufferPointer(start: ch[0], count: Int(out.frameLength)))
    }

    /// Same, for a mono [Float] (wraps it in a buffer at the converter's input rate).
    private func resampleMono(_ samples: [Float], via conv: AVAudioConverter) -> [Float] {
        guard !samples.isEmpty,
              let inBuf = AVAudioPCMBuffer(pcmFormat: conv.inputFormat, frameCapacity: AVAudioFrameCount(samples.count)),
              let dst = inBuf.floatChannelData else { return [] }
        samples.withUnsafeBufferPointer { dst[0].update(from: $0.baseAddress!, count: samples.count) }
        inBuf.frameLength = AVAudioFrameCount(samples.count)
        return resampleMono(inBuf, via: conv)
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
        if recordURL != nil { recordBuffer.append(contentsOf: chunk) }
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
        let items = providers.map { ($0, $0.snapshot()) }

        // Piped/scripted (stderr not a TTY): plain, parseable — keep `=== label ===`.
        guard isatty(STDERR_FILENO) != 0 else {
            var out = "\n══════════════ COMPARISON ══════════════\n"
            for (_, s) in items {
                let ttft = s.ttft >= 0 ? String(format: "%.2fs", s.ttft) : "—"
                let lag = s.lag >= 0 ? String(format: "%.2fs", s.lag) : "—"
                out += "\n=== \(s.label) ===  ttft \(ttft) · lag \(lag)\(s.perf.isEmpty ? "" : " · \(s.perf)")\n\(s.text)\n"
            }
            FileHandle.standardError.write(Data(out.utf8))
            return
        }

        // TTY: boxed, colored panels (cyan = on-device, yellow = cloud).
        var w = winsize()
        let cols = (ioctl(STDOUT_FILENO, TIOCGWINSZ, &w) == 0 && w.ws_col > 0) ? min(Int(w.ws_col), 110) : 100
        let textW = max(24, cols - 6)
        func wrapP(_ t: String) -> [String] {
            var ls: [String] = []; var c = ""
            for word in t.split(separator: " ") {
                if c.isEmpty { c = String(word) }
                else if c.count + 1 + word.count <= textW { c += " " + word }
                else { ls.append(c); c = String(word) }
            }
            if !c.isEmpty { ls.append(c) }
            return ls
        }
        var out = "\u{1B}[2J\u{1B}[H\u{1B}[1;36m╭─ COMPARISON " + String(repeating: "─", count: max(0, cols - 15)) + "╮\u{1B}[0m\n"
        for (p, s) in items {
            let accent = p.isLocal ? "\u{1B}[1;36m" : "\u{1B}[1;33m"   // cyan local · yellow cloud
            let badge = p.isLocal ? "on-device" : "cloud"
            let ttft = s.ttft >= 0 ? String(format: "%.2fs", s.ttft) : "—"
            let lag = s.lag >= 0 ? String(format: "%.2fs", s.lag) : "—"
            var meta = "ttft \(ttft) · lag \(lag)"
            if !s.perf.isEmpty { meta += " · \(s.perf)" }
            if !s.note.isEmpty { meta += " · \(s.note)" }
            out += "\n \(accent)▸ \(s.label)\u{1B}[0m \u{1B}[2m[\(badge)]\u{1B}[0m\n"
            out += "   \u{1B}[2m\(meta)\u{1B}[0m\n"
            out += "   \u{1B}[2m┌" + String(repeating: "─", count: textW + 1) + "\u{1B}[0m\n"
            for l in wrapP(s.text.isEmpty ? "…" : s.text) { out += "   \u{1B}[2m│\u{1B}[0m \(l)\n" }
            out += "   \u{1B}[2m└" + String(repeating: "─", count: textW + 1) + "\u{1B}[0m\n"
        }
        out += "\u{1B}[1;36m╰" + String(repeating: "─", count: cols - 2) + "╯\u{1B}[0m\n"
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
        var recordPath: String? = nil
        var onlyArg: String? = nil
        var denoise = false
        var dfnModel = "standard"
        var denoiserKind = "dfn"   // dfn | dpdfnet
        var voxDelay: Int? = nil   // Voxtral transcription delay ms (160/240/480/960/2400); nil = default 480

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
            case "--record": recordPath = it.next()
            case "--only": onlyArg = it.next()  // comma list: nemotron,voxtral,deepgram,gemini
            case "--denoise": denoise = true    // ANE denoise (mic + --wav, same path)
            case "--dfn-model": dfnModel = it.next() ?? dfnModel  // standard|enhanced|v2b|v2b_n4|v2b_n4_4bit
            case "--denoiser": denoiserKind = (it.next() ?? denoiserKind).lowercased()  // dfn | dpdfnet
            case "--vox-delay": voxDelay = Int(it.next() ?? "")  // Voxtral delay ms (160/240/480/960/2400)
            case "--list-devices":
                let def = AudioDevices.defaultInput()
                for d in AudioDevices.inputs() { print("\(d.name)\(d.id == def ? " (default)" : "")\n    uid: \(d.uid)") }
                return
            default: fatalError("unknown arg \(a)")
            }
        }

        // Fail fast on a bad --wav path before paying for model + denoiser load.
        if let wavPath, !FileManager.default.fileExists(atPath: wavPath) {
            FileHandle.standardError.write(Data("--wav file not found: \(wavPath)\n".utf8)); exit(1)
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

        // Which engines to run (default all). Lets you isolate, e.g. one local vs
        // cloud, so the GPU-bound local models don't contend with each other.
        let want: Set<String>? = onlyArg.map { Set($0.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces).lowercased() }) }
        func wanted(_ n: String) -> Bool { want == nil || want!.contains(n) }

        let vad = useVad ? try await SileroVAD.fromPretrained(vadRepo) : nil
        var providers: [LiveASR] = []

        if wanted("nemotron") {
            FileHandle.standardError.write(Data("loading nemotron...\n".utf8))
            let m = try await NemotronASRModel.fromPretrained(nemoRepo)
            let w = m.makeStreamSession(language: language, chunkMs: chunkMs); _ = w.step([Float](repeating: 0, count: 16000 * 2)); _ = w.finish()
            let s = m.makeStreamSession(language: language, chunkMs: chunkMs)
            providers.append(LocalASR(label: "NEMOTRON 0.6b \(quant(nemoRepo)) (\(chunkMs.map { "\($0)ms" } ?? "native"))",
                                      gated: false, step: { s.step($0); return s.text }, finish: { _ = s.finish(); return s.text }))
        }
        if wanted("nemotron-ane") {
            #if canImport(CoreML)
            FileHandle.standardError.write(Data("loading Nemotron CoreML/ANE encoder (HF download + compile on first run)...\n".utf8))
            let m = try await NemotronASRModel.fromPretrained(nemoRepo)
            do {
                let s = try await NemotronCoreMLStreamSession.create(model: m)
                providers.append(LocalASR(label: "NEMOTRON ANE (CoreML enc)", gated: false,
                                          step: { s.step($0) }, finish: { s.finish() }))
            } catch { FileHandle.standardError.write(Data("(Nemotron ANE unavailable: \(error) — skipping)\n".utf8)) }
            #else
            FileHandle.standardError.write(Data("(CoreML unavailable — skipping nemotron-ane)\n".utf8))
            #endif
        }
        if wanted("voxtral") {
            FileHandle.standardError.write(Data("loading voxtral...\n".utf8))
            let m = try await VoxtralRealtimeModel.fromPretrained(voxRepo)
            let w = m.makeStreamSession(); _ = w.step([Float](repeating: 0, count: 16000 * 2)); _ = w.finish()
            let s = m.makeStreamSession(transcriptionDelayMs: voxDelay)
            providers.append(LocalASR(label: "VOXTRAL 4B \(quant(voxRepo)) (\(voxDelay ?? 480)ms)",
                                      gated: vad != nil, step: { s.step($0); return s.text }, finish: { _ = s.finish(); return s.text }))
        }
        if wanted("voxtral-la") {
            FileHandle.standardError.write(Data("loading voxtral+LA...\n".utf8))
            let m = try await VoxtralRealtimeModel.fromPretrained(voxRepo)
            let s = VoxtralLocalAgreementSession(model: m, tickMs: 600)
            providers.append(LocalASR(
                label: "VOXTRAL+LA \(quant(voxRepo)) (self-agreement)", gated: vad != nil,
                step: { let r = s.step($0); return r.confirmed + (r.volatile.isEmpty ? "" : " ⟨\(r.volatile.trimmingCharacters(in: .whitespaces))⟩") },
                finish: { let r = s.finish(); return r.confirmed + r.volatile }))
        }
        if wanted("two-tier") {
            FileHandle.standardError.write(Data("loading two-tier (Nemotron partial + Voxtral final)...\n".utf8))
            let nm = try await NemotronASRModel.fromPretrained(nemoRepo)
            let vm = try await VoxtralRealtimeModel.fromPretrained(voxRepo)
            let nw = nm.makeStreamSession(language: language, chunkMs: 80); _ = nw.step([Float](repeating: 0, count: 16000)); _ = nw.finish()
            let vw = vm.makeStreamSession(); _ = vw.step([Float](repeating: 0, count: 16000)); _ = vw.finish()
            let s = TwoTierSession(nemotron: nm, voxtral: vm, language: language, fastChunkMs: 80, voxtralDelayMs: voxDelay ?? 960)
            providers.append(LocalASR(
                label: "TWO-TIER Nemotron→Voxtral", gated: vad != nil,
                step: { let r = s.step($0); return r.confirmed + (r.partial.isEmpty ? "" : " ⟨\(r.partial)⟩") },
                finish: { let r = s.finish(); return r.confirmed }))
        }
        if wanted("two-tier-ane") {
            #if canImport(CoreML)
            FileHandle.standardError.write(Data("loading two-tier ANE (Nemotron on ANE + Voxtral on GPU)...\n".utf8))
            let nm = try await NemotronASRModel.fromPretrained(nemoRepo)
            let vm = try await VoxtralRealtimeModel.fromPretrained(voxRepo)
            let vw = vm.makeStreamSession(); _ = vw.step([Float](repeating: 0, count: 16000)); _ = vw.finish()
            do {
                let ane = try await NemotronCoreMLStreamSession.create(model: nm)
                let s = TwoTierSession(fastStep: { _ = ane.step($0) }, fastText: { ane.text },
                                       fastFinish: { _ = ane.finish() }, voxtral: vm, voxtralDelayMs: voxDelay ?? 960)
                providers.append(LocalASR(
                    label: "TWO-TIER ANE Nemotron◇Voxtral", gated: vad != nil,
                    step: { let r = s.step($0); return r.confirmed + (r.partial.isEmpty ? "" : " ⟨\(r.partial)⟩") },
                    finish: { let r = s.finish(); return r.confirmed }))
            } catch { FileHandle.standardError.write(Data("(two-tier ANE unavailable: \(error) — skipping)\n".utf8)) }
            #else
            FileHandle.standardError.write(Data("(CoreML unavailable — skipping two-tier-ane)\n".utf8))
            #endif
        }
        if wanted("apple") {
            if #available(macOS 26.0, iOS 26.0, *) {
                let loc: String = {
                    switch (language ?? "en").lowercased() {
                    case "ru": return "ru_RU"
                    case "en": return "en_US"
                    case let l where l.contains("_"): return l
                    case let l: return "\(l)_\(l.uppercased())"
                    }
                }()
                FileHandle.standardError.write(Data("loading Apple SpeechAnalyzer (\(loc), reference)...\n".utf8))
                do { providers.append(try await SpeechAnalyzerASR.create(locale: loc)) }
                catch { FileHandle.standardError.write(Data("(Apple SpeechAnalyzer unavailable: \(error) — skipping)\n".utf8)) }
            } else {
                FileHandle.standardError.write(Data("(SpeechAnalyzer needs macOS 26+ — skipping apple)\n".utf8))
            }
        }
        if cloud {
            if wanted("deepgram") {
                if let key = Env.value("DEEPGRAM_API_KEY") { providers.append(DeepgramASR(key: key, language: language ?? "ru", model: dgModel)) }
                else { FileHandle.standardError.write(Data("(no DEEPGRAM_API_KEY in .env — skipping DeepGram)\n".utf8)) }
            }
            if wanted("gemini") {
                if let key = Env.value("GEMINI_API_KEY") { providers.append(GeminiASR(key: key, model: geminiModel)) }
                else { FileHandle.standardError.write(Data("(no GEMINI_API_KEY in .env — skipping Gemini)\n".utf8)) }
            }
        }
        guard !providers.isEmpty else { FileHandle.standardError.write(Data("no providers selected (check --only / --cloud)\n".utf8)); exit(1) }

        // --denoise: ANE denoiser streamed at native 48 kHz (GPU stays free).
        // Mic and --wav share this one path — the file just emulates the mic.
        func makeDenoiser() async throws -> any LiveDenoiser {
            if denoiserKind == "dpdfnet" {
                let variant = DPDFNetVariant(rawValue: dfnModel) ?? .dpdfnet2_48khz_hr
                FileHandle.standardError.write(Data("ANE denoise: DPDFNet '\(variant.rawValue)' (48kHz)\n".utf8))
                var cfg = DPDFNetEnhancer.Config(variant: variant); cfg.backend = .ane
                return try await AsyncDPDFNetEnhancer.create(config: cfg)
            }
            let variant = ModelVariant(rawValue: dfnModel) ?? .standard
            FileHandle.standardError.write(Data("ANE denoise: DeepFilterNet '\(variant.rawValue)' (48kHz)\n".utf8))
            return try await AsyncStreamingDenoiser.create(
                config: .init(variant: variant, backend: .ane, compensateDelay: true, enableProfiling: false))
        }

        let liveDenoiser: (any LiveDenoiser)? = denoise ? try await makeDenoiser() : nil

        let runner = CompareRunner(
            providers: providers, vad: vad,
            feedSamples: max(1, 16000 * feedMs / 1000), inputDevice: inputDevice,
            recordURL: recordPath.map { URL(fileURLWithPath: $0) }, denoiser: liveDenoiser)

        if let wavPath {
            // Emulate the mic: load at 48k mono and stream through the SAME single
            // path (denoise included) at real-time pace — deterministic, both
            // denoisers, no separate offline code. (Existence checked up front.)
            let mono48: [Float]
            do {
                let (_, raw) = try loadAudioArray(from: URL(fileURLWithPath: wavPath), sampleRate: 48000)
                mono48 = (raw.ndim > 1 ? raw.mean(axis: -1) : raw).asType(.float32).asArray(Float.self)
            } catch {
                FileHandle.standardError.write(Data("--wav could not decode \(wavPath): \(error)\n".utf8)); exit(1)
            }
            try runner.start(mic: false)
            FileHandle.standardError.write(Data(
                "feeding \(wavPath) (\(String(format: "%.1f", Double(mono48.count) / 48000))s @48k)\(denoise ? " through ANE denoise" : "") at real-time pace...\n".utf8))
            runner.feedFile48Realtime(mono48)
            runner.stop()
            return
        }

        try runner.start()
        FileHandle.standardError.write(Data((seconds == nil ? "READY: speak now (Enter to stop)\n" : "READY: speak now (\(Int(seconds!)) s)\n").utf8))
        if let seconds { try await Task.sleep(nanoseconds: UInt64(seconds * 1_000_000_000)) } else { _ = readLine() }
        runner.stop()
    }
}

@preconcurrency import AVFoundation
import Darwin
import Foundation
import MLX
import MLXAudioCore
import MLXAudioSTS
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

    private let recordURL: URL?
    private var recordBuffer: [Float] = []

    // Live ANE denoise: mic 48k mono -> DeepFilterNet on the Neural Engine ->
    // clean 48k -> downsample to 16k for ASR. GPU stays free for the models.
    private let denoiser: (any LiveDenoiser)?
    private let cleanConv: AVAudioConverter?  // 48k mono -> 16k mono (clean path)
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
              let conv = AVAudioConverter(from: inFmt, to: out) else { fatalError("converter setup failed for \(inFmt)") }
        self.outFmt = out
        self.converter = conv
        if denoiser != nil, let mono48 = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 48000, channels: 1, interleaved: false) {
            self.cleanConv = AVAudioConverter(from: mono48, to: out)
        } else {
            self.cleanConv = nil
        }
    }

    func start(mic: Bool = true) throws {
        startTime = CFAbsoluteTimeGetCurrent()
        hud.begin()
        if mic {
            engine.inputNode.installTap(onBus: 0, bufferSize: 4096, format: inFmt) { [self] buf, _ in
                if let denoiser, let cleanConv {
                    // mic -> 48k mono -> ANE denoise -> 48k -> 16k -> ASR
                    let mono48 = downmix48k(buf)
                    if !mono48.isEmpty {
                        mono48.withUnsafeBufferPointer { _ = denoiser.enqueueInput($0.baseAddress!, count: $0.count) }
                    }
                    let n = dnOut.withUnsafeMutableBufferPointer { denoiser.dequeueOutput(into: $0.baseAddress!, maxCount: $0.count) }
                    if n > 0 {
                        let clean16 = resampleMono(Array(dnOut[0..<n]), via: cleanConv)
                        if !clean16.isEmpty { feed(clean16) }
                    }
                } else {
                    let f = convert(buf); if !f.isEmpty { feed(f) }
                }
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

    /// Downmix a tap buffer to mono [Float] at its native (48k) rate.
    private func downmix48k(_ buffer: AVAudioPCMBuffer) -> [Float] {
        guard let ch = buffer.floatChannelData else { return [] }
        let n = Int(buffer.frameLength)
        let channels = Int(buffer.format.channelCount)
        if channels <= 1 { return Array(UnsafeBufferPointer(start: ch[0], count: n)) }
        var out = [Float](repeating: 0, count: n)
        for i in 0..<n {
            var s: Float = 0
            for c in 0..<channels { s += ch[c][i] }
            out[i] = s / Float(channels)
        }
        return out
    }

    /// Resample mono [Float] through a stateful converter (48k -> 16k).
    private func resampleMono(_ mono48: [Float], via conv: AVAudioConverter) -> [Float] {
        let inFmt = conv.inputFormat
        guard let inBuf = AVAudioPCMBuffer(pcmFormat: inFmt, frameCapacity: AVAudioFrameCount(mono48.count)),
              let dst = inBuf.floatChannelData else { return [] }
        mono48.withUnsafeBufferPointer { dst[0].update(from: $0.baseAddress!, count: mono48.count) }
        inBuf.frameLength = AVAudioFrameCount(mono48.count)
        let cap = AVAudioFrameCount(Double(mono48.count) * outFmt.sampleRate / inFmt.sampleRate) + 16
        guard let out = AVAudioPCMBuffer(pcmFormat: outFmt, frameCapacity: cap) else { return [] }
        let flag = Flag(); var err: NSError?
        conv.convert(to: out, error: &err) { _, status in
            if flag.done { status.pointee = .noDataNow; return nil }
            flag.done = true; status.pointee = .haveData; return inBuf
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
    /// Latest local snapshot dir of the DFN repo in the hf cache, if present.
    static func dfnSnapshotDir() -> String? {
        let base = NSHomeDirectory() + "/.cache/huggingface/hub/models--mlx-community--DeepFilterNet-mlx/snapshots"
        guard let snaps = try? FileManager.default.contentsOfDirectory(atPath: base), !snaps.isEmpty else { return nil }
        return base + "/" + snaps.sorted().last!
    }

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
        var dfnRepo: String? = nil
        var tishPath: String? = nil
        var dfnModel = "standard"
        var denoiserKind = "dfn"   // dfn | dpdfnet

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
            case "--denoise": denoise = true    // DeepFilterNet clean-up (--wav mode)
            case "--dfn": dfnRepo = it.next()   // local MLX DFN model dir (fallback path)
            case "--tish": tishPath = it.next() // AudioEnhanceKit tish-denoise binary (ANE DFN)
            case "--dfn-model": dfnModel = it.next() ?? dfnModel  // standard|enhanced|v2b|v2b_n4|v2b_n4_4bit
            case "--denoiser": denoiserKind = (it.next() ?? denoiserKind).lowercased()  // dfn | dpdfnet
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
        if wanted("voxtral") {
            FileHandle.standardError.write(Data("loading voxtral...\n".utf8))
            let m = try await VoxtralRealtimeModel.fromPretrained(voxRepo)
            let w = m.makeStreamSession(); _ = w.step([Float](repeating: 0, count: 16000 * 2)); _ = w.finish()
            let s = m.makeStreamSession()
            providers.append(LocalASR(label: "VOXTRAL 4B \(quant(voxRepo)) (480ms)",
                                      gated: vad != nil, step: { s.step($0); return s.text }, finish: { _ = s.finish(); return s.text }))
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

        // Live mic + --denoise: DeepFilterNet on the ANE, streamed in the tap at
        // its native 48 kHz (full benefit, GPU free). (--wav + --denoise uses the
        // offline tish-denoise path above.)
        var liveDenoiser: (any LiveDenoiser)? = nil
        if denoise && wavPath == nil {
            if denoiserKind == "dpdfnet" {
                let variant: DPDFNetVariant = DPDFNetVariant(rawValue: dfnModel) ?? .dpdfnet2_48khz_hr
                FileHandle.standardError.write(Data("starting live ANE denoise (DPDFNet '\(variant.rawValue)', 48kHz)...\n".utf8))
                var cfg = DPDFNetEnhancer.Config(variant: variant)
                cfg.backend = .ane
                liveDenoiser = try await AsyncDPDFNetEnhancer.create(config: cfg)
            } else {
                let variant = ModelVariant(rawValue: dfnModel) ?? .standard
                FileHandle.standardError.write(Data("starting live ANE denoise (DeepFilterNet '\(variant.rawValue)', 48kHz)...\n".utf8))
                liveDenoiser = try await AsyncStreamingDenoiser.create(
                    config: .init(variant: variant, backend: .ane, compensateDelay: true, enableProfiling: false))
            }
        }

        let runner = CompareRunner(
            providers: providers, vad: vad,
            feedSamples: max(1, 16000 * feedMs / 1000), inputDevice: inputDevice,
            recordURL: recordPath.map { URL(fileURLWithPath: $0) }, denoiser: liveDenoiser)

        if let wavPath {
            let samples: [Float]
            if denoise {
                let outURL = URL(fileURLWithPath: wavPath).deletingPathExtension().appendingPathExtension("denoised.wav")
                // Prefer AudioEnhanceKit's tish-denoise — DeepFilterNet on the ANE
                // (no GPU contention with the local ASR models). MLX/GPU is the
                // fallback if the binary isn't found.
                let aneTish = tishPath ?? [
                    "/Volumes/DATA/AudioEnhanceKit/.build/release/tish-denoise",
                    "/Volumes/DATA/AudioEnhanceKit/.build/arm64-apple-macosx/release/tish-denoise",
                ].first { FileManager.default.isExecutableFile(atPath: $0) }

                if let aneTish {
                    FileHandle.standardError.write(Data("denoising on ANE (AudioEnhanceKit DFN '\(dfnModel)')...\n".utf8))
                    let p = Process()
                    p.executableURL = URL(fileURLWithPath: aneTish)
                    p.arguments = ["--model", dfnModel, wavPath, outURL.path]
                    p.standardOutput = FileHandle.nullDevice; p.standardError = FileHandle.nullDevice
                    try p.run(); p.waitUntilExit()
                    guard p.terminationStatus == 0 else {
                        FileHandle.standardError.write(Data("tish-denoise failed (exit \(p.terminationStatus))\n".utf8)); exit(1)
                    }
                    let (sr, raw) = try loadAudioArray(from: outURL, sampleRate: 16000)
                    precondition(sr == 16000)
                    samples = (raw.ndim > 1 ? raw.mean(axis: -1) : raw).asType(.float32).asArray(Float.self)
                    FileHandle.standardError.write(Data("denoised (ANE) -> \(outURL.path) (listen) ; feeding cleaned audio\n".utf8))
                } else {
                    // MLX/GPU DeepFilterNet (48 kHz, offline). Needs a local DFN dir
                    // via --dfn (hf hub-cache download yields xet pointers).
                    FileHandle.standardError.write(Data("denoising on GPU (MLX DeepFilterNet v3)...\n".utf8))
                    let dfn: DeepFilterNetModel
                    if let local = dfnRepo ?? Self.dfnSnapshotDir() {
                        dfn = try await DeepFilterNetModel.fromPretrained(local, subfolder: "v3")
                    } else {
                        dfn = try await DeepFilterNetModel.fromPretrained()
                    }
                    let (_, raw48) = try loadAudioArray(from: URL(fileURLWithPath: wavPath), sampleRate: 48000)
                    let mono48 = (raw48.ndim > 1 ? raw48.mean(axis: -1) : raw48).asType(.float32)
                    let clean48f = try dfn.enhance(mono48).reshaped([-1]).asArray(Float.self)
                    writeWavMono(clean48f, sampleRate: 48000, to: outURL)
                    samples = try resampleAudio(clean48f, from: 48000, to: 16000)
                    FileHandle.standardError.write(Data("denoised (GPU) -> \(outURL.path) (listen) ; feeding cleaned audio\n".utf8))
                }
            } else {
                let (sr, raw) = try loadAudioArray(from: URL(fileURLWithPath: wavPath), sampleRate: 16000)
                precondition(sr == 16000, "expected 16k, got \(sr)")
                samples = (raw.ndim > 1 ? raw.mean(axis: -1) : raw).asType(.float32).asArray(Float.self)
            }
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

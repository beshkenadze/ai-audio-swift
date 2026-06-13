import Foundation
import MLXAudioSTT

// Provider abstraction so mic-compare can stack any number of ASR engines —
// local (Nemotron / Voxtral) and cloud (DeepGram / Gemini) — fed from one mic.

struct AudioFrame {
    let samples: [Float]   // 16 kHz mono float, one model chunk
    let pcm16le: Data      // same audio as little-endian Int16 (for cloud sockets)
    let speechActive: Bool // VAD gate state
    let wallNow: Double     // seconds since session start
    let audioEndS: Double   // cumulative audio seconds through this frame
}

struct Snap {
    var label: String
    var text = ""
    var ttft = -1.0       // speech onset → first text
    var lag = -1.0        // -1 = N/A
    var perf = ""         // local: "step 76ms · RTF 0.17"; cloud: "net 140ms"
    var note = ""         // "VAD-gated" / "cloud · key?"
}

protocol LiveASR: AnyObject {
    var label: String { get }
    var isLocal: Bool { get }
    func feed(_ frame: AudioFrame)
    func finish()
    func snapshot() -> Snap
}

/// Wraps a local streaming session via a `step([Float]) -> fullText` closure.
/// Stepping must happen on the runner's serial queue (MLX isn't concurrent).
final class LocalASR: LiveASR {
    let label: String
    let isLocal = true
    private let step: ([Float]) -> String
    private let finishFn: () -> String
    private let gated: Bool

    private let lock = NSLock()
    private var snap: Snap
    private var chunks = 0, audioSamples = 0, skipped = 0
    private var stepMsTotal = 0.0, lastStepMs = 0.0
    private var speechWall = -1.0, firstText = -1.0

    init(label: String, gated: Bool, step: @escaping ([Float]) -> String, finish: @escaping () -> String) {
        self.label = label + (gated ? " ·VAD-gated" : "")
        self.gated = gated
        self.step = step
        self.finishFn = finish
        self.snap = Snap(label: self.label)
    }

    func feed(_ f: AudioFrame) {
        if gated && !f.speechActive { lock.lock(); skipped += f.samples.count; lock.unlock(); return }
        let t0 = ProcessInfo.processInfo.systemUptime
        let text = step(f.samples)
        let ms = (ProcessInfo.processInfo.systemUptime - t0) * 1000
        lock.lock()
        lastStepMs = ms; stepMsTotal += ms; chunks += 1; audioSamples += f.samples.count
        if firstText < 0, !text.trimmingCharacters(in: .whitespaces).isEmpty {
            if speechWall < 0 { speechWall = f.wallNow }
            firstText = f.wallNow - speechWall
        }
        if f.speechActive, speechWall < 0 { speechWall = f.wallNow }
        let audioS = Double(audioSamples) / 16000.0
        let timeline = Double(audioSamples + skipped) / 16000.0
        snap.text = text
        snap.ttft = firstText
        snap.lag = max(0, f.wallNow - timeline)
        snap.perf = String(format: "step %.0fms · RTF %.2f", lastStepMs, audioS > 0 ? (stepMsTotal/1000)/audioS : 0)
        lock.unlock()
    }

    func finish() {
        let text = finishFn()
        lock.lock(); snap.text = text; lock.unlock()
    }

    func snapshot() -> Snap { lock.lock(); defer { lock.unlock() }; return snap }
}

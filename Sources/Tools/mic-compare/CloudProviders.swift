import Foundation

private extension Data {
    func append(to url: URL) throws {
        if let h = try? FileHandle(forWritingTo: url) {
            defer { try? h.close() }
            try h.seekToEnd(); h.write(self)
        } else {
            try write(to: url)
        }
    }
}

// Cloud streaming ASR over WebSockets. Both consume the same 16 kHz mono Int16
// PCM the local pipeline produces. Keys come from .env (see Env.swift).
//
// Caveat: cloud = network round-trip + per-minute cost + audio leaves the device,
// and the exact JSON wire shapes drift between API versions — adjust if a provider
// changes its schema. Tested by the user with real keys (paid).

// MARK: - DeepGram (Nova-2 streaming; Russian supported)

final class DeepgramASR: LiveASR {
    let label = "DEEPGRAM nova-2 (cloud)"
    let isLocal = false
    private let lock = NSLock()
    private var snap: Snap
    private var task: URLSessionWebSocketTask?
    private var finals: [String] = []
    private var interim = ""
    private var lastWallNow = 0.0
    private var speechWall = -1.0
    private var firstText = -1.0

    init(key: String, language: String, model: String = "nova-2") {
        snap = Snap(label: label, note: "cloud")
        var comps = URLComponents(string: "wss://api.deepgram.com/v1/listen")!
        comps.queryItems = [
            .init(name: "encoding", value: "linear16"),
            .init(name: "sample_rate", value: "16000"),
            .init(name: "channels", value: "1"),
            .init(name: "model", value: model),
            .init(name: "language", value: language),
            .init(name: "interim_results", value: "true"),
            .init(name: "punctuate", value: "true"),
        ]
        var req = URLRequest(url: comps.url!)
        req.setValue("Token \(key)", forHTTPHeaderField: "Authorization")
        let t = URLSession.shared.webSocketTask(with: req)
        task = t
        t.resume()
        receive()
    }

    func feed(_ f: AudioFrame) {
        lock.lock(); lastWallNow = f.wallNow; if f.speechActive, speechWall < 0 { speechWall = f.wallNow }; lock.unlock()
        task?.send(.data(f.pcm16le)) { _ in }
    }

    private func receive() {
        task?.receive { [weak self] result in
            guard let self else { return }
            switch result {
            case .failure:
                self.lock.lock(); self.snap.note = "cloud · disconnected"; self.lock.unlock()
            case .success(let msg):
                switch msg {
                case .string(let s): self.handle(s)
                case .data(let d): if let s = String(data: d, encoding: .utf8) { self.handle(s) }
                @unknown default: break
                }
                self.receive()
            }
        }
    }

    private func handle(_ s: String) {
        guard let data = s.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let ch = json["channel"] as? [String: Any],
              let alts = ch["alternatives"] as? [[String: Any]],
              let transcript = alts.first?["transcript"] as? String else { return }
        let isFinal = (json["is_final"] as? Bool) ?? false
        let start = (json["start"] as? Double) ?? 0
        let dur = (json["duration"] as? Double) ?? 0
        lock.lock()
        if isFinal { if !transcript.isEmpty { finals.append(transcript) }; interim = "" }
        else { interim = transcript }
        if firstText < 0, !transcript.trimmingCharacters(in: .whitespaces).isEmpty, speechWall >= 0 {
            firstText = lastWallNow - speechWall
        }
        snap.text = (finals + (interim.isEmpty ? [] : [interim])).joined(separator: " ")
        snap.ttft = firstText
        snap.lag = max(0, lastWallNow - (start + dur))   // wall trailing the audio it covers
        snap.perf = "net"
        lock.unlock()
    }

    func finish() {
        task?.send(.string("{\"type\":\"CloseStream\"}")) { _ in }
        task?.cancel(with: .goingAway, reason: nil)
    }

    func snapshot() -> Snap { lock.lock(); defer { lock.unlock() }; return snap }
}

// MARK: - Gemini Live (BidiGenerateContent; input transcription)

final class GeminiASR: LiveASR {
    let label = "GEMINI live (cloud)"
    let isLocal = false
    private let lock = NSLock()
    private var snap: Snap
    private var task: URLSessionWebSocketTask?
    private var ready = false
    private var transcript = ""
    private var lastWallNow = 0.0
    private var speechWall = -1.0
    private var firstText = -1.0

    init(key: String, model: String) {
        snap = Snap(label: label, note: "cloud")
        let url = URL(string: "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key=\(key)")!
        let t = URLSession.shared.webSocketTask(with: url)
        task = t
        t.resume()
        // setup: model + input transcription. Native-audio models require an AUDIO
        // response modality even when we only read input transcription.
        let setup: [String: Any] = [
            "setup": [
                "model": "models/\(model)",
                "generationConfig": ["responseModalities": ["AUDIO"]],
                "inputAudioTranscription": [:],
            ],
        ]
        if let d = try? JSONSerialization.data(withJSONObject: setup), let s = String(data: d, encoding: .utf8) {
            t.send(.string(s)) { [weak self] err in if let err { self?.dbg("SETUP SEND ERR: \(err)") } else { self?.dbg("SETUP SENT: \(s)") } }
        }
        receive()
    }

    func feed(_ f: AudioFrame) {
        lock.lock(); lastWallNow = f.wallNow; if f.speechActive, speechWall < 0 { speechWall = f.wallNow }; let r = ready; lock.unlock()
        guard r else { return }
        let b64 = f.pcm16le.base64EncodedString()
        let msg: [String: Any] = ["realtimeInput": ["mediaChunks": [["mimeType": "audio/pcm;rate=16000", "data": b64]]]]
        if let d = try? JSONSerialization.data(withJSONObject: msg), let s = String(data: d, encoding: .utf8) {
            task?.send(.string(s)) { _ in }
        }
    }

    private func dbg(_ s: String) {
        guard ProcessInfo.processInfo.environment["GEMINI_DEBUG"] != nil else { return }
        try? (s + "\n").data(using: .utf8)?.append(to: URL(fileURLWithPath: "/tmp/gemini-raw.jsonl"))
    }

    private func receive() {
        task?.receive { [weak self] result in
            guard let self else { return }
            switch result {
            case .failure(let e):
                self.dbg("FAILURE: \(e)")
                self.lock.lock(); self.snap.note = "cloud · disconnected"; self.lock.unlock()
            case .success(let msg):
                switch msg {
                case .string(let s): self.handle(s)
                case .data(let d): self.handle(String(data: d, encoding: .utf8) ?? "<\(d.count) bytes binary>")
                @unknown default: break
                }
                self.receive()
            }
        }
    }

    private func handle(_ s: String) {
        dbg(s)
        guard let data = s.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return }
        lock.lock()
        if json["setupComplete"] != nil { ready = true }
        if let sc = json["serverContent"] as? [String: Any],
           let it = sc["inputTranscription"] as? [String: Any],
           let text = it["text"] as? String {
            transcript += text
            if firstText < 0, !text.trimmingCharacters(in: .whitespaces).isEmpty, speechWall >= 0 {
                firstText = lastWallNow - speechWall
            }
            snap.text = transcript
            snap.ttft = firstText
            snap.perf = "net"
        }
        lock.unlock()
    }

    func finish() { task?.cancel(with: .goingAway, reason: nil) }
    func snapshot() -> Snap { lock.lock(); defer { lock.unlock() }; return snap }
}

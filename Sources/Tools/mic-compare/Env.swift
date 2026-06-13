import Foundation

// Tiny .env reader (KEY=value lines, # comments). Looks in the current dir and
// the executable's dir so `mic-compare` finds it whether run from the repo root
// or the build dir.
enum Env {
    private static let values: [String: String] = {
        var out: [String: String] = [:]
        let candidates = [
            FileManager.default.currentDirectoryPath + "/.env",
            (Bundle.main.executableURL?.deletingLastPathComponent().path ?? ".") + "/.env",
        ]
        for path in candidates {
            guard let text = try? String(contentsOfFile: path, encoding: .utf8) else { continue }
            for raw in text.split(separator: "\n") {
                let line = raw.trimmingCharacters(in: .whitespaces)
                if line.isEmpty || line.hasPrefix("#") { continue }
                guard let eq = line.firstIndex(of: "=") else { continue }
                let k = String(line[..<eq]).trimmingCharacters(in: .whitespaces)
                var v = String(line[line.index(after: eq)...]).trimmingCharacters(in: .whitespaces)
                if v.count >= 2, (v.hasPrefix("\"") && v.hasSuffix("\"")) || (v.hasPrefix("'") && v.hasSuffix("'")) {
                    v = String(v.dropFirst().dropLast())
                }
                if out[k] == nil { out[k] = v }
            }
            break  // first readable .env wins
        }
        return out
    }()

    /// Process environment takes precedence, then .env.
    static func value(_ key: String) -> String? {
        if let v = ProcessInfo.processInfo.environment[key], !v.isEmpty { return v }
        if let v = values[key], !v.isEmpty { return v }
        return nil
    }
}

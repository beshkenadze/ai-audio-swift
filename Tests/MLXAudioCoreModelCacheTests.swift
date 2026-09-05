import Foundation
import HuggingFace
import Testing

@testable import MLXAudioCore

/// Pure-FileManager tests for `ModelUtils.resolveOrDownloadModel` cache validation.
///
/// These exercise only the cache-hit branch, which returns before any network or
/// MLX/Metal work, so they run offline in the test sandbox.
struct ModelCacheValidationTests {

    /// A resource-only repo (e.g. G2P dictionaries) ships no `config.json`. When the
    /// required-extension file is present and non-empty, the cached directory must be
    /// reused instead of re-triggering a download.
    @Test func cachedRepoWithRequiredFilesButNoConfigJSONIsReused() async throws {
        let tempRoot = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("mlxaudio-cache-test-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: tempRoot) }

        let cache = HubCache(cacheDirectory: tempRoot)
        let repoID = try #require(Repo.ID(rawValue: "beshkenadze/kitten-tts-g2p"))

        // Pre-create the cache dir exactly where resolveOrDownloadModel expects it,
        // with a non-empty required-extension file and NO config.json.
        let modelSubdir = repoID.description.replacingOccurrences(of: "/", with: "_")
        let modelDir = tempRoot
            .appendingPathComponent("mlx-audio")
            .appendingPathComponent(modelSubdir)
        try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)
        try Data("us_gold.json contents".utf8)
            .write(to: modelDir.appendingPathComponent("us_gold.json"))

        #expect(!FileManager.default.fileExists(
            atPath: modelDir.appendingPathComponent("config.json").path))

        let client = HubClient(cache: cache)
        let resolved = try await ModelUtils.resolveOrDownloadModel(
            client: client,
            cache: cache,
            repoID: repoID,
            requiredExtension: "json"
        )

        #expect(resolved.standardizedFileURL.path == modelDir.standardizedFileURL.path)
        // The cache must be left untouched (not cleared/re-downloaded).
        #expect(FileManager.default.fileExists(
            atPath: modelDir.appendingPathComponent("us_gold.json").path))
    }
}

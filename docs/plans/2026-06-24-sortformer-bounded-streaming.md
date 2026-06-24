# Sortformer Bounded Long-Form Streaming — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a memory-bounded long-form streaming diarization path to `SortformerModel` so multi-hour recordings run at flat memory and flat per-chunk latency, plus a `mlx-audio-swift-diar` benchmark tool.

**Architecture:** Fix the unbounded-FIFO compression cap (D1), then replace whole-file mel + `preEncode` with a phase-aligned sliding window (D2) feeding a refactored `streamingStepFromEmbeddings`, exposed via a new pull-closure `generateStreamBounded`. See companion design doc: `docs/plans/2026-06-24-sortformer-bounded-streaming-design.md` — read it first for the windowing math and constants.

**Tech Stack:** Swift, MLX-Swift, MLXNN, AVFoundation (`AVAudioFile`/`AVAudioConverter`), XCTest.

**Conventions:**
- Build CLI tools with `swift build --product <name>`; run the `.build` binary for Metal (per project memory, `swift test` can't load the metallib).
- Pure-logic tasks are strict TDD. Encoder/streaming-wiring tasks are verified by the self-consistency gate in Task 7 (cannot unit-test without model weights + Metal).
- Commit after every green step. Do not add Claude/Co-Authored-By trailers.
- Note: MLX-Swift's `MLX.eval` forces lazy-graph evaluation (nothing to do with code evaluation); use it to stop the graph growing across steps.

---

### Task 1: D1 — make `maybeCompressState` a true FIFO cap

**Files:**
- Modify: `Sources/MLXAudioVAD/Models/Sortformer/Sortformer.swift` (`maybeCompressState`, ~lines 1018–1084; change `private static` → `static` for testability)
- Test: `Tests/MLXAudioVADTests.swift` (add `SortformerFifoCapTests` cases)

**Step 1: Write the failing test**

Construct a state whose FIFO overflows by more than `spkcacheUpdatePeriod` and assert the cap holds after one call. Helper to build a state with a FIFO of `n` frames:

```swift
// @testable import MLXAudioVAD  // ensure present at top of test file
private func makeState(fifoFrames: Int, embDim: Int, nSpk: Int) -> StreamingState {
    StreamingState(
        spkcache: MLXArray.zeros([1, 0, embDim]),
        spkcachePreds: MLXArray.zeros([1, 0, nSpk]),
        fifo: MLXArray.zeros([1, fifoFrames, embDim]),
        fifoPreds: MLXArray.zeros([1, fifoFrames, nSpk]),
        framesProcessed: 0,
        meanSilEmb: MLXArray.zeros([1, embDim]),
        nSilFrames: MLXArray.zeros([1])
    )
}

func testFifoCappedAfterOversizedChunk() throws {
    let cfg = try loadSortformerModulesConfigAOSC() // see note below
    let embDim = cfg.fcDModel, nSpk = cfg.numSpeakers
    // FIFO overflow (700) >> spkcacheUpdatePeriod (188): the old code left it > fifoMax.
    let state = makeState(fifoFrames: 700, embDim: embDim, nSpk: nSpk)
    let out = SortformerModel.maybeCompressState(state, spkcacheMax: 188, fifoMax: 188, modulesCfg: cfg)
    XCTAssertLessThanOrEqual(out.fifoLen, 188, "FIFO must be capped regardless of overflow size")
    XCTAssertLessThanOrEqual(out.spkcacheLen, 188, "spkcache must stay capped")
}
```
Note: build `cfg` with `useAosc=true`, `spkcacheUpdatePeriod=188`, `spkcacheLen=188`, `numSpeakers=4`, `fcDModel=512` etc. Construct `ModulesConfig` from a JSON literal matching the v2.1 config and decode it. Keep the fixture next to the test.

**Step 2: Run test to verify it fails**

Run: `swift test --filter SortformerFifoCapTests/testFifoCappedAfterOversizedChunk`
Expected: FAIL — `out.fifoLen` is `700 − 188 = 512`, not `≤ 188` (and/or compile error until `maybeCompressState` is `internal`).

**Step 3: Implement the loop cap**

Wrap the pop/compress body in a loop until the FIFO is within bound. Minimal shape:

```swift
static func maybeCompressState(
    _ state: StreamingState, spkcacheMax: Int, fifoMax: Int, modulesCfg: ModulesConfig
) -> StreamingState {
    var s = state
    while s.fifoLen > fifoMax {
        let before = s.fifoLen
        s = compressOnce(s, spkcacheMax: spkcacheMax, fifoMax: fifoMax, modulesCfg: modulesCfg)
        if s.fifoLen == before { break } // safety: no progress
    }
    return s
}
```
Rename the existing body (lines 1024–1084, minus the early `return state`) to `private static func compressOnce(...)`. `compressOnce` keeps the `popLen = min(fifoLen - fifoMax, spkcacheUpdatePeriod)` step so AOSC per-period semantics are preserved across iterations.

**Step 4: Run test to verify it passes**

Run: `swift test --filter SortformerFifoCapTests`
Expected: PASS.

**Step 5: Commit**

```bash
git add Sources/MLXAudioVAD/Models/Sortformer/Sortformer.swift Tests/MLXAudioVADTests.swift
git commit -m "fix(sortformer): loop FIFO compression to a true cap (bounds chunkDuration>15s)"
```

---

### Task 2: Refactor `streamingStep` → `streamingStepFromEmbeddings`

**Files:**
- Modify: `Sources/MLXAudioVAD/Models/Sortformer/Sortformer.swift` (`streamingStep`, lines 672–743)

**Step 1: Extract the post-preEncode body**

Create `func streamingStepFromEmbeddings(chunkEmbs: MLXArray, chunkDiarLen: Int, state: StreamingState, rightContextEmbs: MLXArray?) -> (MLXArray, StreamingState)` containing exactly lines 689–742 (left-context build, concat, encoder pass, slice, `updateStreamingState`). `chunkEmbs` must be **chunk frames only**.

Rewrite `streamingStep` to:
```swift
let chunkFeat = chunkFeatures.asType(modelDtype)
var (chunkEmbs, chunkEmbLengths) = fcEncoder.preEncode(chunkFeat, length: chunkLength)
let chunkDiarLen = Int(chunkEmbLengths[0].item(Int32.self))
chunkEmbs = chunkEmbs[0..., ..<chunkDiarLen, 0...]
return streamingStepFromEmbeddings(chunkEmbs: chunkEmbs, chunkDiarLen: chunkDiarLen,
                                   state: state, rightContextEmbs: rightContextEmbs)
```

**Step 2: Verify behaviour-preserving**

Run: `swift build` → Expected: builds clean. No logic change; `feed`/`generateStream` call sites unchanged. Full behavioural verification deferred to the Task 7 self-consistency gate.

**Step 3: Commit**

```bash
git add Sources/MLXAudioVAD/Models/Sortformer/Sortformer.swift
git commit -m "refactor(sortformer): split streamingStepFromEmbeddings out of streamingStep"
```

---

### Task 3: Pure window-planning helper (frame accounting)

This is the highest bug-risk area (Codex P1: STFT `+1` drift, halo grid alignment). Pure `Int` arithmetic — TDD hard, no MLX.

**Files:**
- Create: `Sources/MLXAudioVAD/Models/Sortformer/BoundedWindowPlanner.swift`
- Test: `Tests/MLXAudioVADTests.swift` (`BoundedWindowPlannerTests`)

**Step 1: Write failing tests** for a struct that, given step constants and `g0` (global mel index of first new frame) + availability, returns: raw sample range `[rawStart, rawEnd)` (hop-aligned start), mel-slice indices into the window, count of left-halo emb frames to discard, chunk emb-frame count, and rc emb-frame count. Assert:
- Interior step: `rawStart % hop == 0`; left halo = 16 mel / 2 emb; chunk emb = `chunkLen`; rc = 1; no overlap/gap vs previous step (`thisRawStart` consistent with `prevG0 + C`).
- First step (`g0 == 0`): left halo = 0, discard 0 emb frames, `rawStart == 0`.
- EOF step: when `availableFutureFrames < H_R`, rc shrinks to `min(1, availFutureEmb)`; chunk frames = remaining new frames.
- No `±1` drift: feeding contiguous steps reconstructs frames `[0, total)` exactly once each.

**Step 2: Run** `swift test --filter BoundedWindowPlannerTests` → FAIL (type missing).

**Step 3: Implement** `BoundedWindowPlanner` per the design doc "windowing recipe" (constants from config; `H_L=H_R=16`; `discardLeftEmb = H_L/8`; hop-aligned `rawStart`; STFT support `±nFft/2`). Keep it a value type with one `func plan(g0:totalKnownSamples:eof:) -> WindowSpec`.

**Step 4: Run** → PASS.

**Step 5: Commit**
```bash
git commit -am "feat(sortformer): pure BoundedWindowPlanner frame accounting + tests"
```

---

### Task 4: PCM accumulator

**Files:**
- Create: `Sources/MLXAudioVAD/Models/Sortformer/PCMAccumulator.swift`
- Test: `Tests/MLXAudioVADTests.swift` (`PCMAccumulatorTests`)

**Step 1: Write failing tests** for a struct that ingests arbitrary `[Float]` blocks, tracks the absolute sample index of its first retained sample, returns a contiguous `[Float]` slice for an absolute `[rawStart, rawEnd)` range (pulling more from the source if needed is the caller's job — the accumulator only stores + slices + drops). Assert: slice correctness across block boundaries; `drop(beforeAbsolute:)` frees memory but keeps the requested tail; retained count stays ≤ one step + halo after dropping.

**Step 2–4:** Run (FAIL) → implement → Run (PASS).

**Step 5: Commit**
```bash
git commit -am "feat(sortformer): bounded PCMAccumulator + tests"
```

---

### Task 5: Wire `generateStreamBounded`

**Files:**
- Modify: `Sources/MLXAudioVAD/Models/Sortformer/Sortformer.swift` (new method after `generateStream`)

**Step 1: Implement** per the design doc API. Loop: pull from `audioSource` into `PCMAccumulator` until the `BoundedWindowPlanner` window's samples are available (or EOF); build the window `MLXArray` **inside** this task (closure returns `[Float]`); `extractMelFeatures(window, normalize: nil, padTo: 0)`; slice aligned interior; `preEncode`; discard left-halo emb frames; split chunk/rc; `streamingStepFromEmbeddings`; `predsToSegments` offset by a global diar-frame counter; `yield`; `maybeCompressState`; `drop` consumed samples; force-evaluate the small per-step tensors and `state` arrays with `MLX.eval` so the lazy graph cannot grow across steps (Codex P2 — mirror existing evaluation discipline). Honour `Task.checkCancellation()` each step and set `continuation.onTermination` to stop the producer.

**Step 2: Build** `swift build` → clean.

**Step 3: Commit**
```bash
git commit -am "feat(sortformer): generateStreamBounded pull-closure long-form path"
```

---

### Task 6: `mlx-audio-swift-diar` benchmark tool

**Files:**
- Create: `Sources/Tools/mlx-audio-swift-diar/App.swift`, `Sources/Tools/mlx-audio-swift-diar/README.md`
- Modify: `Package.swift` (add `.executable` product + `.executableTarget`, mirror `mlx-audio-swift-lid` at lines ~57 and ~267)

**Step 1: Implement** mirroring `mlx-audio-swift-lid/App.swift`:
- `GPU.set(memoryLimit: 18 * 1024 * 1024 * 1024, relaxed: false)` at startup.
- Windowed reader: open `AVAudioFile`; create a persistent `AVAudioConverter` (file `processingFormat` → 16 kHz mono Float32); the `audioSource` closure pulls a fixed output block (e.g. 1 s) via `converter.convert(to:error:withInputFrom:)`, whose input block reads the next native window from the file; downmix happens via the converter's channel-count change; return `[Float]?` (nil at EOF).
- Drive `generateStreamBounded`; accumulate segments; write RTTM (`--rttm-out`).
- Metrics: total wall time, per-chunk latency series → min/median/max + first-vs-last trend, peak RSS via `task_info(mach_task_self_, TASK_BASIC_INFO, …).resident_size`, MLX peak (`GPU.snapshot()`/peak — **verify exact symbol**, fall back to `Memory` if absent), segment count, RTF.
- Flags: `--input`, `--repo` (default `mlx-community/diar_streaming_sortformer_4spk-v2.1-fp16`), `--chunk-duration`, `--threshold`, `--rttm-out`, `--verbose`.

**Step 2: Build** `swift build --product mlx-audio-swift-diar` → clean.

**Step 3: Commit**
```bash
git add Sources/Tools/mlx-audio-swift-diar Package.swift
git commit -m "feat(tools): mlx-audio-swift-diar bounded long-form bench"
```

---

### Task 7: Self-consistency gate + long-run benchmark

**Step 1: Short-clip self-consistency** (regression guard). Pick/trim a ≤3-min clip. Run `generateStream` (precompute) and `generateStreamBounded` on identical 16 kHz mono PCM; assert RTTM segment sets match within a tiny DER tolerance (e.g. boundaries within ±1 frame, same speaker labels). Wire as a script or a guarded integration test invoked through the built binary.

**Step 2: Long-run bench** on the user-provided file:
```bash
swift build --product mlx-audio-swift-diar
.build/debug/mlx-audio-swift-diar \
  --input "/Users/akira/Library/Application Support/Tish/Recordings/2B165E75-71D3-486D-9927-41AE62636B51/mix_stereo.flac" \
  --chunk-duration 15 --rttm-out /tmp/diar_2B165E75.rttm --verbose
```
Expected/assert: completes 100 %; per-chunk latency **flat** (last ≈ first, not growing); peak RSS + MLX peak **well under 18 GB and not duration-scaling**; non-trivial segment count. Record wall time, latency trend, peak memory, segments into a results appendix in the design doc or a `bench/REPORT.md`.

**Step 3:** If self-consistency or memory-flatness fails, STOP and diagnose (likely a frame-accounting drift from Task 3 or a missing `MLX.eval` in Task 5) before proceeding.

**Step 4: Commit** results/report.

---

### Task 8: Docs — three inference paths

**Files:**
- Modify: doc-comments on `generate`, `generateStream`, `generateStreamBounded` in `Sortformer.swift`
- Modify/create: `Sources/Tools/mlx-audio-swift-diar/README.md` + a section in the design doc

**Step 1:** Document clearly: `generate` = offline single-pass, short clips only (OOMs long); `generateStream` = precompute streaming, **not memory-bounded** (whole-file mel + AOSC full-file `preEncode`), short/medium clips; `generateStreamBounded` = the memory-bounded long-form path (pull closure, flat memory). Note the D1 cap now protects all paths.

**Step 2: Commit**
```bash
git commit -am "docs(sortformer): clarify generate vs generateStream vs generateStreamBounded"
```

---

## Execution order & checkpoints

1 → 2 → 3 → 4 → 5 → 6 → 7 → 8. Tasks 1, 3, 4 are independently testable now. Task 7 is the integration truth gate; do not declare done until both its sub-gates pass on the 67.6-min file.

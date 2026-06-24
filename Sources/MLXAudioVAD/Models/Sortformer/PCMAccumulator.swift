import Foundation

/// Bounded buffer for incoming mono PCM in the memory-bounded Sortformer streaming path.
///
/// Samples are addressed in **absolute coordinates** over the whole stream: index 0 is the very
/// first sample ever appended, and addresses stay stable even after older samples are dropped to
/// free memory. The `generateStreamBounded` loop pulls arbitrary-length `[Float]` blocks in,
/// asks for the exact `[WindowSpec.rawStart, WindowSpec.rawEnd)` slice it needs, then drops the
/// samples it has consumed — keeping retained memory bounded to roughly one step plus a halo.
///
/// Single-threaded by design: the Task-5 loop owns one instance inside one `Task`, so a value
/// type with `mutating` methods is sufficient (no actor / locking needed).
public struct PCMAccumulator {
    /// Retained samples, oldest first. Index 0 corresponds to absolute index `baseAbsolute`.
    private var buffer: [Float]
    /// Absolute index of `buffer[0]` — the first retained sample.
    private var baseAbsolute: Int
    /// Total number of samples ever appended (the exclusive upper bound of valid absolute indices).
    private var totalSeen: Int

    public init() {
        buffer = []
        baseAbsolute = 0
        totalSeen = 0
    }

    /// Absolute index of the first sample still retained. Slicing below this traps.
    public var firstRetainedAbsolute: Int { baseAbsolute }

    /// Total number of samples seen so far across the whole stream (absolute upper bound, exclusive).
    public var totalAppendedCount: Int { totalSeen }

    /// Number of samples currently held in memory (for bounded-memory assertions).
    public var retainedCount: Int { buffer.count }

    /// Ingest a block of mono PCM. Blocks may be any length; they are concatenated in order.
    public mutating func append(_ samples: [Float]) {
        buffer.append(contentsOf: samples)
        totalSeen += samples.count
    }

    /// Return the contiguous samples for absolute range `[absoluteStart, absoluteEnd)`.
    ///
    /// **Contract:** the requested range must lie within `[firstRetainedAbsolute, totalAppendedCount)`.
    /// Asking for already-dropped samples (`absoluteStart < firstRetainedAbsolute`) or not-yet-arrived
    /// samples (`absoluteEnd > totalAppendedCount`) is a precondition failure (trap). The caller must
    /// buffer enough (and not drop too eagerly) before slicing.
    public func slice(from absoluteStart: Int, to absoluteEnd: Int) -> [Float] {
        precondition(absoluteStart <= absoluteEnd, "slice start must be <= end")
        precondition(absoluteStart >= baseAbsolute,
                     "slice start \(absoluteStart) is before the first retained sample \(baseAbsolute)")
        precondition(absoluteEnd <= totalSeen,
                     "slice end \(absoluteEnd) exceeds total appended \(totalSeen)")
        let lo = absoluteStart - baseAbsolute
        let hi = absoluteEnd - baseAbsolute
        return Array(buffer[lo..<hi])
    }

    /// Discard samples before absolute index `absoluteIndex`, freeing memory while retaining
    /// everything from `absoluteIndex` onward. Safe to call repeatedly and idempotent: an index at
    /// or below `firstRetainedAbsolute` (already dropped, or never retained) is a no-op. An index
    /// past `totalAppendedCount` is clamped to drop only what has actually arrived.
    public mutating func drop(beforeAbsolute absoluteIndex: Int) {
        let target = min(absoluteIndex, totalSeen)
        guard target > baseAbsolute else { return }
        let removeCount = target - baseAbsolute
        buffer.removeFirst(removeCount)
        baseAbsolute = target
    }
}

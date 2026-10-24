import Foundation

/// Bounds sustained GPU duty without changing model math or sampler steps.
/// A single kernel/window can still briefly occupy the GPU fully.
final class InferencePacer {
    private let clock = ContinuousClock()
    private var began: ContinuousClock.Instant
    private let dutyCycle: Double
    private let synchronize: () -> Void
    init(dutyCycle: Double, synchronize: @escaping () -> Void) {
        self.dutyCycle = dutyCycle
        self.synchronize = synchronize
        began = clock.now
    }
    /// Starts the next work interval without resting, after time that was not GPU work.
    func restart() {
        began = clock.now
    }

    func rest() {
        guard dutyCycle < 1 else { return }
        synchronize()
        let duration = began.duration(to: clock.now).components
        let work = Double(duration.seconds) + Double(duration.attoseconds) / 1e18
        let delay = work * (1 / dutyCycle - 1)
        if delay > 0 { Thread.sleep(forTimeInterval: delay) }
        began = clock.now
    }
}

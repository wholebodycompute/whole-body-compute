import Foundation
import os.lock

/// Thread-safe global Kill Switch and telemetry counter state for `Whole Body Compute.app`.
///
/// Tracks independent organ pause states:
/// - `isPaused` (Eyes & Hands - `AXUIElement` DOM & `CGEvent` actuation)
/// - `isEarsPaused` (Ears - Hold-Fn dictation & `AVAudioEngine` microphone)
public final class KillSwitch: @unchecked Sendable {
    public static let shared = KillSwitch()

    public struct Snapshot: Sendable, Equatable {
        public let isPaused: Bool
        public let isEarsPaused: Bool
        public let requestsServed: UInt64
        public let requestsBlocked: UInt64
        public let lastCallerSummary: String
        public let lastUpdated: Date
    }

    private var lock = os_unfair_lock_s()
    private var _isPaused: Bool = false
    private var _isEarsPaused: Bool = false
    private var _requestsServed: UInt64 = 0
    private var _requestsBlocked: UInt64 = 0
    private var _lastCallerSummary: String = "None"
    private var _lastUpdated: Date = Date()
    private var observers: [UUID: @Sendable (Snapshot) -> Void] = [:]

    public init(initiallyPaused: Bool = false, initiallyEarsPaused: Bool = false) {
        self._isPaused = initiallyPaused
        self._isEarsPaused = initiallyEarsPaused || EarsEngine.isEarsPausedOnDisk()
    }

    public var isPaused: Bool {
        os_unfair_lock_lock(&lock)
        let val = _isPaused
        os_unfair_lock_unlock(&lock)
        return val
    }

    public var isEarsPaused: Bool {
        os_unfair_lock_lock(&lock)
        let val = _isEarsPaused
        os_unfair_lock_unlock(&lock)
        return val
    }

    private func makeUnlockedSnapshot() -> Snapshot {
        Snapshot(
            isPaused: _isPaused,
            isEarsPaused: _isEarsPaused,
            requestsServed: _requestsServed,
            requestsBlocked: _requestsBlocked,
            lastCallerSummary: _lastCallerSummary,
            lastUpdated: _lastUpdated
        )
    }

    public func snapshot() -> Snapshot {
        os_unfair_lock_lock(&lock)
        let snap = makeUnlockedSnapshot()
        os_unfair_lock_unlock(&lock)
        return snap
    }

    @discardableResult
    public func setPaused(_ paused: Bool) -> Snapshot {
        var cbs: [@Sendable (Snapshot) -> Void] = []
        let snap: Snapshot
        os_unfair_lock_lock(&lock)
        _isPaused = paused
        _lastUpdated = Date()
        snap = makeUnlockedSnapshot()
        cbs = Array(observers.values)
        os_unfair_lock_unlock(&lock)
        for cb in cbs { cb(snap) }
        return snap
    }

    @discardableResult
    public func setEarsPaused(_ paused: Bool, persist: Bool = true) -> Snapshot {
        if persist {
            EarsEngine.persistEarsPausedState(paused)
        }
        var cbs: [@Sendable (Snapshot) -> Void] = []
        let snap: Snapshot
        os_unfair_lock_lock(&lock)
        _isEarsPaused = paused
        _lastUpdated = Date()
        snap = makeUnlockedSnapshot()
        cbs = Array(observers.values)
        os_unfair_lock_unlock(&lock)
        for cb in cbs { cb(snap) }
        return snap
    }

    @discardableResult
    public func toggle() -> Snapshot {
        setPaused(!isPaused)
    }

    @discardableResult
    public func toggleEars(persist: Bool = true) -> Snapshot {
        setEarsPaused(!isEarsPaused, persist: persist)
    }

    @discardableResult
    public func toggleAll(persist: Bool = true) -> Snapshot {
        let shouldPauseAll = !(isPaused && isEarsPaused)
        if persist {
            EarsEngine.persistEarsPausedState(shouldPauseAll)
        }
        var cbs: [@Sendable (Snapshot) -> Void] = []
        let snap: Snapshot
        os_unfair_lock_lock(&lock)
        _isPaused = shouldPauseAll
        _isEarsPaused = shouldPauseAll
        _lastUpdated = Date()
        snap = makeUnlockedSnapshot()
        cbs = Array(observers.values)
        os_unfair_lock_unlock(&lock)
        for cb in cbs { cb(snap) }
        return snap
    }

    public func recordServed(callerSummary: String) {
        var cbs: [@Sendable (Snapshot) -> Void] = []
        let snap: Snapshot
        os_unfair_lock_lock(&lock)
        _requestsServed &+= 1
        _lastCallerSummary = callerSummary
        _lastUpdated = Date()
        snap = makeUnlockedSnapshot()
        cbs = Array(observers.values)
        os_unfair_lock_unlock(&lock)
        for cb in cbs { cb(snap) }
    }

    public func recordBlocked(callerSummary: String? = nil) {
        var cbs: [@Sendable (Snapshot) -> Void] = []
        let snap: Snapshot
        os_unfair_lock_lock(&lock)
        _requestsBlocked &+= 1
        if let callerSummary, !callerSummary.isEmpty {
            _lastCallerSummary = callerSummary
        }
        _lastUpdated = Date()
        snap = makeUnlockedSnapshot()
        cbs = Array(observers.values)
        os_unfair_lock_unlock(&lock)
        for cb in cbs { cb(snap) }
    }

    @discardableResult
    public func addObserver(_ observer: @escaping @Sendable (Snapshot) -> Void) -> UUID {
        let id = UUID()
        os_unfair_lock_lock(&lock)
        observers[id] = observer
        os_unfair_lock_unlock(&lock)
        return id
    }

    public func removeObserver(_ id: UUID) {
        os_unfair_lock_lock(&lock)
        observers.removeValue(forKey: id)
        os_unfair_lock_unlock(&lock)
    }
}

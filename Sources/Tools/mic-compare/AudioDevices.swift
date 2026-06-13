@preconcurrency import AVFoundation
import CoreAudio
import Foundation

// Minimal Core Audio helpers: enumerate input devices, resolve the default,
// match by UID/name, and point an AVAudioEngine's input node at a chosen device.
// AVAudioEngine otherwise silently uses the *system default* input — which is
// often not the device you want (built-in vs USB mic).

enum AudioDevices {
    struct Device { let id: AudioDeviceID; let name: String; let uid: String }

    private static let system = AudioObjectID(kAudioObjectSystemObject)

    private static func addr(
        _ selector: AudioObjectPropertySelector,
        _ scope: AudioObjectPropertyScope = kAudioObjectPropertyScopeGlobal
    ) -> AudioObjectPropertyAddress {
        AudioObjectPropertyAddress(mSelector: selector, mScope: scope, mElement: kAudioObjectPropertyElementMain)
    }

    private static func allDeviceIDs() -> [AudioDeviceID] {
        var a = addr(kAudioHardwarePropertyDevices)
        var size: UInt32 = 0
        guard AudioObjectGetPropertyDataSize(system, &a, 0, nil, &size) == noErr else { return [] }
        var ids = [AudioDeviceID](repeating: 0, count: Int(size) / MemoryLayout<AudioDeviceID>.size)
        guard AudioObjectGetPropertyData(system, &a, 0, nil, &size, &ids) == noErr else { return [] }
        return ids
    }

    private static func stringProp(_ id: AudioDeviceID, _ selector: AudioObjectPropertySelector) -> String {
        var a = addr(selector)
        var size = UInt32(MemoryLayout<CFString?>.size)
        var value: CFString?
        let st = AudioObjectGetPropertyData(id, &a, 0, nil, &size, &value)
        return st == noErr ? (value as String? ?? "?") : "?"
    }

    private static func inputChannelCount(_ id: AudioDeviceID) -> Int {
        var a = addr(kAudioDevicePropertyStreamConfiguration, kAudioObjectPropertyScopeInput)
        var size: UInt32 = 0
        guard AudioObjectGetPropertyDataSize(id, &a, 0, nil, &size) == noErr, size > 0 else { return 0 }
        let raw = UnsafeMutableRawPointer.allocate(byteCount: Int(size), alignment: MemoryLayout<AudioBufferList>.alignment)
        defer { raw.deallocate() }
        guard AudioObjectGetPropertyData(id, &a, 0, nil, &size, raw) == noErr else { return 0 }
        let list = UnsafeMutableAudioBufferListPointer(raw.assumingMemoryBound(to: AudioBufferList.self))
        return list.reduce(0) { $0 + Int($1.mNumberChannels) }
    }

    static func inputs() -> [Device] {
        allDeviceIDs()
            .filter { inputChannelCount($0) > 0 }
            .map { Device(id: $0, name: stringProp($0, kAudioObjectPropertyName), uid: stringProp($0, kAudioDevicePropertyDeviceUID)) }
    }

    static func defaultInput() -> AudioDeviceID? {
        var a = addr(kAudioHardwarePropertyDefaultInputDevice)
        var id = AudioDeviceID(0)
        var size = UInt32(MemoryLayout<AudioDeviceID>.size)
        guard AudioObjectGetPropertyData(system, &a, 0, nil, &size, &id) == noErr, id != 0 else { return nil }
        return id
    }

    /// Match by exact UID, UID substring, or name substring (case-insensitive).
    static func find(_ query: String) -> Device? {
        let q = query.lowercased()
        let all = inputs()
        return all.first { $0.uid.lowercased() == q }
            ?? all.first { $0.uid.lowercased().contains(q) || $0.name.lowercased().contains(q) }
    }

    static func name(of id: AudioDeviceID) -> String { stringProp(id, kAudioObjectPropertyName) }
    static func uid(of id: AudioDeviceID) -> String { stringProp(id, kAudioDevicePropertyDeviceUID) }

    /// Point the engine's input node at `id`. Must be called before reading the
    /// input format / starting the engine.
    static func setInput(_ id: AudioDeviceID, on engine: AVAudioEngine) throws {
        guard let unit = engine.inputNode.audioUnit else {
            throw NSError(domain: "AudioDevices", code: -1,
                          userInfo: [NSLocalizedDescriptionKey: "input node has no audio unit"])
        }
        var dev = id
        let st = AudioUnitSetProperty(
            unit, kAudioOutputUnitProperty_CurrentDevice, kAudioUnitScope_Global, 0,
            &dev, UInt32(MemoryLayout<AudioDeviceID>.size)
        )
        guard st == noErr else {
            throw NSError(domain: "AudioDevices", code: Int(st),
                          userInfo: [NSLocalizedDescriptionKey: "AudioUnitSetProperty(CurrentDevice) failed: \(st)"])
        }
    }
}

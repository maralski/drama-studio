import AVFoundation
import DramaEngine

@MainActor
final class StreamingPlayer {
    var onDrained: (() -> Void)?

    private let engine = AVAudioEngine()
    private let node = AVAudioPlayerNode()
    private let format = AVAudioFormat(standardFormatWithSampleRate: Double(ModelConstants.sampleRate), channels: 1)!
    private var timeline: PlaybackTimeline?
    private var epoch = 0
    private static let schedulingLead: Int64 = 2048

    init() {
        engine.attach(node)
        engine.connect(node, to: engine.mainMixerNode, format: format)
    }

    var isPlaying: Bool { timeline != nil }

    var position: Int? {
        timeline.map { $0.audioPosition(atNodeSample: nodeTime ?? 0) }
    }

    var isStarved: Bool {
        guard let timeline, let position else { return false }
        return position >= timeline.audioEnd
    }

    func play(_ audio: [Float], from offset: Int) throws {
        stop()
        if !engine.isRunning { try engine.start() }
        timeline = PlaybackTimeline(origin: offset)
        feed(audio)
        node.play()
    }

    func feed(_ audio: [Float]) {
        guard var timeline, audio.count > timeline.audioEnd else { return }
        let start = timeline.audioEnd
        let length = audio.count - start
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(length)) else { return }
        buffer.frameLength = AVAudioFrameCount(length)
        audio.withUnsafeBufferPointer { source in
            buffer.floatChannelData![0].update(from: source.baseAddress! + start, count: length)
        }
        let earliest = node.isPlaying ? (nodeTime ?? 0) + Self.schedulingLead : 0
        let when = timeline.append(length: length, earliest: earliest)
        self.timeline = timeline
        let end = timeline.audioEnd
        let epoch = self.epoch
        node.scheduleBuffer(buffer, at: AVAudioTime(sampleTime: when, atRate: format.sampleRate), options: [], completionCallbackType: .dataPlayedBack) { [weak self] _ in
            Task { @MainActor in self?.bufferPlayed(end: end, epoch: epoch) }
        }
    }

    @discardableResult
    func stop() -> Int? {
        let reached = position
        epoch += 1
        timeline = nil
        node.stop()
        return reached
    }

    private var nodeTime: Int64? {
        guard let render = node.lastRenderTime, let time = node.playerTime(forNodeTime: render) else { return nil }
        return time.sampleTime
    }

    private func bufferPlayed(end: Int, epoch: Int) {
        guard epoch == self.epoch, end == timeline?.audioEnd else { return }
        onDrained?()
    }
}

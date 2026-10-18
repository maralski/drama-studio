import AVFoundation

/// Plays the studio's 16-bit mono 44.1 kHz PCM as it arrives. The player node
/// takes float buffers at the stream's rate and the mixer converts them to the
/// output device's format.
enum Speakers {
    static let format = AVAudioFormat(standardFormatWithSampleRate: 44100, channels: 1)!

    static func play(_ chunks: AsyncThrowingStream<Data, any Error>) async throws {
        let engine = AVAudioEngine()
        let player = AVAudioPlayerNode()
        engine.attach(player)
        engine.connect(player, to: engine.mainMixerNode, format: format)
        try engine.start()
        defer { engine.stop() }
        player.play()
        for try await chunk in chunks {
            schedule(chunk, on: player)
        }
        // Buffers play in order, so this one finishing means everything before it drained.
        let tail = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 441)!
        tail.frameLength = tail.frameCapacity
        await player.scheduleBuffer(tail, completionCallbackType: .dataPlayedBack)
    }

    private static func schedule(_ pcm: Data, on player: AVAudioPlayerNode) {
        let count = pcm.count / 2
        guard count > 0, let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(count)) else { return }
        buffer.frameLength = AVAudioFrameCount(count)
        let out = buffer.floatChannelData![0]
        pcm.withUnsafeBytes { raw in
            for index in 0..<count {
                out[index] = Float(Int16(littleEndian: raw.loadUnaligned(fromByteOffset: index * 2, as: Int16.self))) / 32768
            }
        }
        player.scheduleBuffer(buffer, completionHandler: nil)
    }
}

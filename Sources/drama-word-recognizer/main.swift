// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import AVFoundation
import CryptoKit
import Darwin
import Foundation
import WordRecognitionProtocol
import whisper

@main struct WordRecognizer {
    static func main() {
        do { try run() } catch { exit(1) }
    }

    private static func run() throws {
        guard CommandLine.arguments.count == 4, let parent = Int32(CommandLine.arguments[3]), parent > 1 else {
            throw RecognitionChannelError.invalidFrame
        }
        let language = CommandLine.arguments[1]
        let asset = WordRecognitionAssets.model(for: language)
        guard whisper_lang_id(language) >= 0 else { throw RecognitionChannelError.invalidFrame }
        // This monitor remains runnable while the C recognizer is occupied.
        let parentWatch = DispatchSource.makeProcessSource(identifier: parent, eventMask: .exit, queue: .global())
        parentWatch.setEventHandler { _exit(1) }
        parentWatch.resume()
        defer { parentWatch.cancel() }
        guard getppid() == parent else { throw RecognitionChannelError.closed }
        let channel = try RecognitionChannel(descriptor: STDIN_FILENO)
        whisper_log_set({ _, _, _ in }, nil)
        let context = try loadModel(URL(fileURLWithPath: CommandLine.arguments[2]), asset: asset)
        defer { whisper_free(context) }
        let supported = asset.englishOnly ? ["en"] : (0...whisper_lang_max_id()).compactMap { id in
            whisper_lang_str(id).map { String(cString: $0) }
        }
        try channel.send(RecognitionReply.ready(.init(model: asset, languages: supported)), deadline: .now + .seconds(30), shouldContinue: { true })
        while true {
            let request = try channel.receive(RecognitionRequest.self, deadline: .now + .seconds(86_400), shouldContinue: { true })
            guard request.language == language, !request.pcm16.isEmpty, request.pcm16.count.isMultiple(of: 2),
                  request.pcm16.count <= 32 * 1024 * 1024 else { throw RecognitionChannelError.invalidFrame }
            do {
                let text = try transcribe(request.pcm16, language: language, context: context)
                try channel.send(RecognitionReply.transcript(id: request.id, text: text), deadline: .now + .seconds(30), shouldContinue: { true })
            } catch {
                try channel.send(RecognitionReply.failure(id: request.id, message: "independent recognition failed"), deadline: .now + .seconds(5), shouldContinue: { true })
                throw error
            }
        }
    }

    private static func loadModel(_ url: URL, asset: RecognitionModelAsset) throws -> OpaquePointer {
        // Data owns a snapshot, without mmap. Hash exactly the bytes consumed by
        // the synchronous C buffer loader, which copies them into its tensors.
        var data = try Data(contentsOf: url, options: [])
        guard data.count == asset.bytes,
              SHA256.hash(data: data).map({ String(format: "%02x", $0) }).joined() == asset.sha256 else {
            throw RecognitionChannelError.invalidFrame
        }
        var params = whisper_context_default_params()
        params.use_gpu = false
        let context = data.withUnsafeMutableBytes { buffer in
            whisper_init_from_buffer_with_params(buffer.baseAddress, buffer.count, params)
        }
        guard let context else { throw RecognitionChannelError.invalidFrame }
        return context
    }

    private static func transcribe(_ pcm: Data, language: String, context: OpaquePointer) throws -> String {
        let frames = pcm.count / 2
        let sourceFormat = AVAudioFormat(standardFormatWithSampleRate: 44_100, channels: 1)!
        let source = AVAudioPCMBuffer(pcmFormat: sourceFormat, frameCapacity: AVAudioFrameCount(frames))!
        source.frameLength = AVAudioFrameCount(frames)
        pcm.withUnsafeBytes { (bytes: UnsafeRawBufferPointer) in
            for index in 0..<frames {
                let raw = UInt16(bytes[index * 2]) | UInt16(bytes[index * 2 + 1]) << 8
                source.floatChannelData![0][index] = Float(Int16(bitPattern: raw)) / 32768
            }
        }
        let format = AVAudioFormat(standardFormatWithSampleRate: 16_000, channels: 1)!
        let converter = AVAudioConverter(from: sourceFormat, to: format)!
        let output = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(Double(frames) * 16_000 / 44_100) + 1024)!
        var fed = false
        var conversionError: NSError?
        converter.convert(to: output, error: &conversionError) { _, status in
            if fed { status.pointee = .endOfStream; return nil }
            fed = true; status.pointee = .haveData; return source
        }
        if let conversionError { throw conversionError }
        guard output.frameLength > 0 else { throw RecognitionChannelError.invalidFrame }
        var params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY)
        params.n_threads = 4
        params.no_context = true
        params.print_progress = false; params.print_realtime = false
        params.print_timestamps = false; params.print_special = false
        params.temperature = 0; params.temperature_inc = 0; params.greedy.best_of = 1
        let result = language.withCString { pointer -> Int32 in
            params.language = pointer
            return whisper_full(context, params, output.floatChannelData![0], Int32(output.frameLength))
        }
        guard result == 0 else { throw RecognitionChannelError.invalidFrame }
        var text = ""
        for segment in 0..<whisper_full_n_segments(context) {
            guard let pointer = whisper_full_get_segment_text(context, segment) else { throw RecognitionChannelError.invalidFrame }
            text += String(cString: pointer)
        }
        return text.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Darwin
import Foundation
package import WordRecognitionProtocol

/// Each session owns one child, one model and one serial request stream. C
/// recognition never runs inside the app, so cancellation can reclaim it.
package final class IndependentWordRecognizer: @unchecked Sendable {
    private static let capacity = DispatchSemaphore(value: 4)
    private let pid: pid_t
    private let channel: RecognitionChannel
    private let language: String
    private let lock = NSLock()
    private var retired = false
    package private(set) var identity: RecognitionReady?

    /// Where the helper and the language's model must be: beside the
    /// executable, or beside the engine resource bundle in embedded hosts such
    /// as XCTest. Installed apps keep the model in Contents/Resources.
    package static func runtimeFiles(language: String, runtimeDirectory: URL? = nil) -> (helper: URL, model: URL) {
        var executableDirectory = URL(fileURLWithPath: Bundle.main.executablePath ?? CommandLine.arguments[0]).deletingLastPathComponent()
        if let runtimeDirectory {
            executableDirectory = runtimeDirectory
        } else if Bundle.main.bundleURL.pathExtension != "app",
                  !FileManager.default.isExecutableFile(atPath: executableDirectory.appendingPathComponent(WordRecognitionAssets.helperName).path) {
            executableDirectory = EngineResources.bundle().bundleURL.deletingLastPathComponent()
        }
        let helper = executableDirectory.appendingPathComponent(WordRecognitionAssets.helperName)
        let resourceDirectory = Bundle.main.bundleURL.pathExtension == "app" ? Bundle.main.resourceURL! : executableDirectory
        let model = resourceDirectory.appendingPathComponent(WordRecognitionAssets.modelDirectory)
            .appendingPathComponent(WordRecognitionAssets.model(for: language).fileName)
        return (helper, model)
    }

    package init(language: String, deadline: ContinuousClock.Instant, shouldContinue: () -> Bool, runtimeDirectory: URL? = nil) throws {
        while Self.capacity.wait(timeout: .now() + .milliseconds(50)) == .timedOut {
            guard shouldContinue() else { throw CancellationError() }
            guard .now < deadline else { throw SpeechVerificationError.timedOut }
        }
        var releaseReservation = true
        defer { if releaseReservation { Self.capacity.signal() } }
        guard shouldContinue() else { throw CancellationError() }
        let (helper, model) = Self.runtimeFiles(language: language, runtimeDirectory: runtimeDirectory)
        guard FileManager.default.isExecutableFile(atPath: helper.path), FileManager.default.fileExists(atPath: model.path) else {
            throw SpeechVerificationError.unavailable("the bundled independent recognition helper or model is missing; rebuild with tools/make_app.sh")
        }
        var descriptors: [Int32] = [-1, -1]
        guard socketpair(AF_UNIX, SOCK_STREAM, 0, &descriptors) == 0 else {
            throw SpeechVerificationError.unavailable("cannot create the private recognition channel")
        }
        defer { close(descriptors[1]) }
        let parentChannel = try RecognitionChannel(descriptor: descriptors[0])
        var actions: posix_spawn_file_actions_t?
        var attributes: posix_spawnattr_t?
        guard posix_spawn_file_actions_init(&actions) == 0 else {
            throw SpeechVerificationError.unavailable("cannot prepare the recognition helper's descriptors")
        }
        defer { posix_spawn_file_actions_destroy(&actions) }
        guard posix_spawnattr_init(&attributes) == 0 else {
            throw SpeechVerificationError.unavailable("cannot prepare the recognition helper's launch")
        }
        defer { posix_spawnattr_destroy(&attributes) }
        guard posix_spawnattr_setflags(&attributes, Int16(POSIX_SPAWN_CLOEXEC_DEFAULT)) == 0,
              posix_spawn_file_actions_adddup2(&actions, descriptors[1], STDIN_FILENO) == 0,
              posix_spawn_file_actions_addopen(&actions, STDOUT_FILENO, "/dev/null", O_WRONLY, 0) == 0,
              posix_spawn_file_actions_addopen(&actions, STDERR_FILENO, "/dev/null", O_WRONLY, 0) == 0 else {
            throw SpeechVerificationError.unavailable("cannot isolate the recognition helper's descriptors")
        }
        let strings: [UnsafeMutablePointer<CChar>] = [helper.path, language, model.path, String(getpid())].map { value in
            value.withCString { Darwin.strdup($0)! }
        }
        defer { strings.forEach { free($0) } }
        var arguments: [UnsafeMutablePointer<CChar>?] = strings.map { $0 } + [nil]
        var child: pid_t = 0
        var environment: [UnsafeMutablePointer<CChar>?] = [nil]
        let status = posix_spawn(&child, helper.path, &actions, &attributes, &arguments, &environment)
        guard status == 0 else { throw SpeechVerificationError.unavailable("cannot launch the independent recognizer (\(status))") }
        pid = child; channel = parentChannel; self.language = language
        // Reservation transfers to the exclusively owned waitpid reaper.
        releaseReservation = false
        do {
            let reply = try channel.receive(RecognitionReply.self, deadline: deadline, maximumBytes: 256 * 1024, shouldContinue: shouldContinue)
            let asset = WordRecognitionAssets.model(for: language)
            guard case .ready(let ready) = reply,
                  ready.protocolVersion == WordRecognitionAssets.protocolVersion,
                  ready.configurationID == WordRecognitionAssets.configurationID,
                  ready.modelID == asset.id, ready.modelSHA256 == asset.sha256,
                  ready.languages.contains(language) else {
                throw SpeechVerificationError.unavailable("the independent recognizer did not identify the expected model and language")
            }
            guard shouldContinue() else { throw CancellationError() }
            guard ContinuousClock.now < deadline else { throw SpeechVerificationError.timedOut }
            identity = ready
        } catch {
            retire()
            throw Self.translate(error)
        }
    }

    deinit { retire() }

    package func transcript(pcm16: Data, deadline: ContinuousClock.Instant, shouldContinue: () -> Bool) throws -> String {
        guard lock.try() else { throw SpeechVerificationError.unavailable("the independent recognizer is already checking audio") }
        defer { lock.unlock() }
        guard !retired else { throw SpeechVerificationError.unavailable("the independent recognition session has stopped") }
        do {
            let id = UUID()
            try channel.send(RecognitionRequest(id: id, language: language, pcm16: pcm16), deadline: deadline, shouldContinue: shouldContinue)
            let reply = try channel.receive(RecognitionReply.self, deadline: deadline, maximumBytes: 256 * 1024, shouldContinue: shouldContinue)
            guard case .transcript(let receivedID, let text) = reply, receivedID == id, text.utf8.count <= 256 * 1024 else {
                throw SpeechVerificationError.unavailable("the independent recognizer returned an invalid response")
            }
            guard shouldContinue() else { throw CancellationError() }
            guard ContinuousClock.now < deadline else { throw SpeechVerificationError.timedOut }
            return text
        } catch {
            retireLocked()
            throw Self.translate(error)
        }
    }
    package func cancel() { retire() }

    private static func translate(_ error: Error) -> Error {
        if error is CancellationError { return error }
        if case RecognitionChannelError.timedOut = error { return SpeechVerificationError.timedOut }
        if error is SpeechVerificationError { return error }
        return SpeechVerificationError.unavailable("the independent recognizer stopped or returned an invalid response")
    }
    private func retire() {
        lock.lock(); defer { lock.unlock() }
        retireLocked()
    }
    private func retireLocked() {
        guard !retired else { return }
        retired = true
        channel.shutdown()
        let child = pid
        // Only this reaper calls waitpid. Until it observes exit, the PID cannot
        // be reused. Capacity includes retiring children, including hung C calls.
        DispatchQueue.global().async {
            kill(child, SIGTERM)
            let grace = ContinuousClock.now + .milliseconds(200)
            var status: Int32 = 0
            while ContinuousClock.now < grace {
                let result = waitpid(child, &status, WNOHANG)
                if result == child || (result == -1 && errno == ECHILD) { Self.capacity.signal(); return }
                if result == -1 && errno != EINTR { break }
                usleep(10_000)
            }
            kill(child, SIGKILL)
            while waitpid(child, &status, 0) == -1 && errno == EINTR {}
            Self.capacity.signal()
        }
    }
}

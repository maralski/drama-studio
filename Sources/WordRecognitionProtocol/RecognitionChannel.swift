// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Darwin
import Foundation

package enum RecognitionChannelError: Error {
    case closed, invalidFrame, timedOut, system(Int32)
}

/// Private inherited socket, with bounded frames and cancellation during I/O.
/// Shutdown wakes blocked operations; the owner closes only after users release it.
package final class RecognitionChannel: @unchecked Sendable {
    package let descriptor: Int32
    package init(descriptor: Int32) throws {
        self.descriptor = descriptor
        var enabled: Int32 = 1
        guard setsockopt(descriptor, SOL_SOCKET, SO_NOSIGPIPE, &enabled, socklen_t(MemoryLayout<Int32>.size)) == 0,
              fcntl(descriptor, F_SETFL, O_NONBLOCK) == 0 else {
            close(descriptor)
            throw RecognitionChannelError.system(errno)
        }
    }
    deinit { close(descriptor) }
    package func shutdown() { _ = Darwin.shutdown(descriptor, SHUT_RDWR) }

    package func send<T: Encodable>(_ value: T, deadline: ContinuousClock.Instant, shouldContinue: () -> Bool) throws {
        let body = try JSONEncoder().encode(value)
        guard body.count > 0, body.count <= Self.maximumFrame else { throw RecognitionChannelError.invalidFrame }
        var length = UInt32(body.count).bigEndian
        let header = withUnsafeBytes(of: &length) { Data($0) }
        try write(header, deadline: deadline, shouldContinue: shouldContinue)
        try write(body, deadline: deadline, shouldContinue: shouldContinue)
    }

    package func receive<T: Decodable>(_ type: T.Type, deadline: ContinuousClock.Instant, maximumBytes: Int = 64 * 1024 * 1024, shouldContinue: () -> Bool) throws -> T {
        let header = try read(count: 4, deadline: deadline, shouldContinue: shouldContinue)
        let length = header.reduce(UInt32(0)) { ($0 << 8) | UInt32($1) }
        guard length > 0, length <= Self.maximumFrame, length <= maximumBytes else { throw RecognitionChannelError.invalidFrame }
        return try JSONDecoder().decode(type, from: read(count: Int(length), deadline: deadline, shouldContinue: shouldContinue))
    }

    private static let maximumFrame = 64 * 1024 * 1024
    private func ready(_ events: Int16, deadline: ContinuousClock.Instant, shouldContinue: () -> Bool) throws {
        while true {
            guard shouldContinue() else { throw CancellationError() }
            guard ContinuousClock.now < deadline else { throw RecognitionChannelError.timedOut }
            var item = pollfd(fd: descriptor, events: events, revents: 0)
            let result = poll(&item, 1, 50)
            if result < 0 { if errno == EINTR { continue }; throw RecognitionChannelError.system(errno) }
            if result > 0 {
                if item.revents & events != 0 { return }
                throw RecognitionChannelError.closed
            }
        }
    }
    private func write(_ data: Data, deadline: ContinuousClock.Instant, shouldContinue: () -> Bool) throws {
        try data.withUnsafeBytes { buffer in
            var offset = 0
            while offset < buffer.count {
                try ready(Int16(POLLOUT), deadline: deadline, shouldContinue: shouldContinue)
                let count = Darwin.write(descriptor, buffer.baseAddress!.advanced(by: offset), buffer.count - offset)
                if count < 0, errno == EINTR || errno == EAGAIN { continue }
                guard count > 0 else { throw RecognitionChannelError.closed }
                offset += count
            }
        }
    }
    private func read(count: Int, deadline: ContinuousClock.Instant, shouldContinue: () -> Bool) throws -> Data {
        var data = Data(count: count)
        try data.withUnsafeMutableBytes { buffer in
            var offset = 0
            while offset < count {
                try ready(Int16(POLLIN), deadline: deadline, shouldContinue: shouldContinue)
                let received = Darwin.read(descriptor, buffer.baseAddress!.advanced(by: offset), count - offset)
                if received < 0, errno == EINTR || errno == EAGAIN { continue }
                guard received > 0 else { throw RecognitionChannelError.closed }
                offset += received
            }
        }
        return data
    }
}

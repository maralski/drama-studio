import Foundation
import Network

public enum ServerState: Sendable, Equatable {
    case stopped
    case starting
    case listening(port: UInt16)
    case failed(String)

    public var port: UInt16? {
        if case .listening(let port) = self { return port }
        return nil
    }
}

public final class APIServer: @unchecked Sendable {
    public let service: APIService
    public let requestedPort: UInt16
    private let onState: @Sendable (ServerState) -> Void
    private let queue = DispatchQueue(label: "DramaServer.listener")
    private var listener: NWListener?
    private var whenStopped: (@Sendable () -> Void)?

    /// Port 0 asks the system for a free port.
    public init(service: APIService, port: UInt16 = APIService.defaultPort, onState: @escaping @Sendable (ServerState) -> Void = { _ in }) {
        self.service = service
        self.requestedPort = port
        self.onState = onState
    }

    public func start() {
        queue.async { self.startOnQueue() }
    }

    /// `stopped` runs once the port is free, so a restart on the same port can bind.
    public func stop(stopped: (@Sendable () -> Void)? = nil) {
        queue.async {
            guard let listener = self.listener else {
                stopped?()
                return
            }
            self.whenStopped = stopped
            self.listener = nil
            listener.cancel()
        }
    }

    private func startOnQueue() {
        guard listener == nil else { return }
        onState(.starting)
        let parameters = NWParameters.tcp
        parameters.allowLocalEndpointReuse = true
        parameters.requiredLocalEndpoint = .hostPort(host: "127.0.0.1", port: NWEndpoint.Port(rawValue: requestedPort) ?? .any)
        let listener: NWListener
        do {
            listener = try NWListener(using: parameters)
        } catch {
            onState(.failed(Self.describe(error, port: requestedPort)))
            return
        }
        var failed = false
        listener.stateUpdateHandler = { [weak self, weak listener] state in
            guard let self else { return }
            switch state {
            case .ready:
                self.onState(.listening(port: listener?.port?.rawValue ?? self.requestedPort))
            case .failed(let error), .waiting(let error):
                failed = true
                self.onState(.failed(Self.describe(error, port: self.requestedPort)))
                listener?.cancel()
                if self.listener === listener { self.listener = nil }
            case .cancelled:
                if !failed { self.onState(.stopped) }
                self.whenStopped?()
                self.whenStopped = nil
            default:
                break
            }
        }
        listener.newConnectionHandler = { [weak self] connection in self?.accept(connection) }
        self.listener = listener
        listener.start(queue: queue)
    }

    private static func describe(_ error: Error, port: UInt16) -> String {
        if case NWError.posix(.EADDRINUSE) = error { return "Port \(port) is already in use." }
        return error.localizedDescription
    }

    /// Lives on the listener's queue for one connection. A class, so each
    /// read appends in place instead of copying everything read so far.
    private final class Pending {
        var bytes = Data()
        var head: HTTPHead?
        var sentContinue = false
        var body: StreamingBody?
    }

    /// A request body the service reads while it answers.
    private final class StreamingBody {
        var framing: IncrementalBody
        let continuation: AsyncThrowingStream<Data, Error>.Continuation
        let work: Task<Void, Never>

        init(framing: IncrementalBody, continuation: AsyncThrowingStream<Data, Error>.Continuation, work: Task<Void, Never>) {
            self.framing = framing
            self.continuation = continuation
            self.work = work
        }

        func feed(_ data: Data) {
            guard !framing.isFinished else { return }
            do {
                for payload in try framing.feed(data) { continuation.yield(payload) }
                if framing.isFinished { continuation.finish() }
            } catch {
                continuation.finish(throwing: error)
            }
        }
    }

    private func accept(_ connection: NWConnection) {
        connection.start(queue: queue)
        receive(on: connection, into: Pending())
    }

    private func receive(on connection: NWConnection, into pending: Pending) {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 1 << 20) { [weak self] data, _, isComplete, error in
            guard let self else { return connection.cancel() }
            if let body = pending.body {
                if let data { body.feed(data) }
                guard error == nil, !isComplete else {
                    // A client may half-close to end its text; after the body's end, a close is a hang-up.
                    if body.framing.isFinished { body.work.cancel() } else { body.continuation.finish() }
                    return
                }
                return self.receive(on: connection, into: pending)
            }
            if let data { pending.bytes.append(data) }
            if pending.head == nil, case .head(let head) = HTTPRequestParser.parseHead(pending.bytes), APIRequest.streamsBody(head.request) {
                return self.stream(head, on: connection, into: pending, ended: isComplete || error != nil)
            }
            if let head = pending.head, pending.bytes.count < head.totalLength {
                guard error == nil, !isComplete else { return connection.cancel() }
                return self.receive(on: connection, into: pending)
            }
            switch HTTPRequestParser.parse(pending.bytes) {
            case .complete(let request):
                pending.bytes = Data()
                let work = Task { await self.send(await self.service.respond(to: request), on: connection) }
                self.watchForHangUp(on: connection) { work.cancel() }
            case .invalid(let status, let message):
                Task { await self.send(.error(APIError(status, message)), on: connection) }
            case .incomplete(let head):
                guard error == nil, !isComplete else { return connection.cancel() }
                pending.head = head
                if let head, head.expectsContinue, !pending.sentContinue {
                    pending.sentContinue = true
                    connection.send(content: Data("HTTP/1.1 100 Continue\r\n\r\n".utf8), completion: .idempotent)
                }
                self.receive(on: connection, into: pending)
            }
        }
    }

    private func stream(_ head: ParsedHead, on connection: NWConnection, into pending: Pending, ended: Bool) {
        if head.expectsContinue { connection.send(content: Data("HTTP/1.1 100 Continue\r\n\r\n".utf8), completion: .idempotent) }
        let (stream, continuation) = AsyncThrowingStream<Data, Error>.makeStream()
        let work = Task { await self.send(await self.service.respond(to: head.request, body: stream), on: connection) }
        let body = StreamingBody(framing: IncrementalBody(head.framing), continuation: continuation, work: work)
        pending.body = body
        body.feed(Data(pending.bytes.dropFirst(head.bodyStart)))
        pending.bytes = Data()
        if ended {
            continuation.finish()
        } else {
            receive(on: connection, into: pending)
        }
    }

    /// A client sends nothing after its request, so a read that ends means it
    /// hung up; cancelling the request's task stops a generation nobody will hear.
    private func watchForHangUp(on connection: NWConnection, _ hangUp: @escaping @Sendable () -> Void) {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 1) { _, _, isComplete, error in
            if isComplete || error != nil { hangUp() }
        }
    }

    private func send(_ response: APIResponse, on connection: NWConnection) async {
        defer { connection.cancel() }
        switch response.body {
        case .data(let body):
            _ = await write(response.head + body, on: connection, final: true)
        case .chunked(let stream):
            guard await write(response.head, on: connection, final: false) else { return response.abandon() }
            do {
                for try await chunk in stream {
                    guard await write(ChunkedEncoding.frame(chunk), on: connection, final: false) else { return response.abandon() }
                }
            } catch {
                return
            }
            _ = await write(ChunkedEncoding.terminator, on: connection, final: true)
        }
    }

    private func write(_ data: Data, on connection: NWConnection, final: Bool) async -> Bool {
        await withCheckedContinuation { continuation in
            connection.send(content: data, contentContext: final ? .finalMessage : .defaultMessage, isComplete: true, completion: .contentProcessed { error in
                continuation.resume(returning: error == nil)
            })
        }
    }
}

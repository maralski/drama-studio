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

    /// Port 0 asks the system for a free port.
    public init(service: APIService, port: UInt16 = APIService.defaultPort, onState: @escaping @Sendable (ServerState) -> Void = { _ in }) {
        self.service = service
        self.requestedPort = port
        self.onState = onState
    }

    public func start() {
        queue.async { self.startOnQueue() }
    }

    public func stop() {
        queue.async {
            self.listener?.cancel()
            self.listener = nil
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

    private func accept(_ connection: NWConnection) {
        connection.start(queue: queue)
        receive(on: connection, buffer: Data(), head: nil, sentContinue: false)
    }

    private func receive(on connection: NWConnection, buffer: Data, head: HTTPHead?, sentContinue: Bool) {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 1 << 20) { [weak self] data, _, isComplete, error in
            guard let self else { return connection.cancel() }
            var buffer = buffer
            if let data { buffer.append(data) }
            if let head, buffer.count < head.totalLength {
                guard error == nil, !isComplete else { return connection.cancel() }
                return self.receive(on: connection, buffer: buffer, head: head, sentContinue: sentContinue)
            }
            switch HTTPRequestParser.parse(buffer) {
            case .complete(let request):
                let work = Task { await self.send(await self.service.respond(to: request), on: connection) }
                self.watchForHangUp(on: connection) { work.cancel() }
            case .invalid(let status, let message):
                Task { await self.send(.error(APIError(status, message)), on: connection) }
            case .incomplete(let head):
                guard error == nil, !isComplete else { return connection.cancel() }
                if let head, head.expectsContinue, !sentContinue {
                    connection.send(content: Data("HTTP/1.1 100 Continue\r\n\r\n".utf8), completion: .idempotent)
                }
                self.receive(on: connection, buffer: buffer, head: head, sentContinue: sentContinue || head?.expectsContinue == true)
            }
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

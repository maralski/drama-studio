// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation

public protocol MCPTransport: Sendable {
    /// Nil at end of input.
    func readLine() async -> String?
    func write(_ line: String) async
}

public struct MCPServer: Sendable {
    public static let protocolVersion = "2025-06-18"
    public static let version = "0.1"

    public let tools: [MCPTool]
    public let context: ToolContext
    private let toolQueue = SerialQueue()

    public init(context: ToolContext, tools: [MCPTool] = MCPTool.all) {
        self.context = context
        self.tools = tools
    }

    enum RPCError {
        static let parse = -32700
        static let invalidRequest = -32600
        static let methodNotFound = -32601
        static let invalidParams = -32602
    }

    public func handle(_ line: String) async -> String? {
        guard let message = try? JSON(parsing: Data(line.utf8)) else {
            return Self.error(id: .null, code: RPCError.parse, message: "Parse error: the message is not JSON.")
        }
        guard case .object(let fields) = message, fields["jsonrpc"] == "2.0", let method = fields["method"]?.string else {
            return Self.error(id: message["id"] ?? .null, code: RPCError.invalidRequest, message: "Invalid request: expected a JSON-RPC 2.0 object with a method.")
        }
        guard let id = fields["id"] else { return nil }
        let params = fields["params"]?.object ?? [:]
        switch method {
        case "initialize":
            return Self.result(id: id, [
                "protocolVersion": .string(Self.protocolVersion),
                "capabilities": ["tools": ["listChanged": false]],
                "serverInfo": ["name": "drama-studio", "title": "Drama Studio", "version": .string(Self.version)],
                "instructions": "Drama Studio speaks text aloud with Fish Audio Drama 3 on this Mac. Call studio_status to see whether the model is loaded, list_voices to pick a voice, then synthesize_speech. One generation runs at a time.",
            ])
        case "ping":
            return Self.result(id: id, [:])
        case "tools/list":
            return Self.result(id: id, ["tools": .array(tools.map(\.listing))])
        case "tools/call":
            guard let name = params["name"]?.string else {
                return Self.error(id: id, code: RPCError.invalidParams, message: "tools/call needs a tool name.")
            }
            guard let tool = tools.first(where: { $0.name == name }) else {
                return Self.error(id: id, code: RPCError.invalidParams, message: "Unknown tool: \(name).")
            }
            return Self.result(id: id, await call(tool, arguments: params["arguments"]?.object ?? [:]).json)
        default:
            return Self.error(id: id, code: RPCError.methodNotFound, message: "Method not found: \(method).")
        }
    }

    /// Tool calls take turns because the studio generates one request at a
    /// time; running them side by side would only earn this client a 429.
    func call(_ tool: MCPTool, arguments: [String: JSON]) async -> ToolResult {
        await toolQueue.run { [context] in
            do {
                return .success(try await tool.run(arguments, context))
            } catch let error as ToolInputError {
                return .failure(error.message)
            } catch {
                return .failure((error as? LocalizedError)?.errorDescription ?? String(describing: error))
            }
        }
    }

    /// Reads until end of input. Messages are handled concurrently so a ping
    /// is not stuck behind a long generation.
    public func run(_ transport: some MCPTransport) async {
        await withTaskGroup(of: Void.self) { group in
            while let line = await transport.readLine() {
                guard !line.trimmingCharacters(in: .whitespaces).isEmpty else { continue }
                group.addTask {
                    if let reply = await handle(line) { await transport.write(reply) }
                }
            }
        }
    }

    static func result(id: JSON, _ result: JSON) -> String {
        JSON.object(["jsonrpc": "2.0", "id": id, "result": result]).serialized
    }

    static func error(id: JSON, code: Int, message: String) -> String {
        JSON.object(["jsonrpc": "2.0", "id": id, "error": ["code": .int(code), "message": .string(message)]]).serialized
    }
}

actor SerialQueue {
    private var tail: Task<Void, Never>?

    func run<T: Sendable>(_ operation: @escaping @Sendable () async -> T) async -> T {
        let previous = tail
        let task = Task {
            await previous?.value
            return await operation()
        }
        tail = Task { _ = await task.value }
        return await task.value
    }
}

/// Logs go to stderr because stdout carries the protocol.
public final class StdioTransport: MCPTransport, @unchecked Sendable {
    private let writeLock = NSLock()

    public init() {}

    /// Blocks a dispatch thread rather than one of the few cooperative ones.
    public func readLine() async -> String? {
        await withCheckedContinuation { continuation in
            DispatchQueue.global().async { continuation.resume(returning: Swift.readLine(strippingNewline: true)) }
        }
    }

    /// Replies finish out of order, so whole lines are written under a lock.
    public func write(_ line: String) async {
        writeLock.withLock { FileHandle.standardOutput.write(Data((line + "\n").utf8)) }
    }
}

// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import DramaMCP

await MCPServer(context: ToolContext(api: HTTPDramaAPI())).run(StdioTransport())

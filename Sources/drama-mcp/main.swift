import DramaMCP

await MCPServer(context: ToolContext(api: HTTPDramaAPI())).run(StdioTransport())

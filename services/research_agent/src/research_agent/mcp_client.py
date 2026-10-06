"""Traced MCP client for the remote web-search server."""

import json
from typing import Any

from mcp.client import Client
from mcp.server.mcpserver import MCPServer

from fin_tracing import inject_carrier, tool_span


class WebSearchClient:
    """One MCP session per research run. Pass a URL, or an MCPServer for in-process tests."""

    def __init__(self, server: str | MCPServer, *, timeout_seconds: float = 30.0) -> None:
        self._client = Client(server, read_timeout_seconds=timeout_seconds)
        self._server_label = server if isinstance(server, str) else "in-process"
        self._tools: dict[str, Any] = {}

    async def __aenter__(self) -> "WebSearchClient":
        await self._client.__aenter__()
        listing = await self._client.list_tools()
        self._tools = {tool.name: tool for tool in listing.tools}
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self._client.__aexit__(*exc_info)

    def tool_definitions(self) -> list[dict[str, Any]]:
        """MCP tools in Anthropic tool format, so Claude sees exactly what the server offers."""
        return [
            {"name": t.name, "description": t.description or "", "input_schema": t.input_schema}
            for t in self._tools.values()
        ]

    async def call(self, name: str, arguments: dict[str, Any]) -> Any:
        tool = self._tools.get(name)
        if tool is None:
            raise ValueError(f"Unknown MCP tool: {name}")
        with tool_span(
            name,
            description=tool.description or "",
            parameters_schema=tool.input_schema,
            arguments=arguments,
        ) as span:
            span.set_attributes({"finagent.mcp.side": "client", "finagent.mcp.server": self._server_label})
            span.add_event("mcp.request_sent")
            # MCP has no per-call headers; traceparent + baggage travel in params._meta.
            result = await self._client.call_tool(name, arguments, meta=inject_carrier())
            span.add_event("mcp.response_received", {"is_error": bool(result.is_error)})
            text = "\n".join(getattr(item, "text", "") for item in result.content)
            if result.is_error:
                raise RuntimeError(f"MCP tool {name} failed: {text}")
            output = json.loads(text) if text.startswith(("[", "{")) else text
            span.set_output(output)
            return output

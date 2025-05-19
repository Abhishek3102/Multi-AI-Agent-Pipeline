from contextlib import AsyncExitStack
from typing import Any, Literal, Optional, TypedDict, cast
from langchain_core.tools import BaseTool
from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp_integration.mcp_prompt_loader import load_mcp_prompt
from mcp_integration.mcp_tool_loader import load_mcp_tools
from langchain_core.messages import HumanMessage, AIMessage

DEFAULT_ENCODING = "utf-8"
DEFAULT_ENCODING_ERROR_HANDLER = "strict"
DEFAULT_HTTP_TIMEOUT = 5
DEFAULT_SSE_READ_TIMEOUT = 300


class StdioConnection(TypedDict):
    transport: Literal["stdio"]
    command: str
    args: list[str]
    env: Optional[dict[str, str]]
    encoding: str
    encoding_error_handler: Literal["strict", "ignore", "replace"]


class SSEConnection(TypedDict):
    transport: Literal["sse"]
    url: str
    headers: Optional[dict[str, Any]]
    timeout: float
    sse_read_timeout: float


class MultiServerMCPClient:
    def __init__(self, connections: dict[str, StdioConnection | SSEConnection] = None) -> None:
        self.connections = connections or {}
        self.exit_stack = AsyncExitStack()
        self.sessions: dict[str, ClientSession] = {}
        self.server_tools: dict[str, list[BaseTool]] = {}

    async def _initialize_session_and_tools(self, server_name: str, session: ClientSession) -> None:
        await session.initialize()
        self.sessions[server_name] = session
        self.server_tools[server_name] = await load_mcp_tools(session)

    async def connect_to_server(self, server_name: str, transport: str, **kwargs) -> None:
        if transport == "sse":
            await self._connect_sse(server_name, **kwargs)
        elif transport == "stdio":
            await self._connect_stdio(server_name, **kwargs)
        else:
            raise ValueError("Unsupported transport. Use 'stdio' or 'sse'.")

    async def _connect_stdio(self, server_name: str, **kwargs) -> None:
        params = StdioServerParameters(
            command=kwargs["command"],
            args=kwargs["args"],
            env=kwargs.get("env"),
            encoding=kwargs.get("encoding", DEFAULT_ENCODING),
            encoding_error_handler=kwargs.get("encoding_error_handler", DEFAULT_ENCODING_ERROR_HANDLER),
        )
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(params))
        read, write = stdio_transport
        session = cast(ClientSession, await self.exit_stack.enter_async_context(ClientSession(read, write)))
        await self._initialize_session_and_tools(server_name, session)

    async def _connect_sse(self, server_name: str, **kwargs) -> None:
        sse_transport = await self.exit_stack.enter_async_context(
            sse_client(kwargs["url"], kwargs.get("headers"), kwargs.get("timeout", DEFAULT_HTTP_TIMEOUT), kwargs.get("sse_read_timeout", DEFAULT_SSE_READ_TIMEOUT))
        )
        read, write = sse_transport
        session = cast(ClientSession, await self.exit_stack.enter_async_context(ClientSession(read, write)))
        await self._initialize_session_and_tools(server_name, session)

    def get_all_tools(self) -> list[BaseTool]:
        all_tools = []
        for tools in self.server_tools.values():
            all_tools.extend(tools)
        return all_tools

    async def get_prompt(self, server_name: str, prompt_name: str, args: Optional[dict[str, Any]] = None) -> list[HumanMessage | AIMessage]:
        session = self.sessions[server_name]
        return await load_mcp_prompt(session, prompt_name, args)

    async def __aenter__(self):
        for name, config in self.connections.items():
            transport = config.pop("transport")
            await self.connect_to_server(name, transport=transport, **config)
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.exit_stack.aclose()

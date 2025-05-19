from typing import Any
from langchain_core.tools import BaseTool, StructuredTool, ToolException
from mcp import ClientSession
from mcp.types import CallToolResult, Tool as MCPTool, TextContent, ImageContent, EmbeddedResource

NonTextContent = ImageContent | EmbeddedResource

def parse_tool_result(result: CallToolResult) -> tuple[str | list[str], list[NonTextContent] | None]:
    text_parts = []
    non_text_parts = []
    for content in result.content:
        if isinstance(content, TextContent):
            text_parts.append(content.text)
        else:
            non_text_parts.append(content)

    content_output = text_parts if len(text_parts) > 1 else (text_parts[0] if text_parts else "")
    if result.isError:
        raise ToolException(content_output)
    return content_output, non_text_parts or None


def convert_tool(session: ClientSession, tool: MCPTool) -> BaseTool:
    async def execute_tool(**kwargs: dict[str, Any]) -> tuple[str | list[str], list[NonTextContent] | None]:
        result = await session.call_tool(tool.name, kwargs)
        return parse_tool_result(result)

    return StructuredTool(
        name=tool.name,
        description=tool.description or "",
        args_schema=tool.inputSchema,
        coroutine=execute_tool,
        response_format="content_and_artifact"
    )


async def load_mcp_tools(session: ClientSession) -> list[BaseTool]:
    tools_response = await session.list_tools()
    return [convert_tool(session, tool) for tool in tools_response.tools]

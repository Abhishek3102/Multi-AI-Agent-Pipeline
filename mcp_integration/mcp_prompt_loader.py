from typing import Any, Optional
from langchain_core.messages import HumanMessage, AIMessage
from mcp import ClientSession
from mcp.types import PromptMessage


def convert_to_langchain_msg(message: PromptMessage) -> HumanMessage | AIMessage:
    if message.content.type == "text":
        if message.role == "user":
            return HumanMessage(content=message.content.text)
        elif message.role == "assistant":
            return AIMessage(content=message.content.text)
        else:
            raise ValueError(f"Unknown role: {message.role}")
    raise ValueError(f"Unsupported content type: {message.content.type}")


async def load_mcp_prompt(session: ClientSession, name: str, arguments: Optional[dict[str, Any]] = None) -> list[HumanMessage | AIMessage]:
    response = await session.get_prompt(name, arguments)
    return [convert_to_langchain_msg(msg) for msg in response.messages]

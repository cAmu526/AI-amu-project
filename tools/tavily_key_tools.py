import asyncio
from typing import List
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from config.config import Config
from utils.logger import get_logger

logger = get_logger(__name__)

# 全局变量
_mcp_client: MultiServerMCPClient | None = None
_tools: List[BaseTool] | None = None
_init_lock = asyncio.Lock()


def _get_client() -> MultiServerMCPClient:
    """获取 MCP 客户端（同步，但不初始化）"""
    global _mcp_client
    if _mcp_client is None:
        _mcp_client = MultiServerMCPClient({
            "tavily-maps": {
                "url": "https://mcp.tavily.com/mcp/?tavilyApiKey=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",
                "transport": "streamable_http"
            }
        })
    return _mcp_client


async def get_tavily_tools() -> List[BaseTool]:
    global _tools
    if _tools is not None:
        return _tools

    async with _init_lock:
        # 双重检查（防止并发时重复初始化）
        if _tools is not None:
            return _tools

        client = _get_client()
        tools = await client.get_tools()
        logger.info(f"成功加载 {len(tools)} 个tavily MCP 工具: {[t.name for t in tools]}")
        _tools = tools
        return _tools


if __name__ == '__main__':
    async def main():
        tools1 = await get_tavily_tools()


    asyncio.run(main())

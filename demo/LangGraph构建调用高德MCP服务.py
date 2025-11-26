# amap_mcp_agent.py
import asyncio
from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from utils.llm import get_llm
from utils.logger import get_logger
from tools.amap_mcp_tools import get_amap_tools

logger = get_logger(__name__)


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


async def create_langgraph_app():
    tools = await get_amap_tools()
    tool_node = ToolNode(tools=tools)

    # 绑定工具到 LLM
    llm = get_llm()
    llm_with_tools = llm.bind_tools(tools)

    def call_model(state: AgentState):
        logger.info("call_model...")
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    def should_continue(state: AgentState) -> str:
        last_msg = state["messages"][-1]
        if hasattr(last_msg, "tool_calls") and len(last_msg.tool_calls) > 0:
            return "tools"
        return END

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    workflow.add_edge("tools", "agent")

    return workflow.compile()


async def main():
    app = await create_langgraph_app()

    # 示例问题（可替换）
    user_query = "帮我查一下深圳未来三天的天气，并给出穿衣建议。"
    print(f"\n👤 用户: {user_query}\n🤖 助手: ", end="", flush=True)

    inputs = {"messages": [HumanMessage(content=user_query)]}

    async for event in app.astream_events(inputs, version="v2"):
        kind = event["event"]
        logger.info("event kind: %s", kind)
        if kind == "on_chain_stream":
            chunk = event["data"].get("chunk", {})
            if isinstance(chunk, dict) and "messages" in chunk:
                # 取最新的一条消息
                last_msg = chunk["messages"][-1]
                # 只有当是 AI 生成的回答时才打印
                if isinstance(last_msg, AIMessage) and last_msg.content:
                    print(last_msg.content, end="", flush=True)

        elif kind == "on_tool_end":
            tool_name = event["name"]
            logger.info("tool_name: %s", tool_name)

            output = event["data"].get("output", "")
            content_str = output.content
            logger.info("output: %s", content_str)


async def test():
    app = await create_langgraph_app()
    user_query = "帮我查一下深圳龙华未来三天的天气，并给出穿衣建议。"
    print(f"\n👤 用户: {user_query}\n")
    inputs = {"messages": [HumanMessage(content=user_query)]}

    result = await app.ainvoke(inputs)
    print("\n🤖 最终回答:")
    print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
    # asyncio.run(test())

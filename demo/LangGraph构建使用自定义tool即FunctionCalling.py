from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from utils.llm import get_llm
from tools.custom_tools import TOOLS, TOOL_MAP

tools = TOOLS
tool_map = TOOL_MAP


# 2. 定义状态
class State(TypedDict):
    messages: Annotated[list, add_messages]


# 3. 初始化模型并绑定工具
llm = get_llm()
llm_with_tools = llm.bind_tools(tools)


# 4. 定义节点函数
def call_model(state: State):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def call_tool(state: State):
    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        raise ValueError("没有工具调用")

    tool_calls = last_message.tool_calls
    results = []
    for call in tool_calls:
        tool_name = call["name"]
        tool_args = call["args"]
        tool_result = tool_map[tool_name].invoke(tool_args)
        results.append(
            ToolMessage(
                content=str(tool_result),
                tool_call_id=call["id"]
            )
        )
    return {"messages": results}


# 5. 定义条件边：是否需要调用工具？
def should_continue(state: State) -> Literal["call_tool", "__end__"]:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0:
        return "call_tool"
    return "__end__"


# 6. 构建图
graph = StateGraph(State)
graph.add_node("call_model", call_model)
graph.add_node("call_tool", call_tool)

graph.add_edge(START, "call_model")
graph.add_conditional_edges("call_model", should_continue)
graph.add_edge("call_tool", "call_model")

app = graph.compile()

# 7. 运行示例
if __name__ == "__main__":
    # inputs = {"messages": [{"role": "user", "content": "北京的天气怎么样？"}]}
    # inputs = {"messages": [{"role": "user", "content": "120加上3等于多少？"}]}
    inputs = {"messages": [HumanMessage(content="北京的天气怎么样")]}
    for output in app.stream(inputs, stream_mode="values"):
        last_msg = output["messages"][-1]
        print(f"[{last_msg.type.upper()}]: {last_msg.content}")

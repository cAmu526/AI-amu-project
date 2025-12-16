from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, START, END, add_messages
from utils.llm import get_llm


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def call_llm(state: AgentState):
    messages = state["messages"]
    response = get_llm().invoke(messages)
    return {"messages": [response]}


def create_langgraph_app():
    graph_builder = StateGraph(AgentState)

    # 设置入口点
    graph_builder.set_entry_point("call_model_node")
    # 添加节点
    graph_builder.add_node("call_model_node", call_llm)

    graph_builder.add_edge("call_model_node", END)

    # 编译
    app = graph_builder.compile()

    return app


# 测试多轮对话
if __name__ == "__main__":
    app = create_langgraph_app()
    messages = []
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["退出", "exit", "quit"]:
            break
        messages.append(HumanMessage(content=user_input))

        inputs = {"messages": messages}
        for chunk in app.stream(inputs):
            if "call_model_node" in chunk:
                ai_msg = chunk["call_model_node"]["messages"][-1]
                print(f"Bot: {ai_msg.content}")
                messages.append(ai_msg)  # 保存 AI 回复，用于下一轮

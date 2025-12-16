import os
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END, add_messages
from utils.llm import get_llm
from utils.postgresql_connection import get_pgsql_checkpointer


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def call_llm(state: AgentState):
    messages = state["messages"]
    response = get_llm().invoke(messages)
    return {"messages": [response]}


def create_langgraph_app():
    graph_builder = StateGraph(AgentState)

    # 设置入口点
    graph_builder.set_entry_point("call_model_node")#老写法，LangGraph v1.0版本为 graph_builder.add_edge(START. "call_model_node")
    # 添加节点
    graph_builder.add_node("call_model_node", call_llm)

    graph_builder.add_edge("call_model_node", END)

    # 编译
    app = graph_builder.compile(checkpointer=get_pgsql_checkpointer())

    return app


# 测试多轮对话
if __name__ == "__main__":
    app = create_langgraph_app()
    config = {"configurable": {"thread_id": "user_test_1"}}  # 关键！必须有 thread_id
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["退出", "exit", "quit"]:
            break

        inputs = {"messages": [HumanMessage(content=user_input)]}

        for chunk in app.stream(inputs, config=config):
            if "call_model_node" in chunk:
                ai_msg = chunk["call_model_node"]["messages"][-1]
                print(f"Bot: {ai_msg.content}")

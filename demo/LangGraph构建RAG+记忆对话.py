import os
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END, add_messages
from utils.llm import get_llm
from utils.postgresql_connection import get_pgsql_checkpointer
from utils.rag.chroma_store import get_retriever
from langchain_core.documents import Document


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def retrieve(state: AgentState):
    """
    检索节点：从最后一条用户消息中提取查询，进行向量检索。
    """
    # 获取最新的用户输入（通常是最后一条 HumanMessage）
    query = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            query = msg.content
            break

    if not query:
        return {"messages": []}  # 无查询则跳过

    # 调用检索器（默认 collection，可按需传参）
    retriever = get_retriever()
    docs: list[Document] = retriever.invoke(query)

    # 将检索结果格式化为文本上下文
    context = "\n\n".join([doc.page_content for doc in docs])

    # 构造增强后的系统提示（或直接拼接到用户消息）
    augmented_user_message = HumanMessage(
        content=(
            f"请根据以下上下文回答问题。如果上下文不相关，请回答：我的知识库没有相关信息。\n\n"
            f"上下文：\n{context}\n\n"
            f"问题：{query}"
        )
    )

    # 替换原始用户消息为增强版（保留历史对话结构）
    new_messages = []
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage) and msg.content == query:
            new_messages.append(augmented_user_message)
        else:
            new_messages.append(msg)

    return {"messages": new_messages}


def call_llm(state: AgentState):
    messages = state["messages"]
    response = get_llm().invoke(messages)
    return {"messages": [response]}


def create_langgraph_app():
    graph_builder = StateGraph(AgentState)

    # 添加节点
    graph_builder.add_node('retrieve', retrieve)
    graph_builder.add_node("call_model_node", call_llm)

    graph_builder.add_edge(START, "retrieve")
    graph_builder.add_edge("retrieve", "call_model_node")
    graph_builder.add_edge("call_model_node", END)

    # 编译
    app = graph_builder.compile(checkpointer=get_pgsql_checkpointer())

    return app


# 测试多轮对话
if __name__ == "__main__":
    app = create_langgraph_app()
    config = {"configurable": {"thread_id": "user_test_1"}}  # 关键！必须有 thread_id
    while True:
        user_input = input("👤 You: ")
        if user_input.lower() in ["退出", "exit", "quit"]:
            break

        inputs = {"messages": [HumanMessage(content=user_input)]}

        for chunk in app.stream(inputs, config=config):
            if "call_model_node" in chunk:
                ai_msg = chunk["call_model_node"]["messages"][-1]
                print(f"🤖 Bot: {ai_msg.content}")

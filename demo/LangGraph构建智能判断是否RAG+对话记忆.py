import os
from typing import TypedDict, Annotated, Sequence, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END, add_messages
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser

from utils.llm import get_llm
from utils.postgresql_connection import get_pgsql_checkpointer
from utils.rag.chroma_store import get_retriever
from utils.logger import get_logger

logger = get_logger(__name__)


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


# === Step 1: 定义判断是否需要检索的结构化输出 ，相当于定义了一个数据结构===
class NeedRetrieve(BaseModel):
    need_retrieve: bool = Field(description="是否需要从知识库检索信息来回答问题。")
    reason: str = Field(description="判断理由（用于调试）")


# === Step 2: 判断节点：是否需要 RAG ===
def should_retrieve(state: AgentState) -> dict:
    """
    使用 LLM 判断当前问题是否需要检索外部知识。
    返回 "yes" 或 "no"，用于 LangGraph 条件路由。
    """
    # 获取最新用户消息
    query = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            query = msg.content
            break

    if not query:
        return {"route": "no"}

    # 构造判断 prompt 构建一个结构化的聊天提示模板（Chat Prompt Template），用于引导 LLM 进行检索必要性判断。
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个智能助手，负责判断用户的问题是否需要查询外部知识库（如文档、数据库、专业知识）才能准确回答。\n"
                   "仅当问题涉及事实性、特定领域知识、文档内容、历史记录等时才需要检索。\n"
                   "问候、闲聊、通用常识、指令类问题（如'总结一下'）通常不需要检索。"
                   "你必须返回一个 JSON 对象，格式如下：\n"
                   '{{"need_retrieve": true/false, "reason": "你的判断理由"}}'),
        ("human", "问题：{question}")
    ])

    # 使用轻量级 LLM（或主 LLM）做判断
    llm = get_llm()
    structured_llm = llm.with_structured_output(NeedRetrieve)
    # parser = JsonOutputParser(pydantic_object=NeedRetrieve)
    # LCEL表达式
    # chain = prompt | llm | parser
    chain = prompt | structured_llm

    try:
        logger.info("llm判断中...")
        result = chain.invoke({"question": query})
        need = result.need_retrieve
        reason = result.reason
        # need = result.get('need_retrieve')
        # reason = result.get('reason')
        logger.info(f"RAG 判断：{'需要' if need else '不需要'}检索 | 理由: {reason}")
        return {"route": "yes" if need else "no"}

    except Exception as e:
        logger.error(f"⚠️ RAG 判断出错，默认不检索: '{e}'")
        return {"route": "no"}


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
    logger.info("rag检索...")
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
    logger.info("call_llm回复最后结果...")
    response = get_llm().invoke(messages)
    return {"messages": [response]}


def create_langgraph_app():
    graph_builder = StateGraph(AgentState)

    # 添加节点
    graph_builder.add_node("judge_retrieve", should_retrieve)  # 判断节点（不修改 state）
    graph_builder.add_node('retrieve', retrieve)
    graph_builder.add_node("call_model_node", call_llm)

    graph_builder.add_edge(START, "judge_retrieve")
    # 条件边：根据 judge_retrieve 的返回值决定走向
    graph_builder.add_conditional_edges(
        "judge_retrieve",  # 用于判断的节点的结果
        lambda x: x['route'],  # 上一个节点执行的should_retrieve方法 其返回的是 "yes"/"no"，所以这里只用一个lambda表达式写的简单匿名函数作为路由函数
        {
            "yes": "retrieve",
            "no": "call_model_node"
        }
    )

    graph_builder.add_edge("retrieve", "call_model_node")
    graph_builder.add_edge("call_model_node", END)

    # 编译
    app = graph_builder.compile(checkpointer=get_pgsql_checkpointer())

    return app


# 测试多轮对话
if __name__ == "__main__":
    app = create_langgraph_app()
    config = {"configurable": {"thread_id": "user_test_2"}}  # 关键！必须有 thread_id
    while True:
        user_input = input("👤 You: ")
        if user_input.lower() in ["退出", "exit", "quit"]:
            break

        inputs = {"messages": [HumanMessage(content=user_input)]}

        for chunk in app.stream(inputs, config=config):
            if "call_model_node" in chunk:
                ai_msg = chunk["call_model_node"]["messages"][-1]
                print(f"🤖 Bot: {ai_msg.content}")

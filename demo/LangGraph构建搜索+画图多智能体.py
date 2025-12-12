from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from agents.search_chart_agent import research_agent, chart_agent


class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    sender: str


def researcher(state: State):
    # 注意：create_agent 接收 {"messages": [...]}，不是 {"input": ...}
    last_msg = state["messages"][-1]
    inputs = {"messages": [last_msg]}

    # 执行整个 agent graph，获取最终 messages
    final_state = research_agent.invoke(inputs)
    final_message = final_state["messages"][-1]  # 最后一条是 AI 回复

    return {
        "messages": [AIMessage(content=final_message.content, name="researcher")],
        "sender": "researcher"
    }


def chart_gen(state: State):
    result_state = chart_agent.invoke({"messages": state["messages"][-1]})
    last_message = result_state["messages"][-1]

    return {
        "messages": [AIMessage(content=last_message.content, name="chart")],
        "sender": "chart"
    }


def router(state: State):
    last = state["messages"][-1]
    if "FINAL ANSWER" in last.content or "图表已生成" in last.content:
        return END
    return "chart_gen" if state["sender"] == "researcher" else "researcher"


workflow = StateGraph(State)
workflow.add_node("researcher", researcher)
workflow.add_node("chart_gen", chart_gen)
workflow.add_edge(START, "researcher")
workflow.add_conditional_edges("researcher", router, {"chart_gen": "chart_gen", END: END})
workflow.add_conditional_edges("chart_gen", router, {"researcher": "researcher", END: END})

graph = workflow.compile()


if __name__ == "__main__":
    # graph.get_graph().draw_png("workflow.png")

    mermaid_code = graph.get_graph().draw_mermaid()
    print("\n\n=== Mermaid 流程图代码 ===")
    print(mermaid_code)

    # query = "查询 2020-2024 年全球 AI 软件市场规模，然后画一张折线图。"
    # thread = {"configurable": {"thread_id": "demo"}}
    # for step in graph.stream({"messages": [HumanMessage(content=query)]}, thread):
    #     print(step)

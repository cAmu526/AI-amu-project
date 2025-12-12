# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from langchain_core.messages import HumanMessage
from utils.logger import get_logger

from demo.langgraph构建聊天机器人 import create_langgraph_app

logger = get_logger(__name__)
app = FastAPI(title="LangGraph Chat API", version="1.0")

# 加载一次 graph app（全局单例）
langgraph_app = create_langgraph_app()


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000, description="用户输入的问题")
    thread_id: str = Field(..., min_length=1, max_length=100, description="对话线程ID，用于状态持久化")


class ChatResponse(BaseModel):
    response: str
    thread_id: str


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        inputs = {"messages": [HumanMessage(content=request.message)]}
        config = {"configurable": {"thread_id": request.thread_id}}

        final_state = langgraph_app.invoke(inputs, config=config)

        ai_message = None
        for msg in reversed(final_state["messages"]):
            if hasattr(msg, 'content') and not isinstance(msg, HumanMessage):
                ai_message = msg.content
                break

        if ai_message is None:
            ai_message = "抱歉，我无法生成回答。"

        logger.info(f"Thread {request.thread_id}: 回复成功")
        return ChatResponse(response=ai_message, thread_id=request.thread_id)

    except Exception as e:
        logger.error(f"处理请求时出错: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"内部错误: {str(e)}")

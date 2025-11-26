from langchain_openai import ChatOpenAI
from langchain_community.embeddings import DashScopeEmbeddings

from config.config import Config
from typing import Optional
from utils.logger import get_logger

logger = get_logger(__name__)


class LLMInitializationError(Exception):
    """自定义异常类用于LLM初始化错误"""
    pass


# 全局变量：用于缓存已创建的 ChatOpenAI 实例，避免重复初始化（单例模式）
_chat_llm_instance: Optional[ChatOpenAI] = None
_embedding_instance: Optional[DashScopeEmbeddings] = None


def get_llm() -> ChatOpenAI:
    """
    获取一个 ChatOpenAI 语言模型实例（使用单例模式）。

    该函数首次调用时会根据配置创建一个新的 LLM 实例，
    后续调用将直接返回已创建的实例，以提高性能并避免重复连接。

    返回:
        ChatOpenAI: 配置好的语言模型实例。
    """
    global _chat_llm_instance  # 声明使用全局变量
    # 如果尚未创建实例，则根据传入参数（或默认配置）初始化一个 ChatOpenAI 对象
    if not _chat_llm_instance:
        try:
            _chat_llm_instance = ChatOpenAI(
                base_url=Config.LLM_BASE_URL,  # 指定自定义 LLM 服务的 API 地址（如 vLLM）
                api_key=Config.LLM_API_KEY,  # API 密钥（对于本地部署的 vLLM，通常可填任意值）
                model=Config.LLM,  # 模型名称，需与服务端加载的模型匹配
                temperature=Config.LLM_TEMPERATURE,  # 控制输出多样性
            )
            logger.info(f"成功初始化 {Config.LLM} 模型")
        except Exception as e:
            logger.error(f"初始化 {Config.LLM} 失败: {str(e)}")
            raise LLMInitializationError(f"Failed to initialize LLM: {e}")

    # 返回已创建的实例（确保整个应用生命周期内只创建一次）
    return _chat_llm_instance


def get_embedding_model() -> DashScopeEmbeddings:
    global _embedding_instance
    if _embedding_instance is None:
        try:
            _embedding_instance = DashScopeEmbeddings(
                dashscope_api_key=Config.LLM_API_KEY,
                model=Config.EMBEDDING_MODEL,
            )
            logger.info(f"成功初始化 {Config.EMBEDDING_MODEL} 模型")
        except Exception as e:
            logger.error(f"初始化 embedding_model 失败: {str(e)}")
            raise LLMInitializationError(f"Failed to initialize embedding model: {e}")
    return _embedding_instance


if __name__ == '__main__':
    # llm = get_llm()
    # response = llm.invoke("介绍下你自己")
    # print(response.content)
    get_embedding_model()

# -*- coding: utf-8 -*-
"""
功能说明：将 PDF 文件进行文本切分、向量化，并持久化存储到 Chroma 向量数据库中。
支持从指定 PDF 路径加载文档，使用阿里云 DashScope 的 embedding 模型生成向量，
并通过分批方式写入 Chroma（规避 DashScope 单次最多处理 10 个文本的限制）。
"""

from typing import Optional, List, Dict, Tuple
from langchain_chroma import Chroma  # 使用官方推荐的 langchain-chroma 包
from config.config import Config
from utils.llm import get_embedding_model  # 获取阿里云 DashScope Embedding 模型实例
from utils.rag.pdf2chunks import pdf_to_chunks  # 将 PDF 转换为文本块（chunks）
from utils.logger import get_logger
import threading

# 初始化模块日志器
logger = get_logger(__name__)

# 使用 (persist_directory, collection_name) 作为键，缓存多个 Chroma 实例
# 类型提示：Dict[ (路径, 集合名), Chroma实例 ]
_vectorstore_cache: Dict[Tuple[str, str], Chroma] = {}

# 线程锁：确保在多线程环境下，同一 key 不会被重复初始化（避免 race condition）
_cache_lock = threading.Lock()


def get_vectorstore(
        save_path: str = Config.CHROMADB_DIRECTORY,
        collection_name: str = Config.CHROMADB_COLLECTION_NAME
) -> Chroma:
    """
    获取指定路径和集合名的 Chroma 向量数据库实例。

    采用“缓存字典 + 双重检查锁”模式实现线程安全的懒加载单例（按参数区分）。
    首次调用时创建实例并缓存，后续相同参数的调用直接返回缓存实例。

    Args:
        save_path (str): Chroma 数据库的持久化目录路径。
        collection_name (str): 要加载的集合（collection）名称。

    Returns:
        Chroma: 对应配置的 Chroma 向量库实例。

    Example:
        # 加载默认知识库
        vs1 = get_vectorstore()

        # 加载另一个知识库（不同 collection）
        vs2 = get_vectorstore(collection_name="medical_records")

        # vs1 和 vs2 是两个独立的 Chroma 实例
    """
    # 构建唯一缓存键：由路径和集合名共同决定
    cache_key: Tuple[str, str] = (save_path, collection_name)

    # 第一次检查：避免不必要的加锁开销（性能优化）
    if cache_key not in _vectorstore_cache:
        with _cache_lock:
            # 双重检查：防止多个线程在等待锁时重复创建
            if cache_key not in _vectorstore_cache:
                # 创建新的 Chroma 实例并加入缓存
                vectorstore = Chroma(
                    persist_directory=save_path,
                    collection_name=collection_name,
                    embedding_function=get_embedding_model()  # 使用全局 embedding 模型
                )
                _vectorstore_cache[cache_key] = vectorstore
                # 可选：记录日志（如需）
                # logger.info(f"新创建 Chroma 实例: {cache_key}")

    return _vectorstore_cache[cache_key]


def get_retriever(
        save_path: str = Config.CHROMADB_DIRECTORY,
        collection_name: str = Config.CHROMADB_COLLECTION_NAME
):
    """
    获取基于 Chroma 的检索器（Retriever），用于后续 RAG 查询。

    Retriever 支持语义相似度搜索，可直接用于 LangChain 的 RetrievalQA 等组件。

    Args:
        save_path (str): 同 get_vectorstore。
        collection_name (str): 同 get_vectorstore。

    Returns:
        BaseRetriever: Chroma 封装的检索器对象。
    """
    vectorstore = get_vectorstore(save_path=save_path, collection_name=collection_name)
    return vectorstore.as_retriever()


def batch_add_document(chunks: List, max_batch_size: int = 10) -> None:
    """
    分批将文本块（chunks）添加到 Chroma 向量库中。

    由于阿里云 DashScope 的 embedding 接口限制单次最多处理 10 个文本，
    因此必须对 chunks 进行分批处理，避免 API 调用失败。

    Args:
        chunks (List[Document]): 由 pdf_to_chunks 生成的 Document 列表。
        max_batch_size (int): 每批最大文本数量，默认为 10（DashScope 限制）。
    """
    vectorstore = get_vectorstore()
    for i in range(0, len(chunks), max_batch_size):
        batch_chunks = chunks[i:i + max_batch_size]
        logger.info(f"正在添加第 {i // max_batch_size + 1} 批文档，共 {len(batch_chunks)} 个 chunk")
        vectorstore.add_documents(documents=batch_chunks)
    logger.info(f"成功将 {len(chunks)} 个文本块写入 Chroma 向量库")


def pdf_save_chromadb(
        file_path: str,
        page_numbers: Optional[List[int]] = None,
        min_line_length: int = 1,
) -> None:
    """
    主入口函数：将指定 PDF 文件解析、切分、向量化并存入 Chroma 数据库。

    流程：
    1. 调用 pdf_to_chunks 解析 PDF 并生成文本块；
    2. 调用 batch_add_document 分批写入向量库。

    Args:
        file_path (str): PDF 文件的绝对或相对路径。
        page_numbers (Optional[List[int]]): 可选，指定要提取的页码列表（从 0 开始）。
        min_line_length (int): 最小行长度过滤，低于此值的行将被忽略，默认为 1。
    """
    logger.info(f"开始处理 PDF 文件: {file_path}")
    # 1. PDF 转文本块
    chunks = pdf_to_chunks(
        file_path=file_path,
        page_numbers=page_numbers,
        min_line_length=min_line_length
    )
    logger.info(f"PDF 解析完成，共生成 {len(chunks)} 个文本块")

    # 2. 分批写入向量库
    batch_add_document(chunks=chunks)


# ==================== 测试代码 ====================
if __name__ == "__main__":
    # 示例：加载 PDF 并灌库（取消注释即可运行）
    # pdf_save_chromadb(file_path='../../document/健康档案.pdf')

    # 示例：从已有的 Chroma 库中检索“李四六的病历”
    # 注意：需确保之前已成功灌库且包含相关内容
    try:
        result = get_retriever().invoke('李四六的病历')
        print("检索结果:")
        for doc in result:
            print(f"- {doc.page_content[:100]}...")  # 打印前100字符预览
    except Exception as e:
        logger.error(f"检索失败: {e}")

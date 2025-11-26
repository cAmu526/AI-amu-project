# -*- coding: utf-8 -*-
"""
功能说明：将 PDF 文件进行文本切分、向量化，并持久化存储到 Milvus 向量数据库中。
支持从指定 PDF 路径加载文档，使用阿里云 DashScope 的 embedding 模型生成向量，
并通过分批方式写入 Milvus（规避 DashScope 单次最多处理 10 个文本的限制）。
"""

from typing import Optional, List, Dict, Tuple
from langchain_milvus import Milvus
from config.config import Config
from utils.llm import get_embedding_model
from utils.rag.pdf2chunks import pdf_to_chunks
from utils.logger import get_logger
import threading

# 初始化模块日志器
logger = get_logger(__name__)

# 缓存字典：key = (connection_args_str, collection_name)
# 注意：Milvus 的连接参数较复杂，我们用 (uri, collection_name) 作为 key
_vectorstore_cache: Dict[Tuple[str, str], Milvus] = {}
_cache_lock = threading.Lock()


def get_vectorstore(
        collection_name: str = Config.CHROMADB_COLLECTION_NAME  # 仍叫 COLLECTION_NAME，但用于 Milvus
) -> Milvus:
    """
    获取指定集合名的 Milvus 向量数据库实例。

    使用缓存避免重复连接，支持多 collection。
    Milvus Lite 会自动创建 ./milvus.db 文件并持久化。

    Args:
        collection_name (str): Milvus 集合名称。

    Returns:
        Milvus: 对应的向量库实例。
    """
    uri = Config.MILVUS_URI
    cache_key: Tuple[str, str] = (uri, collection_name)

    if cache_key not in _vectorstore_cache:
        with _cache_lock:
            if cache_key not in _vectorstore_cache:
                vectorstore = Milvus(
                    embedding_function=get_embedding_model(),
                    collection_name=collection_name,
                    connection_args={"uri": uri},  # Milvus Lite 或 Server 地址
                    # 可选：自动创建集合（默认 True）
                    # drop_old=False  # 是否覆盖已有集合（灌库时慎用！）
                )
                _vectorstore_cache[cache_key] = vectorstore
                logger.info(f"新创建 Milvus 实例: uri={uri}, collection={collection_name}")

    return _vectorstore_cache[cache_key]


def get_retriever(collection_name: str = Config.CHROMADB_COLLECTION_NAME):
    """
    获取 Milvus 检索器。
    """
    vectorstore = get_vectorstore(collection_name=collection_name)
    return vectorstore.as_retriever()


def batch_add_document(chunks: List, max_batch_size: int = 10) -> None:
    """
    分批将 chunks 添加到 Milvus。
    Milvus 的 add_documents 内部已做批量优化，但仍需遵守 embedding 的 batch 限制。
    """
    # 注意：这里默认使用 Config 中的默认 collection
    vectorstore = get_vectorstore()
    for i in range(0, len(chunks), max_batch_size):
        batch_chunks = chunks[i:i + max_batch_size]
        logger.info(f"正在添加第 {i // max_batch_size + 1} 批文档，共 {len(batch_chunks)} 个 chunk")
        vectorstore.add_documents(documents=batch_chunks)
    logger.info(f"成功将 {len(chunks)} 个文本块写入 Milvus 向量库")


def pdf_save_chromadb(  # 函数名可改为 pdf_save_milvus，但保留兼容性也可
        file_path: str,
        page_numbers: Optional[List[int]] = None,
        min_line_length: int = 1,
        collection_name: str = Config.CHROMADB_COLLECTION_NAME  # 新增参数
) -> None:
    """
    将 PDF 灌入 Milvus。

    新增 collection_name 参数，支持指定目标集合。
    """
    logger.info(f"开始处理 PDF 文件: {file_path} → 集合: {collection_name}")
    chunks = pdf_to_chunks(
        file_path=file_path,
        page_numbers=page_numbers,
        min_line_length=min_line_length
    )
    logger.info(f"PDF 解析完成，共生成 {len(chunks)} 个文本块")

    # 临时切换到指定 collection（通过局部变量传入）
    vectorstore = Milvus(
        embedding_function=get_embedding_model(),
        collection_name=collection_name,
        connection_args={"uri": Config.MILVUS_URI},
        # 注意：如果集合已存在，不要 drop_old=True，否则会清空数据！
    )

    # 分批添加
    for i in range(0, len(chunks), 10):
        batch = chunks[i:i + 10]
        vectorstore.add_documents(batch)

    logger.info(f"已将 {len(chunks)} 个 chunk 写入 Milvus 集合 '{collection_name}'")


# ==================== 测试代码 ====================
if __name__ == "__main__":
    # 示例：灌库
    # pdf_save_chromadb(
    #     file_path='../../document/健康档案.pdf',
    #     collection_name="health_records"
    # )

    # 示例：检索
    try:
        retriever = get_retriever(collection_name="health_records")
        result = retriever.invoke('李四六的病历')
        print("检索结果:")
        for doc in result:
            print(f"- {doc.page_content[:100]}...")
    except Exception as e:
        logger.error(f"检索失败: {e}")

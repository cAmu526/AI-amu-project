import os
from pathlib import Path

from dotenv import load_dotenv

project_path = str(Path(__file__).parent.parent)
env_path = project_path + "/.env"  # 这里要构造绝对路径，因为load_dotenv函数读取的不是文件的目录，而是工作目录即开始时执行的文件，当在其他目录的文件使用本config时，会使得相对路径错乱读不到.env文件
load_dotenv(dotenv_path=env_path, override=True)  # 导入指定env配置 以env文件为准


class Config:
    """统一的配置类，集中管理所有常量"""

    # LLM配置
    LLM = os.getenv("LLM")
    LLM_BASE_URL = os.getenv("LLM_BASE_URL")
    LLM_API_KEY = os.getenv("LLM_API_KEY")
    LLM_TEMPERATURE = os.getenv("LLM_TEMPERATURE")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

    # prompt模板文件路径
    # PROMPT_TEMPLATE_TXT_AGENT = project_path + "/promptsTemplate/prompt_template_agent.txt"
    # PROMPT_TEMPLATE_TXT_GRADE = project_path + "promptsTemplate/prompt_template_grade.txt"
    # PROMPT_TEMPLATE_TXT_REWRITE = project_path + "promptsTemplate/prompt_template_rewrite.txt"
    # PROMPT_TEMPLATE_TXT_GENERATE = project_path + "promptsTemplate/prompt_template_generate.txt"

    # Chroma 数据库配置
    CHROMADB_DIRECTORY = project_path + "/chromaDB"
    CHROMADB_COLLECTION_NAME = "knowledgeDocument"

    # Milvus 配置
    # MILVUS_URI = "./milvus.db"  # Milvus Lite（单机文件）
    MILVUS_URI = os.getenv("MILVUS_URI")  # Milvus Server

    # 日志目录
    LOG_PATH = project_path + "/logs"

    # 数据库 URI
    POSTGRESQL_DB_URI = os.getenv("POSTGRESQL_DB_URI")

    REDIS_DB_HOST = os.getenv("REDIS_DB_HOST")
    REDIS_DB_PORT = os.getenv("REDIS_DB_PORT")
    REDIS_DB_PASSWORD = os.getenv("REDIS_DB_PASSWORD")

    # MCP服务
    AMAP_MCP_URI = os.getenv("AMAP_MCP_URI")
    AMAP_MCP_API_KEY = os.getenv("AMAP_MCP_API_KEY")

    TAVILY_MCP_API_KEY = os.getenv("TAVILY_API_KEY")

    # API服务地址和端口
    HOST = os.getenv("FAST_API_HOST")
    PORT = os.getenv("FAST_API_PORT")


if __name__ == "__main__":
    print(Config.AMAP_MCP_API_KEY)

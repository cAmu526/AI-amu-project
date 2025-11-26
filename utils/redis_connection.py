from redis import Redis
from config.config import Config
from utils.logger import get_logger

# 全局日志记录器
app_logger = get_logger(__name__)  # 写在前面 避免与RedisSaver中的get_logger冲突


_REDIS_CLIENT = None
REDIS_DB_HOST = Config.REDIS_DB_HOST
REDIS_DB_PORT = Config.REDIS_DB_PORT
REDIS_DB_PASSWORD = Config.REDIS_DB_PASSWORD


def get_redis_client() -> Redis:
    """
    获取全局唯一的 Redis 客户端实例（懒加载 + 单例）。
    """
    global _REDIS_CLIENT
    if _REDIS_CLIENT is None:
        try:
            # --- 传入 password 参数 ---
            _REDIS_CLIENT = Redis(
                host=REDIS_DB_HOST,
                port=REDIS_DB_PORT,
                password=REDIS_DB_PASSWORD,  # <-- 添加这一行
                db=0,
                decode_responses=False,
                socket_connect_timeout=5,
                socket_timeout=5,
            )
            # 测试连接
            _REDIS_CLIENT.ping()
            app_logger.info(f"Successfully connected to Redis at {REDIS_DB_HOST}:{REDIS_DB_PORT}")
        except ConnectionError as e:
            app_logger.error("Failed to connect to Redis: %s", e)
            raise
        except Exception as e:
            app_logger.error("Unexpected error during Redis client initialization: %s", e)
            raise
    return _REDIS_CLIENT


def get_redis_checkpointer(ttl_seconds: int = 604800):
    client = get_redis_client()
    # RedisSaver 会复用你提供的已认证的客户端
    from langgraph.checkpoint.redis import RedisSaver
    saver = RedisSaver(
        redis_client=client,
        ttl={"default_ttl": ttl_seconds}
    )
    saver.setup()
    return saver


if __name__ == '__main__':
    get_redis_checkpointer()

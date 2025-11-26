"""
数据库连接池管理模块

本模块封装了 PostgreSQL 连接池的创建、监控、健康检查及与 LangGraph 集成的工具函数。
使用 psycopg_pool 实现连接池，结合 tenacity 实现自动重试机制，并通过独立线程周期性监控连接池状态，
以预防连接耗尽或数据库不可用等问题。
"""

import threading
import time
from typing import Optional

from langgraph.checkpoint.postgres import PostgresSaver
from psycopg2 import OperationalError
from psycopg_pool import ConnectionPool
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from config.config import Config
from utils.logger import get_logger

# 全局日志记录器
logger = get_logger(__name__)

# 全局连接池实例（单例模式）
_db_connection_pool: Optional[ConnectionPool] = None


class ConnectionPoolError(Exception):
    """自定义异常：表示连接池相关错误（如未初始化、已关闭、测试失败等）"""
    pass


def monitor_connection_pool(db_connection_pool: ConnectionPool, interval: int = 60):
    """
    启动一个守护线程，周期性监控 PostgreSQL 连接池的状态。

    功能：
    - 每隔 `interval` 秒打印当前活跃连接数 / 最大连接数。
    - 当活跃连接数超过最大连接数的 85% 时发出警告，用于提前预警资源瓶颈。

    参数：
        db_connection_pool (ConnectionPool): 要监控的连接池实例。
        interval (int): 监控间隔（秒），默认 60 秒。

    返回：
        threading.Thread: 启动的监控线程对象（通常无需外部管理，因设为 daemon=True）。
    """

    def _monitor():
        """内部监控循环函数，在独立线程中运行"""
        while not db_connection_pool.closed:
            try:
                # 获取连接池统计信息
                stats = db_connection_pool.get_stats()
                active = stats.get("connections_in_use", 0)  # 当前正在使用的连接数
                total = db_connection_pool.max_size  # 连接池最大容量
                logger.info(f"Connection postgresql_pool status: {active}/{total} connections in use")

                # 预警：当使用率 ≥ 85% 时记录警告日志
                if active >= total * 0.85:
                    logger.warning(f"Connection postgresql_pool nearing capacity: {active}/{total}")
            except Exception as e:
                # 捕获所有异常避免监控线程崩溃，但记录错误
                logger.error(f"Failed to monitor connection postgresql_pool: {e}")

            # 等待下一次检查
            time.sleep(interval)

    # 创建并启动守护线程（主程序退出时自动终止）
    monitor_thread = threading.Thread(target=_monitor, daemon=True)
    monitor_thread.start()
    return monitor_thread


def get_connection_pool():
    """
    获取全局 PostgreSQL 连接池实例（单例模式）。

    若尚未初始化，则根据配置创建连接池并启动监控线程。
    连接池参数说明：
        - max_size=20: 最多允许 20 个并发连接。
        - min_size=2: 始终保持至少 2 个空闲连接。
        - timeout=10: 从池中获取连接的最大等待时间为 10 秒。
        - connect_timeout=5: 单个连接建立的超时时间为 5 秒。
        - autocommit=True: 自动提交事务（适用于简单查询场景）。
        - prepare_threshold=0: 禁用语句预编译缓存（减少内存开销）。

    返回：
        ConnectionPool: 初始化好的连接池实例。

    异常：
        ConnectionPoolError: 若连接池打开失败。
    """
    global _db_connection_pool
    if _db_connection_pool is None:
        # 根据配置创建连接池
        _db_connection_pool = ConnectionPool(
            conninfo=Config.POSTGRESQL_DB_URI,
            max_size=20,
            min_size=2,
            kwargs={
                "autocommit": True,
                "prepare_threshold": 0,
                "connect_timeout": 5
            },
            timeout=10
        )
        try:
            # 显式打开连接池（psycopg_pool v3+ 需要）
            _db_connection_pool.open()
            logger.info("postgresql connection pool initialized")
        except Exception as e:
            logger.error(f"Failed to open postgresql connection pool: {e}")
            raise ConnectionPoolError(f"无法打开数据库连接池: {str(e)}")

        # 启动后台监控线程
        monitor_connection_pool(_db_connection_pool, interval=60)

    return _db_connection_pool


def check_connection_pool(db_connection_pool: ConnectionPool):
    """
    对给定的连接池执行全面健康检查。

    检查项包括：
        1. 连接池是否为 None 或已关闭。
        2. 当前活跃连接是否已达上限（防止后续操作阻塞）。
        3. 执行简单 SQL 查询（SELECT 1）验证物理连接可用性。

    参数：
        db_connection_pool (ConnectionPool): 待检查的连接池。

    异常：
        ConnectionPoolError: 任一检查失败时抛出。
    """
    # 检查连接池是否存在且未关闭
    if db_connection_pool is None or db_connection_pool.closed:
        logger.error("Connection postgresql_pool is None or closed")
        raise ConnectionPoolError("数据库连接池未初始化或已关闭")

    try:
        # 获取连接使用情况
        active_connections = db_connection_pool.get_stats().get("connections_in_use", 0)
        max_connections = db_connection_pool.max_size

        # 若连接已满，立即报错避免后续操作挂起
        if active_connections >= max_connections:
            logger.error(
                f"Connection postgresql_pool exhausted: {active_connections}/{max_connections} connections in use"
            )
            raise ConnectionPoolError("连接池已耗尽，无可用连接")

        # 执行实际数据库查询测试连接有效性
        if not test_connection(db_connection_pool):
            raise ConnectionPoolError("连接池测试失败")

        logger.info("Connection postgresql_pool status: OK, test connection successful")

    except OperationalError as e:
        # 特定捕获数据库操作异常（如网络中断、认证失败等）
        logger.error(f"postgresql operational error during connection test: {e}")
        raise ConnectionPoolError(f"连接池测试失败，可能已关闭或超时: {str(e)}")
    except Exception as e:
        # 兜底异常处理
        logger.error(f"Failed to verify connection postgresql_pool status: {e}")
        raise ConnectionPoolError(f"无法验证连接池状态: {str(e)}")


@retry(
    stop=stop_after_attempt(3),  # 最多重试 3 次
    wait=wait_exponential(multiplier=1, min=2, max=10),  # 指数退避：2s → 4s → 8s（上限10s）
    retry=retry_if_exception_type(OperationalError)  # 仅对 OperationalError 重试
)
def test_connection(db_connection_pool: ConnectionPool) -> bool:
    """
    测试连接池是否能成功执行简单查询。

    使用连接池获取一个连接，执行 "SELECT 1" 并验证返回结果。
    此函数被 @retry 装饰，具备自动重试能力，适用于临时性网络抖动或数据库短暂不可用场景。

    参数：
        db_connection_pool (ConnectionPool): 连接池实例。

    返回：
        bool: 测试成功返回 True。

    异常：
        ConnectionPoolError: 查询结果异常。
        OperationalError: 数据库底层错误（会触发重试）。
    """
    with db_connection_pool.getconn() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            if result != (1,):
                raise ConnectionPoolError("连接池测试查询失败，返回结果异常")
    return True


_cached_checkpointer: Optional[PostgresSaver] = None


def get_pgsql_checkpointer() -> PostgresSaver:
    """
    创建并初始化 LangGraph 的 PostgreSQL 检查点存储器（用于持久化对话状态/记忆）。

    该函数依赖已初始化的连接池，调用 PostgresSaver.setup() 完成表结构初始化（若不存在）。
    """
    global _cached_checkpointer
    if _cached_checkpointer is None:
        try:
            db_connection_pool = get_connection_pool()
            checkpointer = PostgresSaver(db_connection_pool)
            checkpointer.setup()  # 幂等操作，安全
            _cached_checkpointer = checkpointer
            logger.info("PostgresSaver initialized and cached.")
        except Exception as e:
            logger.error(f"Failed to setup PostgresSaver: {e}")
            raise ConnectionPoolError(f"检查点初始化失败: {str(e)}")

    return _cached_checkpointer



if __name__ == "__main__":
    # 示例：初始化连接池并执行基本健康检查
    pool = get_connection_pool()
    check_connection_pool(pool)
    test_connection(pool)
    get_pgsql_checkpointer()
    get_pgsql_checkpointer()

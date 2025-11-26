import logging
from logging.handlers import TimedRotatingFileHandler
from config.config import Config
from pathlib import Path

LOG_PATH = Path(Config.LOG_PATH)
LOG_PATH.mkdir(exist_ok=True)  # 自动创建 log 目录

# 日志格式
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_BASE_FILE = LOG_PATH / "app.log"


def get_logger(name: str = "app") -> logging.Logger:
    """
    获取一个配置好的 logger 实例。

    参数:
        name (str): logger 的名称，通常传入 __name__ 以标识模块来源。

    返回:
        logging.Logger: 配置好的日志记录器。
    """
    logger = logging.getLogger(name)

    # 避免重复添加 handler（防止多次调用导致日志重复）
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.DEBUG)

    # 创建控制台处理器（输出到终端）
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    console_handler.setFormatter(console_formatter)

    # 创建按天轮转的文件处理器（每天一个日志文件，保留30天）
    file_handler = TimedRotatingFileHandler(
        filename=LOG_BASE_FILE,
        when="midnight",  # 每天午夜轮转
        interval=1,
        backupCount=30,  # 保留最近30天的日志
        encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    file_handler.setFormatter(file_formatter)
    file_handler.suffix = "%Y-%m-%d"  # 轮转后的文件后缀，如 app.log.2025-11-16

    # 添加处理器
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


if __name__ == "__main__":
    logger = get_logger(__name__)
    logger.info("应用程序启动中...")

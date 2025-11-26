from langchain_core.tools import tool
from typing import List, Dict
from langchain_core.tools import BaseTool


# 工具函数定义在模块顶层（只会加载一次）
# 1. 定义工具函数,需要加上@tool装饰器和function函数的 docstring（文档字符串）
@tool
def get_weather(location: str) -> str:
    """获取指定城市的天气"""
    return f"{location} 的天气是晴朗，25°C。"


@tool
def add_numbers(a: int, b: int) -> int:
    """计算两个整数的和"""
    return a + b


# 构建工具列表和映射（模块导入时执行一次）
TOOLS: List[BaseTool] = [get_weather, add_numbers]
TOOL_MAP: Dict[str, BaseTool] = {tool.name: tool for tool in TOOLS}


# 提供访问接口（可选，方便类型提示或未来扩展）
def get_tools() -> tuple[List[BaseTool], Dict[str, BaseTool]]:
    return TOOLS, TOOL_MAP

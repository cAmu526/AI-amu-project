from langchain_tavily import TavilySearch
from langchain_experimental.tools import PythonREPLTool
from config.config import Config

search = TavilySearch(max_results=3, tavily_api_key=Config.TAVILY_MCP_API_KEY)
python = PythonREPLTool()

research_tools = [search]
chart_tools = [python]

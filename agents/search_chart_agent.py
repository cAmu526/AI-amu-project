from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from tools.search_chart_tools import research_tools, chart_tools
from utils.llm import get_llm
from tools.custom_tools import TOOLS

llm = get_llm()

research_agent = create_agent(
    model=llm,
    tools=research_tools,
    system_prompt="你是研究专家，负责收集最新数据并返回纯文本报告。"
)

chart_agent = create_agent(
    model=llm,
    tools=chart_tools,
    system_prompt="你是数据可视化专家，收到研究文本后生成折线图 Python 代码并执行。"
)

test_agent = create_agent(
    model=llm,
    tools=TOOLS,
    system_prompt=""
)

if __name__ == '__main__':
    # test_agent.invoke(
    #     {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
    # )

    inputs = {"messages": [HumanMessage(content="帮我查下北京的天气怎么样，然后计算下1加4")]}
    for chunk in test_agent.stream(inputs):
        print(chunk)

    # inputs = {"messages": [HumanMessage(content="世界上石油存储量有多少？")]}
    # for chunk in research_agent.stream(inputs):
    #     print(chunk)

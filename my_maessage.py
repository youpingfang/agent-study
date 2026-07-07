import os
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from dotenv import load_dotenv
from langchain.agents import create_agent
# 1. 定义工具 (直接使用 @tool 装饰器更简洁)
@tool
def tavily_search(query: str):
    """搜索互联网获取最新信息"""
    tavily_tool = TavilySearch(max_results=10, api_key=os.getenv("TAVILY_API_KEY"))
    return tavily_tool.invoke(query)
load_dotenv()
# 2. 初始化模型
model = init_chat_model(
    model="deepseek-v4-flash",
    model_provider="deepseek",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)

# 3. 绑定工具
model_with_tools = model.bind_tools([tavily_search])

# 4. 使用 LangGraph 的 create_react_agent (最简单的 invoke 方式)
# 它自动处理了“思考 -> 调用工具 -> 获取结果 -> 最终回复”的循环
agent_executor = create_agent(model, [tavily_search])

# 5. 直接 invoke
query = "请帮我查询今天 IT 之家的前面 10 条数据。"
response = agent_executor.invoke({
    
    "messages": [SystemMessage(content="你是一个专业的新闻搜索助手"), HumanMessage(content=query)]
})
for msg in response["messages"]:
    msg.pretty_print()


from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langchain_tavily import TavilySearch
import requests
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

from langchain.agents.middleware import AgentState, AgentMiddleware
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from init_LLM import deepseek_flash_model, deepseek_pro_model
from get_env_utils import city_key, travily_api_key


@tool
def get_weather_data(city: str) -> str:  
    """
    获取指定城市的实时天气信息，包括温度、湿度、风速、天气描述等，城市名称要转换成拼音，而且首字母要大写，比日北京：Beijing，
    """
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={city_key}&units=metric&lang=zh_cn"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        
    except requests.exceptions.RequestException as e:
        return f"获取天气失败: {e}"
    
    weather_desc = data["weather"][0]["description"]
    temp = data["main"]["temp"]
    feels_like = data["main"]["feels_like"]
    humidity = data["main"]["humidity"]
    wind_speed = data["wind"]["speed"]
    city_name = data["name"]

    return json.dumps({
        "城市": city_name,
        "天气": weather_desc,
        "温度": f"{temp}°C",
        "体感温度": f"{feels_like}°C",
        "湿度": f"{humidity}%",
        "风速": f"{wind_speed} m/s",
    }, ensure_ascii=False)

@tool
def tavily_search(query: str) -> str:
    """
    搜索指定关键词的新闻，返回新闻标题即可。
    """
    tavily_search_tool = TavilySearch(max_results=10, api_key=travily_api_key)

    return tavily_search_tool.invoke(query)


# 初始化数据库连接, 并创建数据库表, 用于存储线程的中间状态, 每个线程一个内存
connect = sqlite3.connect("agent_memory.db",check_same_thread=False)
memoryconfig: RunnableConfig = {"configurable": {"thread_id": "user_session_agent_memory"}}
checkpointer = SqliteSaver(connect)
#setup用来初始化数据库表
checkpointer.setup()

agent = create_agent(
    model=deepseek_flash_model, 
    tools=[get_weather_data, tavily_search],  
    system_prompt="你是一个天气查询和新闻搜索助手",
    # 初始化SqliteSaver(connect)传入数据库连接
    checkpointer=checkpointer,
)

result = agent.invoke({"messages": [HumanMessage(content="南京的天气如何")]}, memoryconfig)
print(result["messages"][-1].content)

result = agent.invoke({"messages": [HumanMessage(content="我刚刚问的是哪个城市的天气")]}, memoryconfig)
print(result["messages"][-1].content)



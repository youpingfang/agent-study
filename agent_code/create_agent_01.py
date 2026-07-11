
#!/usr/bin/env uv run
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
import requests
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from init_LLM import deepseek_flash_model
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
# tavily_search_tool = TavilySearch(max_results=5, api_key=travily_api_key)
@tool
def tavily_search(query: str) -> str:
    """
    搜索指定关键词的新闻，返回新闻标题即可。
    """
    tavily_search_tool = TavilySearch(max_results=5, api_key=travily_api_key)

    return tavily_search_tool.invoke(query)

agent = create_agent(
    model=deepseek_flash_model, 
    tools=[get_weather_data, tavily_search],
    system_prompt="你是一个天气查询搜索助手"
)
result = agent.invoke({"messages": [HumanMessage(content="珍蚌是什么意思？")]})
for msg in result["messages"]:
    msg.pretty_print()


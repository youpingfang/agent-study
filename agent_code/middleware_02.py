
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
import requests

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

# 尝试去掉括号
@AgentMiddleware
def dynamic_model_selector(state: AgentState) -> AgentState:
    """
    根据问题复杂度动态选择模型。

    规则：
      - 问题字符 < 20 → 用便宜模型
      - 问题字符 >= 20 → 用贵模型
      - 如果上一条消息是 ToolMessage（工具刚执行完）→ 换回便宜模型
    """
    messages = list(state["messages"])

    # 判断上一条消息是不是工具结果
    last_msg = messages[-1] if messages else None
    if last_msg and last_msg.type == "tool":
        # 工具刚执行完，换回便宜的模型
        model = deepseek_flash_model
    else:
        # 看最后一条 HumanMessage 的长度
        for msg in reversed(messages):
            if msg.type == "human" and isinstance(msg.content, str):
                if len(msg.content) > 20:
                    model = deepseek_pro_model
                else:
                    model = deepseek_flash_model
                break
        else:
            model = deepseek_flash_model  # 默认

    # 把选好的模型存到 state 的 metadata 里
    # 这样可以在外部看到选了哪个模型
    print(f"🔀 模型选择: {'💰 贵的' if model == deepseek_pro_model else '🪙 便宜的'}")
  

agent = create_agent(
    model=deepseek_flash_model, 
    tools=[get_weather_data, tavily_search],   
    middleware=[dynamic_model_selector,]
)

result = agent.invoke({"messages": [HumanMessage(content="南京的天气")]})
print(result["messages"][-1].content)


from  langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
import os
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from dotenv import load_dotenv
from langchain.agents import create_agent
import requests

load_dotenv()
# 2. 初始化模型
model = init_chat_model(
    model="deepseek-v4-flash",
    model_provider="deepseek",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)


# 1. 定义工具 (直接使用 @tool 装饰器更简洁)
@tool
def weather_search(city:str):
    """
    获取指定城市的实时天气信息，包括温度、湿度、风速、天气描述等，城市名称要转换成拼音，而且首字母要大写，比日北京：Beijing
    """
    weather_api_key = os.getenv("WEATHER_API_KEY")
    
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={weather_api_key}&units=metric&lang=zh_cn"
    
    try:
        response = requests.get(url)
        data = response.json()
        if response.status_code != 200:
            return f"无法查询到{city}的天气，请检查城市名称是否正确。"
        weather = data["weather"][0]
        return f"天气: {weather['description']}, 温度: {data['main']['temp']}°C, 湿度: {data['main']['humidity']}%,"
    except Exception as e:
        return f"查询{city}的天气时出错：{e}"

# 3. 绑定工具并创建 Agent
tools = [weather_search]
agents = create_agent(model, tools)
# 调用 Agent
response = agents.invoke({
    "messages": [HumanMessage(content="帮我查询今天北京的天气。")]
})

# 打印最终回答
print(response["messages"][-1].content)


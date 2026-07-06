from  langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from dotenv import load_dotenv
import requests
from get_env_utils import deepseek_api_key, deepseek_api_url, city_key, weather_api_key, travily_api_key
# 2. 初始化模型
model = init_chat_model(
    model="deepseek-v4-flash",
    model_provider="deepseek",
    api_key=deepseek_api_key,
)

prompts = ChatPromptTemplate.from_messages([
    ("system", "你是一个天气助手，用户会询问天气相关的问题，你需要调用 weather_search 函数来获取天气信息即可。"),
    ("human", "{input}")
])
    

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
llm_with_tools = model.bind_tools([weather_search])         
        # 使用 initialize_agent 替代不存在的 create_tool_calling_agent
agent = initialize_agent(
    tools=[weather_search],
    llm=llm_with_tools,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    system_message=prompts,
)


from init_LLM import deepseek_flash_model
from langchain_core.tools import tool
from get_env_utils import city_key
import requests
import json


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
    
weather_model_tools = deepseek_flash_model.bind_tools([get_weather_data])
result = weather_model_tools.invoke("北京的天气")
for toolcall in result.tool_calls:
    if toolcall["name"] == "get_weather_data":
        weather_result = get_weather_data.invoke(toolcall)
        print(weather_result.content)



#!/usr/bin/env -S uv run
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam
import os
import json
import requests
from dotenv import load_dotenv
import sys

load_dotenv()

deepseek_api_key = os.getenv("Deepseek_API_KEY")
deepseek_api_url = os.getenv("Deepseek_API_URL")
city_key = os.getenv("OPENWEATHER_API_KEY")


client = OpenAI(
    api_key=deepseek_api_key,
    base_url=deepseek_api_url,
)


def get_weather(city: str) -> str:
    """
    调用 OpenWeatherMap API 获取指定城市的实时天气信息
    """
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={city_key}&units=metric&lang=zh_cn"
    try:
        response = requests.get(url)
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


TOOLS: list[ChatCompletionToolParam] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的实时天气信息，包括温度、湿度、风速、天气描述等，城市名称要转换成拼音，而且首字母要大写，比日北京：Beijing，",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，例如 北京、上海、Tokyo、London",
                    }
                },
                "required": ["city"],
            },
        },
    },
]

SYSTEM_PROMPT = "你是一个天气助手，用户会询问天气相关的问题，你需要调用 get_weather 函数来获取天气信息即可。"


def chat_once(city_name: str):
    """处理一次天气查询并输出结果"""
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]
    messages.append({"role": "user", "content": f"{city_name}的天气怎么样"})

    while True:
        response = client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            stream=False,
            temperature=0.7,
            max_tokens=1000,
        )

        assistant_message = response.choices[0].message
        messages.append(assistant_message.model_dump(exclude_none=True))

        if not assistant_message.tool_calls:
            print(f"AI: {assistant_message.content or ''}\n")
            break

        for tool_call in assistant_message.tool_calls:
            print(f"AI 调用了工具: {tool_call.function.name}，参数: {tool_call.function.arguments}")
            if tool_call.function.name == "get_weather":
                args = json.loads(tool_call.function.arguments)
                print(f"正在查询 {args['city']} 的天气...")
                result = get_weather(args["city"])
                print(f"查询结果: {result}\n")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })


if len(sys.argv) > 1:
    # 命令行模式：./get_weather.py 北京
    chat_once(" ".join(sys.argv[1:]))
else:
    # 交互模式
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    while True:
        prompt = input("请输入你想查询的城市: ").strip()
        if prompt.lower() == "exit":
            break

        messages.append({"role": "user", "content": prompt})

        while True:
            response = client.chat.completions.create(
                model="deepseek-v4-flash",
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                stream=False,
                temperature=0.7,
                max_tokens=1000,
            )

            assistant_message = response.choices[0].message
            messages.append(assistant_message.model_dump(exclude_none=True))

            if not assistant_message.tool_calls:
                print(f"AI: {assistant_message.content or ''}\n")
                break

            for tool_call in assistant_message.tool_calls:
                print(f"AI 调用了工具: {tool_call.function.name}，参数: {tool_call.function.arguments}")
                if tool_call.function.name == "get_weather":
                    args = json.loads(tool_call.function.arguments)
                    print(f"正在查询 {args['city']} 的天气...")
                    result = get_weather(args["city"])
                    print(f"查询结果: {result}\n")

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    })

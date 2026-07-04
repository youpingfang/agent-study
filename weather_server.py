#!/usr/bin/env -S uv run
"""
MCP 服务端 — 天气查询工具
======================

MCP（Model Context Protocol）是一种让 AI 模型安全调用外部工具的协议。
这个文件是 MCP 的"服务端"（Server），它提供具体的工具（Tool），
通过标准输入输出（stdio）与客户端通信。

数据流：
  Client (你写的代码 / Claude Desktop)
    → 通过 stdio 发送 JSON-RPC 请求
    → Server (这个文件) 执行工具函数
    → 通过 stdio 返回 JSON-RPC 响应

关键词：Server、Tool、stdio 传输
"""

# ============================================================
# 第一部分：导入依赖
# ============================================================
import os
import json
import requests
from pathlib import Path
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# 明确指定 .env 的路径，避免子进程工作目录问题
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

# 从 .env 文件中读取 API Key
# 这些 Key 不要写死在代码里，而是放在 .env 中，防止提交到 Git
deepseek_api_key = os.getenv("Deepseek_API_KEY")
deepseek_api_url = os.getenv("Deepseek_API_URL")
city_key = os.getenv("OPENWEATHER_API_KEY")


# ============================================================
# 第三部分：创建 MCP 服务端实例
# ============================================================
# FastMCP 是 MCP 协议的高级封装，你只需要关注"提供什么工具"就行
# 它自动处理了：
#   1. JSON-RPC 协议通信
#   2. stdio 传输层（通过标准输入输出与客户端对话）
#   3. 工具注册和参数校验
mcp = FastMCP("weather-server")
# 参数 "weather-server" 是这个服务的名称，在客户端连接时会用到


# ============================================================
# 第四部分：核心工具函数
# ============================================================
def get_weather_data(city: str) -> dict:
    """
    调用 OpenWeatherMap API 获取天气数据

    这个函数是纯粹的"业务逻辑"，不关心 MCP 协议。
    把它独立出来，方便后面的单元测试或复用。
    """
    # 城市名转拼音 + 首字母大写 is handled by the LLM calling this tool
    url = (f"http://api.openweathermap.org/data/2.5/weather"
           f"?q={city}&appid={city_key}&units=metric&lang=zh_cn")

    try:
        response = requests.get(url)
        data = response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"获取天气失败: {e}"}

    # 解析 OpenWeatherMap 返回的数据
    # 注意：API 返回 200 是成功，其他都是错误
    if data.get("cod") != 200:
        msg = data.get("message", "未知错误")
        return {"error": f"查询失败: {msg}（城市名请用拼音，如 Beijing）"}

    return {
        "城市": data["name"],
        "天气": data["weather"][0]["description"],
        "温度": f"{data['main']['temp']}°C",
        "体感温度": f"{data['main']['feels_like']}°C",
        "湿度": f"{data['main']['humidity']}%",
        "风速": f"{data['wind']['speed']} m/s",
    }


# ============================================================
# 第五部分：注册 MCP 工具
# ============================================================
# @mcp.tool() 装饰器把一个普通 Python 函数注册为 MCP 工具（Tool）
# 注册后，客户端（或 Claude Desktop）就能发现并调用它
#
# MCP Tool 的核心概念：
#   - name：函数名就是工具名（get_weather）
#   - description：函数的 docstring 就是工具描述，LLM 靠它判断是否调用
#   - parameters：函数的参数列表（city: str），自动成为工具的入参 schema
@mcp.tool()
def get_weather(city: str) -> str:
    """
    获取指定城市的实时天气信息，包括温度、湿度、风速、天气状况等。
    城市名称请转换为拼音，首字母大写，例如：北京 → Beijing。

    函数签名中的 city 参数会自动成为 MCP Tool 的入参定义。
    """
    result = get_weather_data(city)

    if "error" in result:
        return result["error"]

    # 返回格式化的文本给客户端
    return (f"城市：{result['城市']}\n"
            f"天气：{result['天气']}\n"
            f"温度：{result['温度']}\n"
            f"体感温度：{result['体感温度']}\n"
            f"湿度：{result['湿度']}\n"
            f"风速：{result['风速']}")


# ============================================================
# 第六部分：启动服务端
# ============================================================
# __main__ 入口：当直接运行这个文件时，启动 MCP 服务端
# mcp.run() 默认使用 stdio 传输（通过标准输入输出通信）
#
# 两种传输方式：
#   1. stdio（开发/本地用）：通过命令行管道通信
#      - 启动：无参数，直接运行
#      - 客户端通过 subprocess 启动服务端进程
#   2. SSE（生产/远程用）：通过 HTTP 通信（本文件不涉及）
if __name__ == "__main__":
    mcp.run()

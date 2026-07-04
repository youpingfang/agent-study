#!/usr/bin/env uv run

import os
import json
from dotenv import load_dotenv

load_dotenv()

# ===== 用 init_chat_model 管理模型配置（方便切换厂商）=====
from langchain.chat_models import init_chat_model

chat_model = init_chat_model(
    model="deepseek-chat",
    temperature=0.7,
    api_key=os.getenv("Deepseek_API_KEY"),
    model_provider="deepseek",
)

# ===== 从 chat_model 中获取底层 OpenAI client =====
# chat_model（ChatDeepSeek）底层封装了一个 OpenAI 兼容的 client
# 直接拿来用，不用自己 new OpenAI()
client = chat_model.client

# ⚠️ chat_model.client 直接是 openai.Completions 对象（不是 OpenAI 根对象）
#    所以调用是 client.create(...) 而不是 client.chat.completions.create(...)

# ===== 构建对话（LangChain Messages）=====
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    convert_to_openai_messages,
)

lc_messages = [
    SystemMessage(content="你是一个科技平台的搜索助手。"),
    HumanMessage(content="请帮我查询今天 IT 之家的前面 10 条数据。"),
]

# ===== 转成 OpenAI 格式 =====
openai_msgs = convert_to_openai_messages(lc_messages)

# ===== 第 1 次调用：用 client.create（不走 invoke，避免 tools 兼容问题）=====
# ⚠️ 注意：chat_model.invoke(messages, tools=[...]) 会报
#    "tools[0]: missing field 'type'"，所以这里直接调底层的 client
response = client.create(
    model="deepseek-v4-flash",
    messages=openai_msgs,
    tools=[{
        "type": "function",
        "function": {
            "name": "tavily_search",
            "description": "搜索互联网获取最新信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词"},
                },
                "required": ["query"],
            },
        },
    }],
    tool_choice="auto",
    temperature=0.7,
)

assistant_msg = response.choices[0].message
openai_msgs.append(assistant_msg.model_dump())

# ===== 执行 Tavily 搜索工具 =====
if assistant_msg.tool_calls:
    from langchain_tavily import TavilySearch

    tavily_tool = TavilySearch(
        max_results=10,
        api_key=os.getenv("TAVILY_API_KEY"),
    )

    for tc in assistant_msg.tool_calls:
        args = json.loads(tc.function.arguments)
        print(f"🔧 调用工具: {tc.function.name}, 参数: {args}")

        tool_result = tavily_tool.invoke(args)

        openai_msgs.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": json.dumps(tool_result, ensure_ascii=False),
        })

    # ===== 第 2 次调用：同样用 client.create =====
    response2 = client.create(
        model="deepseek-v4-flash",
        messages=openai_msgs,
        temperature=0.7,
    )
    final = response2.choices[0].message.content
    print(f"\n🤖 最终回复:\n{final}")
else:
    print(f"\n🤖 回复:\n{assistant_msg.content}")

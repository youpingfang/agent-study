#!/usr/bin/env -S uv run
"""
LangChain Messages 实战 Demo 2：对接 DeepSeek API
===============================================

与 agent_messages_demo.py（纯演示消息对象）不同，这个文件：
  - 真正调用了 DeepSeek API
  - 用 init_chat_model 创建模型实例
  - 演示了 4 个实战场景：普通对话、多轮对话、工具调用、流式输出
"""

import os
import json
from dotenv import load_dotenv
from pathlib import Path

# ===========================================================================
# 第零步：加载环境变量
# ===========================================================================
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

# ===========================================================================
# 第一步：导入消息类型（和 demo1 一样）
# ===========================================================================
from langchain_core.messages import (
    SystemMessage,   # {"role": "system", "content": "..."}
    HumanMessage,    # {"role": "user", "content": "..."}
    AIMessage,       # {"role": "assistant", "content": "..."}
    ToolMessage,     # {"role": "tool", "content": "...", "tool_call_id": "..."}
)

# ===========================================================================
# 第二步：用 init_chat_model 创建 DeepSeek 模型实例
# ===========================================================================
# init_chat_model 是 LangChain 推荐的统一入口：
#   它封装了不同厂商的 API 差异，只需指定 model_provider 即可切换
from langchain.chat_models import init_chat_model

model = init_chat_model(
    model="deepseek-chat",                  # DeepSeek 模型名
    temperature=0.7,                         # 回答随机性（0=确定, 1=创意）
    api_key=os.getenv("Deepseek_API_KEY"),   # 从 .env 读取 Key
    model_provider="deepseek",               # 指定厂商为 DeepSeek
    # init_chat_model 会自动匹配正确的 base_url，无需手动写
    # 底层映射：model_provider="deepseek" → https://api.deepseek.com/v1
)


# ===========================================================================
# 辅助函数
# ===========================================================================
def separator(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ===========================================================================
# Demo 1：最简单的一次性对话
# ===========================================================================
def demo_01_simple_chat():
    """
    发送一条 SystemMessage + 一条 HumanMessage，拿到 AIMessage。

    关键概念：
      model.invoke(messages) → 返回一条 AIMessage
      invoke 是 LangChain 最基础的调用方式，给定输入，返回完整输出
    """
    separator("Demo 1：一次性对话（invoke）")

    messages = [
        # SystemMessage 在最前面，定义 AI 的行为
        SystemMessage(content="你是一个 Linux 专家，回答简洁专业，尽量用命令示例说明。"),
        # HumanMessage 是用户的实际问题
        HumanMessage(content="如何查看当前目录下占用空间最大的 5 个文件？"),
    ]

    # invoke()：发送消息列表，等待并返回完整回复
    response: AIMessage = model.invoke(messages)

    print(f"🤖 DeepSeek 的回复：\n")
    print(response.content)
    print()
    print(f"📊 元数据：")
    print(f"   model:       {response.response_metadata.get('model_name', 'N/A')}")
    print(f"   finish_reason: {response.response_metadata.get('finish_reason', 'N/A')}")
    print(f"   输入 token:    {response.usage_metadata.get('input_tokens', 'N/A')}")
    print(f"   输出 token:    {response.usage_metadata.get('output_tokens', 'N/A')}")


# ===========================================================================
# Demo 2：多轮对话（用 Messages 维护上下文）
# ===========================================================================
def demo_02_multi_turn():
    """
    多轮对话的核心：把每轮的 AIMessage 追加到消息列表，再传给下一次 invoke。

    关键概念：
      对话历史 = messages 列表
      每次调用前把新的 HumanMessage 追加进去
      每次调用后把返回的 AIMessage 追加进去
      这样 AI 就"记得"之前说了什么
    """
    separator("Demo 2：多轮对话（上下文管理）")

    # 对话初始状态：只有一条 SystemMessage
    messages = [
        SystemMessage(content="你是一个 Python 编程老师，回答要循序渐进。"),
    ]

    # ===== 第 1 轮 =====
    messages.append(HumanMessage(content="Python 中列表和元组有什么区别？"))
    response = model.invoke(messages)
    messages.append(response)  # ← 关键：把 AI 的回复追加进历史

    print("第 1 轮：")
    print(f"  👤: 列表和元组有什么区别？")
    print(f"  🤖: {response.content[:80]}...")
    print()

    # ===== 第 2 轮 =====
    # AI 记得上一轮的内容，所以能理解"它"指的是元组
    messages.append(HumanMessage(content="那什么时候应该用它而不是列表？"))
    response = model.invoke(messages)
    messages.append(response)

    print("第 2 轮（AI 记得上文）：")
    print(f"  👤: 那什么时候应该用它而不是列表？")
    print(f"  🤖: {response.content[:120]}...")
    print()

    print(f"📊 当前对话历史共 {len(messages)} 条消息")
    for i, m in enumerate(messages):
        role = {"system": "⚙️", "human": "👤", "ai": "🤖"}.get(m.type, "  ")
        c = m.content[:60].replace("\n", " ") + "..."
        print(f"  [{i}] {role} {c}")


# ===========================================================================
# Demo 3：绑定工具（Tool Call / Function Calling）
# ===========================================================================
def demo_03_tool_binding():
    """
    演示最关键的功能：给模型绑定工具，让它能调用外部函数。

    流程：
      1. 定义工具的 schema（名称、描述、参数格式）
      2. 用 openai SDK 直连 DeepSeek（不用 LangChain 的 bind_tools，兼容性问题）
      3. 发送用户消息
      4. 模型返回 tool_calls
      5. 手动执行工具，把结果用 tool role 返回
      6. 再次调用模型，让它基于工具结果给出最终回答

    ⚠️ DeepSeek 对 LangChain 的 invoke(tools=...) / bind_tools() 兼容不好，
       所以这里回归最原始的做法：用 openai SDK + convert_to_openai_messages。
    """
    separator("Demo 3：工具调用（Tool Call / Function Calling）")

    # ---- 步骤 1：用 openai SDK 直连 DeepSeek ----
    from openai import OpenAI
    from openai.types.chat import ChatCompletionToolParam

    client = OpenAI(
        api_key=os.getenv("Deepseek_API_KEY"),
        base_url="https://api.deepseek.com/v1",
    )

    # ---- 步骤 2：定义工具的 JSON Schema ----
    TOOLS: list[ChatCompletionToolParam] = [
        {
            "type": "function",
            "function": {
                "name": "get_current_time",
                "description": "获取指定时区的当前时间",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "timezone": {
                            "type": "string",
                            "description": "时区名称，如 Asia/Shanghai、America/New_York",
                        }
                    },
                    "required": ["timezone"],
                },
            },
        },
    ]

    # ---- 步骤 3：用 LangChain messages 构建对话，再转成 OpenAI 格式 ----
    from langchain_core.messages import convert_to_openai_messages

    lc_messages = [
        SystemMessage(content="你是一个时间助手。用户问时间就调 get_current_time 工具。"),
        HumanMessage(content="现在北京时间几点？"),
    ]

    openai_msgs = convert_to_openai_messages(lc_messages)

    # ---- 步骤 4：第一次调用 —— 模型决定调用工具 ----
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=openai_msgs,
        tools=TOOLS,
        tool_choice="auto",
        temperature=0.7,
    )

    assistant_msg = response.choices[0].message
    openai_msgs.append(assistant_msg.model_dump())

    print("第一次调用 —— 模型返回 tool_calls：")
    print(f"  content:     '{assistant_msg.content}'")
    print(f"  tool_calls:  {len(assistant_msg.tool_calls or [])} 个")
    for tc in (assistant_msg.tool_calls or []):
        print(f"    → 工具名: {tc.function.name}")
        print(f"    → 参数:   {tc.function.arguments}")
        print(f"    → ID:     {tc.id}")
    print()

    # ---- 步骤 5：手动执行工具并返回结果 ----
    from datetime import datetime
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for tc in (assistant_msg.tool_calls or []):
        openai_msgs.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": json.dumps({
                "timezone": json.loads(tc.function.arguments)["timezone"],
                "current_time": current_time,
                "timezone_offset": "+08:00",
            }, ensure_ascii=False),
        })
        print(f"工具 {tc.function.name} 执行结果：{current_time}")

    # ---- 步骤 6：第二次调用 —— 模型根据工具结果给出最终回答 ----
    response2 = client.chat.completions.create(
        model="deepseek-chat",
        messages=openai_msgs,
        tools=TOOLS,
        temperature=0.7,
    )

    final = response2.choices[0].message.content

    print(f"二次调用 —— 基于工具结果的最终回答：")
    print(f"  🤖: {final}")
    print()
    print(f"📊 整个对话的消息流转（原生 OpenAI 格式）：")
    for i, m in enumerate(openai_msgs):
        role = m.get("role", "?")
        msg_preview = str(m.get("content", ""))[:60].replace("\n", " ")
        if m.get("tool_calls"):
            msg_preview = f"[调用工具: {[tc['function']['name'] for tc in m['tool_calls']]}]"
        print(f"  [{i}] {role:10s} {msg_preview}...")


def demo_04_streaming():
    """
    流式输出：一句话一句话地返回，而不是等全写完再返回。

    关键概念：
      model.stream(messages) → 返回一个迭代器，每个迭代是一段增量文本
      用户体验更好（像 ChatGPT 那样逐字输出）

    注意：
      - stream 每次只返回 delta（增量），不是完整内容
      - token_usage 信息在最后一个 chunk 里
      - 有 tool_calls 时，流式返回可能不完整，建议用 invoke
    """
    separator("Demo 4：流式输出（stream）")

    messages = [
        SystemMessage(content="你是一个简洁的助手，回答问题不超过 3 句话。"),
        HumanMessage(content="解释一下什么是递归，给一个简短的 Python 例子"),
    ]

    print("🤖 DeepSeek 流式回复（逐字输出效果）：\n")
    full_content = ""
    for chunk in model.stream(messages):
        # chunk 是 AIMessageChunk，.content 是这一小段增量文本
        if chunk.content:
            print(chunk.content, end="", flush=True)
            full_content += str(chunk.content)

    print(f"\n\n📊 完整回复共 {len(full_content)} 个字符")


# ===========================================================================
# Demo 5：对比 —— 同一条消息，有 SystemMessage vs 没有
# ===========================================================================
def demo_05_system_message_effect():
    """
    对比 SystemMessage 的实际效果：
      同样的问题，有/没有 SystemMessage 给出完全不同的风格

    这展示了 SystemMessage 的核心价值：
      它不是装饰品，而是切实控制 AI 的输出风格和范围
    """
    separator("Demo 5：SystemMessage 的实际影响对比")

    question = "解释一下什么是闭包"

    # 情况 A：没有 SystemMessage
    messages_a = [HumanMessage(content=question)]
    response_a = model.invoke(messages_a)

    # 情况 B：有 SystemMessage，要求用大白话解释
    messages_b = [
        SystemMessage(content="用幼儿园小朋友都能听懂的大白话解释，不要用任何术语，50 字以内。"),
        HumanMessage(content=question),
    ]
    response_b = model.invoke(messages_b)

    print("同一个问题的两种回答风格：\n")
    print(f"❌ 无 SystemMessage（默认风格）：")
    print(f"   {response_a.content[:200]}...")
    print()
    print(f"✅ 有 SystemMessage（大白话风格）：")
    print(f"   {response_b.content}")
    print()
    print("结论：SystemMessage 对 AI 行为的影响是非常显著的。")


# ===========================================================================
# 运行所有 Demo
# ===========================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  LangChain Messages 实战：对接 DeepSeek API")
    print("=" * 60)
    print()
    print("model = init_chat_model(")
    print("    model='deepseek-chat',")
    print("    model_provider='deepseek',")
    print("    api_key=os.getenv('Deepseek_API_KEY'),")
    print(")")
    print()

    demo_01_simple_chat()
    demo_02_multi_turn()
    demo_03_tool_binding()
    demo_04_streaming()
    demo_05_system_message_effect()

    print("\n" + "=" * 60)
    print("  🎉 全部 Demo 完成！")
    print("=" * 60)

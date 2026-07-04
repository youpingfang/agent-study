#!/usr/bin/env -S uv run
"""
LangChain Messages 全面演示
==========================

这个文件详细展示 LangChain 中所有消息类型的用法，
以及它们如何映射到 OpenAI 风格的 API 调用。

LangChain Messages 的核心价值：
  不直接拼接字符串对话，而是用结构化对象管理对话历史，
  让 LLM、Agent、Tool 之间的通信变得清晰可控。

消息类型一览：
  1. SystemMessage  — 系统指令（角色设定）
  2. HumanMessage   — 用户输入
  3. AIMessage      — AI 回复（含 tool_calls）
  4. ToolMessage    — 工具执行结果
"""

import json
from typing import Any

# ===========================================================================
# 第一部分：消息类型导入
# ===========================================================================
# 这四种是核心消息类型，LangChain 的对话管理都围绕它们展开
from langchain_core.messages import (
    SystemMessage,   # 系统提示词，相当于 OpenAI 的 {"role": "system"}
    HumanMessage,    # 用户说的话，相当于 {"role": "user"}
    AIMessage,       # AI 的回复，相当于 {"role": "assistant"}
    ToolMessage,     # 工具调用的结果，相当于 {"role": "tool"}
)


def separator(title: str = "") -> None:
    """辅助函数：打印分隔线，让输出更好看"""
    if title:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")
    else:
        print("-" * 60)


# ===========================================================================
# 第二部分：每种消息类型的详解
# ===========================================================================

def demo_01_system_message():
    """
    SystemMessage：设定 AI 的行为准则和角色

    核心用途：
      - 定义 AI 的身份（"你是一个...助手"）
      - 设置行为约束（"不要回答...只回答..."）
      - 提供格式要求（"用 JSON 回复"）

    OpenAI 映射：{"role": "system", "content": "..."}

    最佳实践：
      ⚠️ 只放一条 SystemMessage，放在 messages 列表最前面
      ⚠️ 不要在 SystemMessage 里写动态数据（上下文、记忆等），
          那些应该放在 HumanMessage 里
    """
    separator("第一：SystemMessage — 系统指令")

    msg = SystemMessage(
        content=(
            "你是一个 Python 学习助手。请用通俗易懂的方式解释概念，"
            "尽量结合生活实例，不要使用过于抽象的术语。"
            "如果用户问的问题你完全不了解，直接说不知道，不要编造。"
        )
    )

    print("SystemMessage 对象：")
    print(f"  type:        {msg.type}")           # "system"
    print(f"  content:     {msg.content[:50]}...") # 前 50 个字符
    print(f"  response_metadata: {msg.response_metadata}")
    print()

    # 转换成 OpenAI 消息格式
    openai_msg = msg.model_dump()
    print("转换为 OpenAI 格式 (model_dump)：")
    print(f"  {openai_msg}")
    print(f"\n  等价于: {{'role': 'system', 'content': '...'}}")


def demo_02_human_message():
    """
    HumanMessage：代表用户的输入

    核心用途：
      - 用户的原始问题
      - 作为 Function Calling 中，"用户说要用哪个工具"的起点

    OpenAI 映射：{"role": "user", "content": "..."}

    最佳实践：
      ⚠️ HumanMessage 也可以放图片（多模态），但这不在此演示范围
      ⚠️ content 可以是纯文本字符串，也可以是复杂的内容块列表
    """
    separator("第二：HumanMessage — 用户输入")

    # 纯文本
    msg1 = HumanMessage(content="什么是闭包？给我一个简单的例子")

    # content 是一段文本
    msg2 = HumanMessage(content="帮我查一下上海的天气")

    print("HumanMessage 对象：")
    print(f"  消息1 — type: {msg1.type}")
    print(f"  消息1 — content: {msg1.content}")
    print(f"  消息1 — id: {msg1.id or '未设置(None)'}")  # LangChain 可能不自动生成 ID
    print()
    print(f"  消息2 — content: {msg2.content}")
    print()


def demo_03_ai_message_plain():
    """
    AIMessage：AI 的回复（普通文本）

    核心用途：
      - 记录 AI 的纯文本回复
      - 在对话历史中，AIMessage 后面通常跟着 HumanMessage（多轮对话）

    OpenAI 映射：{"role": "assistant", "content": "..."}

    最佳实践：
      ⚠️ 对话历史必须是交替的：Human → AI → Human → AI → ...
      ⚠️ AIMessage 可能不包含 content（仅 tool_calls 时）
    """
    separator("第三：AIMessage — AI 的普通回复")

    msg = AIMessage(content="闭包就是一个函数记住了它出生时的环境...")

    print("AIMessage 对象（纯文本回复）：")
    print(f"  type:          {msg.type}")
    print(f"  content:       {msg.content[:40]}...")
    print(f"  tool_calls:    {msg.tool_calls}")
    print(f"  usage_metadata: {msg.usage_metadata}")
    print()

    # 多轮对话示例
    print(">>> 多轮对话历史的组成：")
    conversation = [
        HumanMessage(content="什么是递归？"),
        AIMessage(content="递归就是函数调用自己..."),
        HumanMessage(content="给我一个 Python 例子"),
        AIMessage(content="def factorial(n): return 1 if n<=1 else n*factorial(n-1)"),
    ]
    for i, m in enumerate(conversation):
        role = "👤 用户" if isinstance(m, HumanMessage) else "🤖 AI"
        print(f"  [{i}] {role}: {m.content[:50]}...")
    print()


def demo_04_ai_message_with_tools():
    """
    AIMessage（含 tool_calls）：AI 决定调用工具

    这是 Agent 的核心！当 LLM 判断需要调用工具时，返回的 AIMessage 包含
    tool_calls 字段而不是普通的 content。

    OpenAI 映射：
      {"role": "assistant", "content": null, "tool_calls": [{...}]}

    注意：tool_calls 可能有多条（并行调用多个工具）
    """
    separator("AIMessage — 含工具调用（Agent 的关键）")

    # 模拟 LLM 返回一个带 tool_calls 的 AIMessage
    msg = AIMessage(
        content="",  # 可能有内容也可能为空
        tool_calls=[
            {
                "name": "get_weather",      # 工具名
                "args": {"city": "Beijing"}, # 工具参数
                "id": "call_abc123",        # 调用 ID（ToolMessage 需要用它匹配）
                "type": "tool_call",
            }
        ],
    )

    print("AIMessage 对象（含 tool_calls）：")
    print(f"  type:        {msg.type}")
    print(f"  content:     '{msg.content}' ← 调用工具时可以没有文本")
    print(f"  tool_calls:")
    for tc in msg.tool_calls:
        print(f"    - name: {tc['name']}")
        print(f"    - args: {tc['args']}")
        print(f"    - id:   {tc['id']}")
    print()

    # 多工具并行调用示例
    msg_multi = AIMessage(
        content="",
        tool_calls=[
            {"name": "get_weather", "args": {"city": "Beijing"}, "id": "call_1", "type": "tool_call"},
            {"name": "get_weather", "args": {"city": "Shanghai"}, "id": "call_2", "type": "tool_call"},
        ],
    )
    print(">>> 并行调用多个工具的 AIMessage：")
    for tc in msg_multi.tool_calls:
        print(f"  同时调用: {tc['name']}({tc['args']})")
    print()


def demo_05_tool_message():
    """
    ToolMessage：工具的执行结果

    当工具执行完毕后，结果通过 ToolMessage 返回给 LLM。

    OpenAI 映射：{"role": "tool", "content": "...", "tool_call_id": "..."}

    关键字段：
      - content: 工具执行的结果（字符串）
      - tool_call_id: 必须匹配 AIMessage 中的 tool_call.id
      - name: 工具名（可选但推荐）

    最佳实践：
      ⚠️ tool_call_id 必须精确匹配，否则 LLM 不知道这个结果对应哪个调用
      ⚠️ 如果多个工具并行调用，就要有多个 ToolMessage
      ⚠️ content 建议返回 JSON 字符串，方便 LLM 解析
    """
    separator("ToolMessage — 工具执行结果")

    # 模拟工具执行的返回结果
    tool_msg = ToolMessage(
        content=(
            '{"城市":"Beijing","天气":"晴","温度":"35°C","湿度":"40%","风速":"3m/s"}'
        ),
        tool_call_id="call_abc123",  # 必须匹配 AIMessage 中的 id
        name="get_weather",          # 工具名
    )

    print("ToolMessage 对象：")
    print(f"  type:          {tool_msg.type}")
    print(f"  content:       {json.loads(tool_msg.content)}") # 解析 JSON 看看
    print(f"  tool_call_id:  {tool_msg.tool_call_id}")
    print(f"  name:          {tool_msg.name}")
    print()

    # 展示完整的 Agent 循环（AIMessage → ToolMessage）
    print(">>> 完整的 Agent 工具调用流程：")
    print()
    agent_flow = [
        HumanMessage(content="北京现在天气怎么样？"),
        AIMessage(
            content="",
            tool_calls=[{
                "name": "get_weather",
                "args": {"city": "Beijing"},
                "id": "call_abc123",
                "type": "tool_call",
            }],
        ),
        ToolMessage(
            content='{"城市":"Beijing","天气":"晴","温度":"35°C"}',
            tool_call_id="call_abc123",
            name="get_weather",
        ),
        AIMessage(content="北京现在是晴天，温度 35°C，注意防晒哦！"),
    ]
    for i, m in enumerate(agent_flow):
        if isinstance(m, HumanMessage):
            print(f"  [{i}] 👤 用户: {m.content}")
        elif isinstance(m, ToolMessage):
            print(f"  [{i}] 🔧 工具结果: {m.content}")
        elif isinstance(m, AIMessage):
            if m.tool_calls:
                for tc in m.tool_calls:
                    print(f"  [{i}] 🤖 AI 决定调用: {tc['name']}({tc['args']})")
            else:
                print(f"  [{i}] 🤖 AI 回复: {m.content}")
    print()


# ===========================================================================
# 第三部分：消息转换 — LangChain ↔ OpenAI
# ===========================================================================

def demo_06_convert_to_openai():
    """
    将 LangChain Messages 转换为 OpenAI API 格式

    这是实际开发中最重要的操作之一。
    LangChain 自己管理消息，调用 OpenAI 时需要转换格式。
    """
    separator("消息转换 — LangChain → OpenAI 格式")

    # 构造一组 LangChain 消息
    lc_messages = [
        SystemMessage(content="你是天气助手，只回答天气问题"),
        HumanMessage(content="北京天气怎么样"),
        AIMessage(
            content="",
            tool_calls=[{
                "name": "get_weather",
                "args": {"city": "Beijing"},
                "id": "call_123",
                "type": "tool_call",
            }],
        ),
        ToolMessage(
            content='{"温度":"35°C","天气":"晴"}',
            tool_call_id="call_123",
        ),
    ]

    # 使用 langchain_core 自带的方法转换
    from langchain_core.messages.utils import convert_to_messages
    # convert_to_openai_messages 是推荐的转换方式
    from langchain_core.messages import convert_to_openai_messages

    openai_messages = convert_to_openai_messages(lc_messages)

    print("LangChain Messages → OpenAI 格式：\n")
    for i, (lc, o) in enumerate(zip(lc_messages, openai_messages)):
        role_map = {
            "system": "SystemMessage",
            "human": "HumanMessage",
            "assistant": "AIMessage",
            "tool": "ToolMessage",
        }
        print(f"  [{i}] {role_map.get(o['role'], o['role'])} → role='{o['role']}'")

    print()

    # 转换后的 OpenAI 格式可以直接传给 openai 库
    print(">>> 转换后的完整 OpenAI 消息列表：")
    for msg in openai_messages:
        # 简化输出，只显示关键字段
        simplified = {"role": msg["role"]}
        if msg.get("content"):
            simplified["content"] = msg["content"][:40] + ("..." if len(str(msg.get("content", ""))) > 40 else "")
        else:
            simplified["content"] = None
        if msg.get("tool_calls"):
            simplified["tool_calls"] = [tc["function"]["name"] for tc in msg["tool_calls"]]
        if msg.get("tool_call_id"):
            simplified["tool_call_id"] = msg["tool_call_id"]
        print(f"  {simplified}")
    print()


# ===========================================================================
# 第四部分：对话历史管理
# ===========================================================================

def demo_07_conversation_history():
    """
    对话历史管理的最佳实践

    关键要点：
      1. 历史是 LangChain Messages 列表
      2. SystemMessage 固定在最前面
      3. 多轮对话按 Human → AI → Human → AI 交替
      4. 工具调用是 Human → AIMessage(tool_calls) → ToolMessage → AIMessage
      5. 可以用 trim_messages 截断过长历史（控制 token 消耗）
    """
    separator("对话历史管理")

    # ===== 示例 1：构建多轮对话 =====
    print(">>> 示例 1：构建完整的多轮对话历史\n")

    history: list[Any] = [
        SystemMessage(content="你是 Python 专家"),
        HumanMessage(content="什么是装饰器？"),
        AIMessage(content="装饰器是一个接受函数作为参数并返回新函数的函数..."),
        HumanMessage(content="给我一个计时的装饰器例子"),
        AIMessage(content="import time\ndef timer(func):\n    def wrapper(*args, **kwargs):\n        start = time.time()\n        result = func(*args, **kwargs)\n        print(f'耗时{time.time()-start}s')\n        return result\n    return wrapper"),
        HumanMessage(content="再给一个缓存结果的装饰器"),
        # AI 回答...
    ]

    print(f"  当前对话共有 {len(history)} 条消息")
    for i, m in enumerate(history):
        role = {
            "system": "⚙️ 系统",
            "human": "👤 用户",
            "ai": "🤖 AI",
        }.get(m.type, f"  {m.type}")
        content_preview = m.content[:55].replace("\n", " ") + "..."

        # 当 role 是 system 时，缩短 content 显示
        if m.type == "system":
            content_preview = m.content[:40].replace("\n", " ") + "..."
        print(f"  [{i}] {role}: {content_preview}")
    print()

    # ===== 示例 2：使用 trim_messages 控制长度 =====
    from langchain_core.messages import trim_messages

    long_history: list[Any] = [
        SystemMessage(content="你是助手"),
        HumanMessage(content="第 1 个问题"),
        AIMessage(content="第 1 个回答"),
        HumanMessage(content="第 2 个问题"),
        AIMessage(content="第 2 个回答"),
        HumanMessage(content="第 3 个问题"),
        AIMessage(content="第 3 个回答"),
        HumanMessage(content="第 4 个问题"),
        AIMessage(content="第 4 个回答"),
        HumanMessage(content="第 5 个问题"),
    ]

    # trim_messages 从旧消息开始裁剪，保留最近的 token_count 个 token
    trimmed = trim_messages(
        long_history,
        max_tokens=80,         # 最多保留 80 个 token
        strategy="last",       # 保留最后的消息
        token_counter=len,     # 用字符长度模拟 token 计数（实际项目用 tiktoken）
        include_system=True,   # 始终保留 SystemMessage
        start_on="human",      # 确保第一条是 HumanMessage（成对裁剪）
        allow_partial=False,   # 不保留被部分截断的消息
    )

    print(">>> 示例 2：裁剪后的对话历史（原 10 条 → 裁剪后）")
    for i, m in enumerate(trimmed):
        print(f"  [{i}] {m.type}: {m.content}")
    print()


# ===========================================================================
# 第五部分：常见模式和最佳实践
# ===========================================================================

def demo_08_best_practices():
    """
    总结：LangChain Messages 的最佳实践

    这是一份速查表，覆盖了实际开发中最常见的需求和陷阱。
    """
    separator("最佳实践速查表")

    tips = [
        ("SystemMessage 放最开始",
         "messages = [SystemMessage(...), HumanMessage(...)]",
         "不要中途插入 SystemMessage，LLM 对位置有预期"),

        ("content 为空 vs 不为空",
         "tool_calls 的 AIMessage 可以 content=''",
         "但有的模型要求 tool_calls 时 content 不能为空字符串，用 None 或省略"),

        ("tool_call_id 必须匹配",
         "ToolMessage 的 tool_call_id 必须和 AIMessage 中的 id 一致",
         "否则 LLM 不知道这个结果对应哪个调用"),

        ("多轮对话保持交替",
         "Human → AI → Human → AI，不允许两个 HumanMessage 相邻",
         "两个 HumanMessage 紧邻时，第二个会被忽略或报错"),

        ("使用 langchain_core.messages 的工厂方法",
         "不用自己拼字典，用 SystemMessage(content=...) 之类",
         "LangChain 帮你处理 id 生成、类型标记等"),

        ("用 trim_messages 控制 token 消耗",
         "过长对话会超出 LLM 上下文窗口，trim_messages 自动截掉旧消息",
         "保留最新几轮对话 + SystemMessage"),

        ("Tool 结果放 ToolMessage，不要放 HumanMessage",
         "有人会把工具结果塞到 HumanMessage.content 里，虽然能工作但不规范",
         "AI 无法区分这是用户说的还是工具返回的"),

        ("使用 convert_to_openai_messages 转换",
         "LangChain 的官方转换函数，处理边界情况更完善",
         "不要自己手动遍历 messages 拼字典"),
    ]

    for title, good, bad in tips:
        print(f"  ✅ {title}")
        print(f"     正确: {good}")
        print(f"     原因: {bad}")
        print()


# ===========================================================================
# 第六部分：运行所有演示
# ===========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  LangChain Messages 完全指南")
    print("=" * 60)
    print()
    print("本文档覆盖以下内容：")
    print("  1. 四种核心消息类型（System/Human/AI/Tool）")
    print("  2. LangChain ↔ OpenAI 格式转换")
    print("  3. 对话历史管理（构建 + 裁剪）")
    print("  4. 最佳实践速查表")
    print()

    # 按顺序运行所有演示
    demo_01_system_message()
    demo_02_human_message()
    demo_03_ai_message_plain()
    demo_04_ai_message_with_tools()
    demo_05_tool_message()
    demo_06_convert_to_openai()
    demo_07_conversation_history()
    demo_08_best_practices()

    print("=" * 60)
    print("  🎉 演示完成！")
    print("=" * 60)

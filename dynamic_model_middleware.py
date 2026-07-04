"""
动态模型选择中间件 — pre_model_hook 使用示例
=============================================

create_react_agent 支持两个 hook 参数：
  - pre_model_hook:  模型调用前执行，接收 messages，返回修改后的 messages
  - post_model_hook: 模型调用后执行，接收 messages，返回修改后的 messages

典型场景：
  1. 根据问题类型自动切换模型（简单用便宜模型，复杂用贵模型）
  2. 在每次调用前注入上下文或记忆（当前时间、用户信息等）
  3. 记录调用日志或监控 token 消耗
  4. 动态注入 SystemMessage
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# 1. 基础导入
# ============================================================
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent


# 一个简单的工具
@tool
def get_weather(city: str) -> str:
    """获取城市天气"""
    return f"{city}：晴天，25°C"


# ============================================================
# 2. 定义多个模型
# ============================================================
cheap_model = init_chat_model(
    model="deepseek-chat",
    model_provider="deepseek",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)

expensive_model = init_chat_model(
    model="deepseek-chat",      # 实际项目这里换成更强的模型
    model_provider="deepseek",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)


# ============================================================
# 3. pre_model_hook：每次模型调用前执行
# ============================================================
# 函数签名：def hook(messages: list) -> list:
# 接收当前消息列表，返回修改后的消息列表

def inject_system_prompt(messages: list) -> list:
    """
    每次模型调用前，在最前面插入 SystemMessage。
    常用于注入当前时间、用户偏好等上下文。
    """
    from datetime import datetime
    messages = list(messages)
    messages.insert(0, SystemMessage(
        content=f"当前时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}。"
                f"你是一个知识渊博的助手。"
    ))
    return messages


def dynamic_model_selector(messages: list) -> list:
    """
    ⚠️ 请注意：pre_model_hook 不能切换模型！
    它只能修改 messages 列表。
    如果要切换模型，需要在 create_react_agent 外部控制。

    这里演示的是：在 model 调用前注入不同的 SystemMessage，
    来"模拟"不同模型的行为。
    """
    messages = list(messages)

    # 看最后一条 HumanMessage 的长度
    for msg in reversed(messages):
        if msg.type == "human" and isinstance(msg.content, str):
            if len(msg.content) > 20:
                # 复杂问题 → 注入更详细的系统提示
                hint = "你是一个高级分析助手，请给出详细、全面的回答。"
                print("🔀 复杂问题模式（注入详细提示）")
            else:
                # 简单问题 → 简洁回答
                hint = "你是一个简洁的助手，请用一句话回答。"
                print("🔀 简单问题模式（注入简洁提示）")

            # 在最前面插入 SystemMessage
            messages.insert(0, SystemMessage(content=hint))
            break

    # 工具结果后 → 正常模式
    if messages and messages[-1].type == "tool":
        # 不额外加提示，保持默认
        print("🔧 工具结果后（默认模式）")

    return messages


# ============================================================
# 4. post_model_hook：每次模型调用后执行
# ============================================================

def log_model_call(messages: list) -> list:
    """
    模型调用后执行，可以记录日志或修改 AI 回复。
    这里只做日志记录。
    """
    # 最后一条消息应该是 AI 的回复
    if messages:
        last = messages[-1]
        if last.type == "ai":
            tokens = getattr(last, "usage_metadata", {})
            print(f"📊 AI 回复完成 | token: {tokens}")
        elif hasattr(last, 'tool_calls') and last.tool_calls:
            print(f"🔧 AI 决定调工具: {[tc['name'] for tc in last.tool_calls]}")
    return messages


# ============================================================
# 5. 创建带 hook 的 Agent
# ============================================================
agent_with_hooks = create_react_agent(
    model=cheap_model,
    tools=[get_weather],
    pre_model_hook=dynamic_model_selector,   # 调用前
    post_model_hook=log_model_call,          # 调用后
)


# ============================================================
# 6. 真正的动态模型切换方案
# ============================================================
# pre_model_hook 不能改模型，所以要动态切换模型的话，
# 需要自己在外面控制。这里演示最直接的方式：

def invoke_with_model_selection(question: str, model=cheap_model) -> dict:
    """
    根据问题选择模型，然后调用 agent。
    这才是真正的"动态模型选择"。
    """
    # 规则：超过 30 个字用贵模型，否则用便宜的
    chosen = expensive_model if len(question) > 30 else cheap_model
    print(f"🔀 选择模型: {'💰 贵的' if chosen == expensive_model else '🪙 便宜的'}")

    # 每次都新建 agent（绑定不同的 model）
    agent = create_react_agent(
        model=chosen,
        tools=[get_weather],
    )
    return agent.invoke({
        "messages": [HumanMessage(content=question)]
    })


# ============================================================
# 7. 运行测试
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("测试 1：pre_model_hook — 简单问题")
    print("=" * 60)
    result = agent_with_hooks.invoke({
        "messages": [HumanMessage(content="天气怎么样？")]
    })
    print(f"  回答: {result['messages'][-1].content[:80]}...\n")

    print("=" * 60)
    print("测试 2：pre_model_hook — 复杂问题")
    print("=" * 60)
    result = agent_with_hooks.invoke({
        "messages": [
            HumanMessage(
                content="请详细分析一下北京过去一周的天气变化趋势，"
                        "并给出出行建议，同时对比上海和广州的天气差异"
            )
        ]
    })
    print(f"  回答: {result['messages'][-1].content[:80]}...\n")

    print("=" * 60)
    print("测试 3：真正的动态模型选择")
    print("=" * 60)
    result = invoke_with_model_selection("北京天气怎么样？")
    print(f"  回答: {result['messages'][-1].content[:80]}...\n")

    print("=" * 60)
    print("测试 4：真正动态模型选择（长问题 → 贵模型）")
    print("=" * 60)
    result = invoke_with_model_selection(
        "请详细分析北京过去一周的天气变化趋势并给出出行建议"
    )
    print(f"  回答: {result['messages'][-1].content[:80]}...")

    print("\n" + "=" * 60)
    print("总结：")
    print("  1. pre_model_hook — 修改 messages（注入提示、上下文）")
    print("  2. post_model_hook — 记录日志、后处理")
    print("  3. 动态换模型 — 在外面控制 model 参数")
    print("=" * 60)


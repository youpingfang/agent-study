"""
动态模型选择中间件 — pre/post_model_hook 使用示例
================================================

create_react_agent 支持两个 hook 参数：
  - pre_model_hook:  模型调用前执行，可修改 messages
  - post_model_hook: 模型调用后执行，可修改 messages

注意：这两个 hook 只能修改 messages，不能切换模型。
真正的"动态模型选择"需要在外部控制。
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
# 2. 定义模型
# ============================================================
cheap_model = init_chat_model(
    model="deepseek-chat",
    model_provider="deepseek",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)


# ============================================================
# 3. pre_model_hook：模型调用前执行
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


def dynamic_prompt_selector(state: dict) -> dict:
    """
    根据问题复杂度，注入不同的 SystemMessage。
    pre/post_model_hook 接收和返回的都是状态字典，
    必须返回 {"messages": [...], "llm_input_messages": [...]} 等。
    """
    messages = list(state.get("messages", []))

    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str) and len(content) > 20:
                hint = "你是一个高级分析助手，请给出详细、全面的回答。"
                print("🔀 复杂问题模式（注入详细提示）")
            else:
                hint = "你是一个简洁的助手，请用一句话回答。"
                print("🔀 简单问题模式（注入简洁提示）")

            # 返回 llm_input_messages，这样不会污染 state 中的 messages
            return {
                "llm_input_messages": [{"role": "system", "content": hint}] + messages,
                "messages": messages,  # 保留原 messages，hook 只影响 LLM 输入
            }

    return {"messages": messages}

    return messages


# ============================================================
# 4. post_model_hook：模型调用后执行
# ============================================================

def log_model_call(state: dict) -> dict:
    """模型调用后记录日志"""
    messages = state.get("messages", [])
    if messages:
        last = messages[-1]
        if isinstance(last, AIMessage):
            if last.tool_calls:
                print(f"🔧 AI 决定调工具: {[tc['name'] for tc in last.tool_calls]}")
            else:
                print(f"📊 AI 回复: {str(last.content)[:50]}...")
    return state


# ============================================================
# 5. 创建带 hook 的 Agent
# ============================================================
agent_with_hooks = create_react_agent(
    model=cheap_model,
    tools=[get_weather],
    pre_model_hook=dynamic_prompt_selector,  # 调用前
    post_model_hook=log_model_call,           # 调用后
)


# ============================================================
# 6. 真正的动态模型选择（在外部控制）
# ============================================================

def invoke_with_model_selection(question: str, model=cheap_model) -> dict:
    """
    根据问题选择模型，然后调用 agent。
    这才是真正的"动态模型选择"——在外部切换 model。
    """
    chosen = cheap_model
    if len(question) > 30:
        chosen = model  # 这里可以换成贵模型
        print("🔀 选择模型: 贵的模型")
    else:
        print("🔀 选择模型: 便宜的模型")

    # 每次都新建 agent（绑定不同的 model）
    agent = create_react_agent(model=chosen, tools=[get_weather])
    return agent.invoke({"messages": [HumanMessage(content=question)]})


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
    print("测试 3：真正的外部动态模型选择")
    print("=" * 60)
    result = invoke_with_model_selection("北京天气怎么样？")
    print(f"  回答: {result['messages'][-1].content[:80]}...\n")

    print("=" * 60)
    print("总结")
    print("=" * 60)
    print("  1. pre_model_hook  → 修改 messages（注入提示、上下文）")
    print("  2. post_model_hook → 记录日志、后处理")
    print("  3. 动态换模型 → 在外面控制 model 参数，不要指望 hook 能换模型")

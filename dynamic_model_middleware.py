"""
动态模型选择中间件 — wrap_model_call 使用示例
==============================================

@wrap_model_call 是 langgraph-prebuilt 提供的装饰器，
让你在模型的每次调用前后插入自定义逻辑。

典型场景：
  1. 根据问题类型自动切换模型（简单问题用便宜模型，复杂用贵模型）
  2. 在每次调用前注入上下文或记忆
  3. 记录调用日志或监控 token 消耗
  4. 动态注入 SystemMessage
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# 1. 基础：创建一个 React Agent
# ============================================================
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState


# 一个简单的工具
@tool
def get_weather(city: str) -> str:
    """获取城市天气"""
    return f"{city}：晴天，25°C"


model = init_chat_model(
    model="deepseek-chat",
    model_provider="deepseek",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)

agent = create_react_agent(model, [get_weather])


# ============================================================
# 2. @wrap_model_call 的基本用法
# ============================================================
# wrap_model_call 接收一个函数，这个函数签名是：
#   def wrapper(state: AgentState) -> AgentState
# 你可以在函数里修改 state["messages"]，然后返回修改后的 state
#
# 被 @wrap_model_call 装饰的函数会在每次模型调用前执行
from langgraph.prebuilt import wrap_model_call


@wrap_model_call
def inject_system_prompt(state: AgentState) -> AgentState:
    """
    每次模型调用前，往 messages 最前面插入一条 SystemMessage。

    这个是很常用的模式——比如每次让模型先想一想再回答，
    或者注入当前时间、用户信息等上下文。
    """
    # state["messages"] 是当前对话历史（LangChain Messages 列表）
    # 在最前面插入一条 SystemMessage
    messages = list(state["messages"])
    messages.insert(0, SystemMessage(content="你是知识渊博的助手。"))

    # ⚠️ 一定要返回修改后的 state
    return AgentState(messages=messages, next="agent")


# ============================================================
# 3. 实战：根据问题动态选择模型
# ============================================================

# 定义多个模型
cheap_model = init_chat_model(
    model="deepseek-chat",      # 便宜的模型
    model_provider="deepseek",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)

expensive_model = init_chat_model(
    model="deepseek-chat",      # 这里可以换成更强的模型
    model_provider="deepseek",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)


@wrap_model_call
def dynamic_model_selector(state: AgentState) -> AgentState:
    """
    根据问题复杂度动态选择模型。

    规则：
      - 问题字符 < 20 → 用便宜模型
      - 问题字符 >= 20 → 用贵模型
      - 如果上一条消息是 ToolMessage（工具刚执行完）→ 换回便宜模型
    """
    messages = list(state["messages"])

    # 判断上一条消息是不是工具结果
    last_msg = messages[-1] if messages else None
    if last_msg and last_msg.type == "tool":
        # 工具刚执行完，换回便宜的模型
        model = cheap_model
    else:
        # 看最后一条 HumanMessage 的长度
        for msg in reversed(messages):
            if msg.type == "human" and isinstance(msg.content, str):
                if len(msg.content) > 20:
                    model = expensive_model
                else:
                    model = cheap_model
                break
        else:
            model = cheap_model  # 默认

    # 把选好的模型存到 state 的 metadata 里
    # 这样可以在外部看到选了哪个模型
    print(f"🔀 模型选择: {'💰 贵的' if model == expensive_model else '🪙 便宜的'}")

    # ⚠️ 返回的 state 会被传递给实际的模型调用
    return AgentState(messages=messages, model=model)


# ============================================================
# 4. 把中间件挂载到 agent 上
# ============================================================
# create_react_agent 支持 model_prompt 参数传入 wrap_model_call 装饰的函数
# 注意：这里用 model_prompt 参数来指定中间件

# 方法 A：用 @wrap_model_call 装饰的函数名
agent_with_middleware = create_react_agent(
    model=cheap_model,
    tools=[get_weather],
    # 传入中间件列表，按顺序执行
    # 可以传多个，如 [dynamic_model_selector, inject_system_prompt]
    model_prompt=[dynamic_model_selector],
)

# ============================================================
# 5. 运行测试
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("测试 1：简单问题（应该选便宜模型）")
    print("=" * 60)
    result = agent_with_middleware.invoke({
        "messages": [HumanMessage(content="天气怎么样？")]
    })
    print(f"  回答: {result['messages'][-1].content[:80]}...\n")

    print("=" * 60)
    print("测试 2：复杂问题（应该选贵模型）")
    print("=" * 60)
    result = agent_with_middleware.invoke({
        "messages": [
            HumanMessage(
                content="请详细分析一下北京过去一周的天气变化趋势，"
                        "并给出出行建议，同时对比上海和广州的天气差异"
            )
        ]
    })
    print(f"  回答: {result['messages'][-1].content[:80]}...\n")

    print("=" * 60)
    print("测试 3：调工具（工具结果后换回便宜模型）")
    print("=" * 60)
    result = agent_with_middleware.invoke({
        "messages": [HumanMessage(content="北京天气怎么样？")]
    })
    print(f"  回答: {result['messages'][-1].content[:80]}...")

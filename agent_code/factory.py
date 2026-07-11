"""
Agent 工厂：使用高阶函数和闭包构建可配置的 Agent
"""

from typing import Callable, Optional


def create_agent_factory(
    default_model: str = "gpt-3.5-turbo",
    default_temperature: float = 0.7
) -> Callable:
    """
    创建 Agent 工厂函数

    这是一个高阶函数，返回一个配置好的 Agent 创建器

    Args:
        default_model: 默认模型
        default_temperature: 默认温度

    Returns:
        Agent 创建函数
    """
    def create_agent(
        name: str,
        system_prompt: str,
        tools: Optional[list] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None
    ) -> dict:
        """
        创建 Agent 实例

        Args:
            name: Agent 名称
            system_prompt: 系统提示词
            tools: 可用工具列表
            model: 模型（可选，使用默认值）
            temperature: 温度（可选，使用默认值）

        Returns:
            Agent 配置字典
        """
        return {
            "name": name,
            "model": model or default_model,
            "temperature": temperature or default_temperature,
            "system_prompt": system_prompt,
            "tools": tools or [],
            "created_with_defaults": {
                "model": default_model,
                "temperature": default_temperature
            }
        }

    return create_agent


def create_tool_registry():
    """
    创建工具注册表（使用闭包）

    Returns:
        包含注册和获取函数的字典
    """
    tools: dict[str, Callable] = {}

    def register(name: str, func: Callable) -> None:
        """注册工具"""
        tools[name] = func
        print(f"工具 '{name}' 已注册")

    def get(name: str) -> Optional[Callable]:
        """获取工具"""
        return tools.get(name)

    def list_tools() -> list[str]:
        """列出所有工具"""
        return list(tools.keys())

    return {
        "register": register,
        "get": get,
        "list": list_tools
    }


def compose(*functions: Callable) -> Callable:
    """
    函数组合：将多个函数组合成一个

    Args:
        *functions: 要组合的函数列表

    Returns:
        组合后的函数

    Example:
        >>> def add_one(x): return x + 1
        >>> def double(x): return x * 2
        >>> f = compose(add_one, double)
        >>> f(3)  # double(add_one(3)) = double(4) = 8
        8
    """
    from functools import reduce

    def compose_two(f: Callable, g: Callable) -> Callable:
        return lambda x: g(f(x))

# reduce的作用：
    return reduce(compose_two, functions, lambda x: x)


# 演示使用
def main() -> None:
    """主函数：演示高阶函数和闭包"""

    # 1. Agent 工厂
    print("=== Agent 工厂 ===")
    factory = create_agent_factory(
        default_model="gpt-4",
        default_temperature=0.5
    )

    agent1 = factory(
        name="ResearchBot",
        system_prompt="You are a research assistant"
    )

    agent2 = factory(
        name="ChatBot",
        system_prompt="You are a friendly chatbot",
        model="gpt-3.5-turbo",
        temperature=0.9
    )

    print(f"Agent1: {agent1['name']}, 模型: {agent1['model']}")
    print(f"Agent2: {agent2['name']}, 模型: {agent2['model']}")

    # 2. 工具注册表
    print("\n=== 工具注册表 ===")
    registry = create_tool_registry()

    def weather_tool(location: str) -> str:
        return f"Weather in {location}: Sunny"

    def calculator_tool(expression: str) -> str:
        return f"Calculating: {expression}"

    registry["register"]("weather", weather_tool)
    registry["register"]("calculator", calculator_tool)

    print(f"已注册工具: {registry['list']()}")

    tool = registry["get"]("weather")
    if tool:
        print(tool("Beijing"))

    # 3. 函数组合
    print("\n=== 函数组合 ===")

    def clean_text(text: str) -> str:
        return text.strip().lower()

    def remove_punctuation(text: str) -> str:
        import string
        return text.translate(str.maketrans("", "", string.punctuation))

    def tokenize(text: str) -> list[str]:
        return text.split()

    # 组合处理流程
    process_text = compose(clean_text, remove_punctuation)

    input_text = "  Hello, World!  "
    processed = process_text(input_text)
    print(f"原始: '{input_text}'")
    print(f"处理后: '{processed}'")


if __name__ == "__main__":
    main()
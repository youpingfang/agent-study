#!/usr/bin/env -S uv run
"""
MCP 客户端 — 调用天气服务
======================

这个脚本是 MCP 的"客户端"（Client）。
它启动本地 MCP 服务端进程，通过 stdio 与服务端通信，
调用服务端注册的工具（Tool），获取结果。

数据流：
  Client (这个文件)
    → 1. 启动 Server 子进程 (subprocess)
    → 2. 通过 stdio 建立 MCP 连接
    → 3. 发送"查询工具列表"请求
    → 4. 发送"调用工具"请求（带上参数）
    → 5. 接收工具执行结果
    → 6. 关闭连接

类比理解：
  这就像你之前写的 agent.py 中 LLM 调 Function Calling 的流程，
  但 MCP 把工具调用标准化了，让任何 LLM 客户端都能发现和调用远程工具。

关键词：Client、stdio transport、Session、Tool call
"""

# ============================================================
# 第一部分：导入依赖
# ============================================================
import asyncio
import sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# 关键类说明：
#   ClientSession   — MCP 客户端会话，封装了与服务端的通信
#   StdioServerParameters — 描述如何启动服务端子进程的配置
#   stdio_client    — 创建一个通过 stdio 连接的服务端上下文管理器


# ============================================================
# 第二部分：配置 MCP 服务端路径
# ============================================================
# 告诉客户端："服务端程序在哪里？怎么启动它？"
# command：启动命令（这里是 uv run python 文件）
# args：传给命令的参数列表
SERVER_PATH = "weather_server.py"  # 服务端脚本路径（与当前文件同目录）
server_params = StdioServerParameters(
    command="uv",           # 使用 uv 来运行（自动激活虚拟环境）
    args=["run", SERVER_PATH],  # uv run weather_server.py
    # env 可以不传，默认继承当前环境变量
)


# ============================================================
# 第三部分：核心逻辑 — 与 MCP 服务端交互
# ============================================================
async def call_weather_tool(city: str) -> None:
    """
    连接 MCP 服务端并调用天气查询工具

    整个流程分为 4 步：
      1. 建立连接（启动子进程 + 建立 Session）
      2. 列出可用工具（list_tools）
      3. 调用具体工具（call_tool）
      4. 关闭连接
    """

    # ------------------------------------------------
    # 关键概念：上下文管理器（async with）
    # stdio_client 自动处理：
    #   1. 用 subprocess 启动 weather_server.py 进程
    #   2. 通过管道（stdin/stdout）连接子进程
    #   3. 退出时自动关闭子进程
    # ------------------------------------------------
    async with stdio_client(server_params) as (read, write):
        # read  — 从服务端读取数据（stdout）
        # write — 向服务端写入数据（stdin）

        # ------------------------------------------------
        # 建立 MCP 会话
        # ClientSession 封装了 JSON-RPC 协议通信
        # 它会自动处理：握手、请求/响应匹配、错误处理
        # ------------------------------------------------
        async with ClientSession(read, write) as session:
            # ---------- 第 1 步：初始化会话 ----------
            # initialize() 发送初始化请求，服务端返回支持的协议版本和能力
            await session.initialize()
            print(f"✅ 已连接到 MCP 天气服务端\n")

            # ---------- 第 2 步：列出所有可用工具 ----------
            # list_tools() 返回服务端注册的所有 Tool 信息
            # 对应 weather_server.py 中用 @mcp.tool() 装饰的函数
            tools = await session.list_tools()
            print("📦 服务端注册的工具列表：")
            for tool in tools.tools:
                print(f"   🔧 工具名: {tool.name}")
                print(f"   📝 描述:   {tool.description}")
                print(f"   📊 参数:   {tool.inputSchema}")
                print()
            print("-" * 50)

            # ---------- 第 3 步：调用具体工具 ----------
            # call_tool() 向服务端发送"调用工具"请求
            # 参数说明：
            #   name — 工具名（必须与服务端注册的一致）
            #   arguments — 传给工具的参数（字典形式）
            print(f"🌤️  正在查询「{city}」的天气...")
            result = await session.call_tool(
                name="get_weather",     # 对应 server 的 @mcp.tool() 函数名
                arguments={"city": city},
            )

            # ---------- 第 4 步：输出结果 ----------
            # call_tool 返回的结果中包含 content 列表
            # 每个 content 可以是 TextContent 或 ImageContent 等类型
            for content in result.content:
                if content.type == "text":
                    print(f"\n📋 查询结果：")
                    print(content.text)

            print(f"\n✅ 查询完成！")

            # with 块结束时自动关闭会话和子进程


# ============================================================
# 第四部分：入口
# ============================================================
async def main():
    """主函数：解析命令行参数并调用天气工具"""

    # 从命令行参数获取城市名，没传则默认为北京
    city = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "北京"

    print(f"\n{'='*50}")
    print(f"  MCP 天气客户端 v1.0")
    print(f"{'='*50}\n")

    try:
        await call_weather_tool(city)
    except Exception as e:
        print(f"❌ 发生错误: {e}")
        print("  请确认 weather_server.py 在同一目录下")
        print("  并且已执行 uv add mcp 安装依赖")
        sys.exit(1)


# ============================================================
# 第五部分：运行
# ============================================================
# asyncio.run() 是 Python 3.7+ 的标准异步入口
# MCP 客户端使用 async/await 异步模型，所以需要用这层包装
if __name__ == "__main__":
    asyncio.run(main())

from langchain_core.tools import tool
from pydantic import BaseModel, Field
import time,json
from typing import Optional


class WebSearchInput(BaseModel):
    """网络搜索输入参数"""
    query: str = Field(description="搜索查询")
    max_results: int = Field(default=5, description="最大结果数")


@tool(args_schema=WebSearchInput)
def search_web(query: str, max_results: int = 5) :
    """
    搜索网络并返回结果
    
    Args:
        query: 搜索查询
        max_results: 返回的最大结果数
    
    Returns:
        搜索结果列表（JSON 格式）
    """
    # 参数验证
    if not query or len(query.strip()) == 0:
        return json.dumps({"error": "搜索查询不能为空"})
    
    if max_results < 1 or max_results > 10:
        return json.dumps({"error": "max_results 必须在 1-10 之间"})
    
    # 重试逻辑
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # 模拟 API 调用
            results = [
                {
                    "title": f"搜索结果 {i+1}: {query}",
                    "url": f"https://example.com/result{i+1}",
                    "snippet": f"关于 {query} 的相关信息..."
                }
                for i in range(max_results)
            ]
            
            return json.dumps({
                "query": query,
                "results": results,
                "total": len(results)
            }, ensure_ascii=False, indent=2)
            
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            return json.dumps({
                "error": f"搜索失败: {str(e)}"
            })

res=search_web.invoke({"query":"今天南京的天气"})
print(res)
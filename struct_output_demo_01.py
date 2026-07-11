"""
结构化输出示例
"""
from init_LLM import deepseek_chat_model
from pydantic import BaseModel, Field
class Movie(BaseModel):
    title: str = Field(..., description="电影名称")
    director: str = Field(..., description="导演")
    year: int = Field(..., description="上映年份") 
    rating: float = Field(..., description="评分") 

res = deepseek_chat_model.with_structured_output(Movie).invoke("请告诉我电影《肖申克的救赎》的基础信息。")
print(res)


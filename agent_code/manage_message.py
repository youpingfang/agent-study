"""
Agent 消息历史管理器
演示列表在 AI Agent 中的实际应用
"""

from typing import List, Dict, Optional, Literal
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Message:
    """消息数据类"""
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: datetime
    metadata: Optional[Dict] = None


class MessageHistory:
    """消息历史管理器"""
    
    def __init__(self, max_messages: int = 100):
        """
        初始化消息历史
        
        Args:
            max_messages: 最大保存消息数
        """
        self.messages: List[Message] = []
        self.max_messages = max_messages
    
    def add_message(
        self,
        role: Literal["user", "assistant", "system"],
        content: str,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        添加消息
        
        Args:
            role: 消息角色
            content: 消息内容
            metadata: 元数据
        """
        message = Message(
            role=role,
            content=content,
            timestamp=datetime.now(),
            metadata=metadata
        )
        
        self.messages.append(message)
        
        # 保持在最大限制内
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
    
    def get_recent(self, n: int = 10) -> List[Message]:
        """
        获取最近的 n 条消息
        
        Args:
            n: 消息数量
        
        Returns:
            消息列表
        """
        return self.messages[-n:]
    
    def get_by_role(self, role: str) -> List[Message]:
        """
        按角色筛选消息
        
        Args:
            role: 消息角色
        
        Returns:
            筛选后的消息列表
        """
        return [msg for msg in self.messages if msg.role == role]
    
    def search(self, keyword: str) -> List[Message]:
        """
        搜索包含关键词的消息
        
        Args:
            keyword: 搜索关键词
        
        Returns:
            包含关键词的消息列表
        """
        return [
            msg for msg in self.messages
            if keyword.lower() in msg.content.lower()
        ]
    
    def get_context_window(
        self,
        max_tokens: int = 2000,
        tokens_per_message: int = 50
    ) -> List[Message]:
        """
        获取符合 token 限制的上下文窗口
        
        Args:
            max_tokens: 最大 token 数
            tokens_per_message: 每条消息平均 token 数
        
        Returns:
            上下文消息列表
        """
        max_messages = max_tokens // tokens_per_message
        return self.get_recent(max_messages)
    
    def to_langchain_format(self) -> List[Dict[str, str]]:
        """
        转换为 LangChain 消息格式
        
        Returns:
            LangChain 格式的消息列表
        """
        return [
            {
                "role": msg.role,
                "content": msg.content
            }
            for msg in self.messages
        ]
    
    def clear(self) -> None:
        """清空消息历史"""
        self.messages = []
    
    def get_summary(self) -> Dict:
        """
        获取历史摘要
        
        Returns:
            摘要信息
        """
        total = len(self.messages)
        by_role = {}
        
        for msg in self.messages:
            by_role[msg.role] = by_role.get(msg.role, 0) + 1
        
        return {
            "total_messages": total,
            "by_role": by_role,
            "first_message": self.messages[0].timestamp if self.messages else None,
            "last_message": self.messages[-1].timestamp if self.messages else None,
        }


# 使用示例
def main():
    """演示消息历史管理器"""
    history = MessageHistory(max_messages=50)
    
    # 添加对话
    history.add_message("user", "你好")
    history.add_message("assistant", "您好！有什么可以帮助您？")
    history.add_message("user", "天气怎么样？")
    history.add_message("assistant", "让我为您查询天气信息...")
    history.add_message("user", "谢谢")
    
    # 获取最近消息
    recent = history.get_recent(3)
    print(f"最近 3 条消息: {len(recent)}")
    
    # 按角色筛选
    user_messages = history.get_by_role("user")
    print(f"用户消息数: {len(user_messages)}")
    
    # 搜索
    results = history.search("天气")
    print(f"包含'天气'的消息: {len(results)}")
    
    # 获取摘要
    summary = history.get_summary()
    print(f"\n摘要: {summary}")
    
    # 转换为 LangChain 格式
    langchain_messages = history.to_langchain_format()
    print(f"\nLangChain 格式: {langchain_messages[:2]}")


if __name__ == "__main__":
    main()
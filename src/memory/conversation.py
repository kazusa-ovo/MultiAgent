from __future__ import annotations

from src.core.types import Message

class ConversationBuffer:
    def __init__(self,max_size: int =20):
        self.max_size = max_size
        self._messages: list[Message] = []

    def add(self,role: str, content: str) -> None:
        self._messages.append(Message(role=role,content=content))
        if len(self._messages) >= self.max_size:
            self._messages = self._messages[-self.max_size:]

    def get_all(self) -> list[Message]:
        return list(self._messages)

    def get_recent(self, n: int = 10) -> list[Message]:
        return list(self._messages[-n:])

    def clear(self) -> None:
        self._messages.clear()

    def to_dict_list(self) -> list[dict[str,str]]:
        return [
            {
            "role":m.role,
            "content":m.content
            }
            for m in self._messages
        ]

    def summarize_for_context(self,max_char: int = 2000) -> str:
        lines: list[str] = []
        total: int = 0
        for m in reversed(self._messages):
            line = f"[{m.role}]: {m.content}"
            total += len(line)
            if total > max_char:
                break
            lines.append(line)
        return "\n".join(reversed(lines))

    def __len__(self) -> int:
        return len(self._messages)




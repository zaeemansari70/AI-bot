# chat_memory.py
from dataclasses import dataclass, field
from typing import List
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

@dataclass
class ChatState:
    system_base: str
    messages: List = field(default_factory=list)
    turn_count: int = 0

    def init(self):
        self.messages = [SystemMessage(content=self.system_prompt())]

    def system_prompt(self) -> str:
        return self.system_base

    def add_user(self, text: str):
        self.messages.append(HumanMessage(content=text))
        self.turn_count += 1

    def add_ai(self, text: str):
        self.messages.append(AIMessage(content=text))

    def rebuild_system_message(self):
        # Replace the first system message with updated prompt that includes summary
        self.messages[0] = SystemMessage(content=self.system_prompt())

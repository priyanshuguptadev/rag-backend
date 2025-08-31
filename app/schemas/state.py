from pydantic import BaseModel
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from typing import List


class State(BaseModel):
    question: str
    context: List[Document]
    answer: str
    chat_history: List[BaseMessage]

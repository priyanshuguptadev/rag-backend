from pydantic import BaseModel
from typing import List, Dict, Any


class UserQuery(BaseModel):
    question: str
    history: List[Dict[str, Any]]

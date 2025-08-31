from fastapi import APIRouter
from app.services.rag_service import answer_question
from app.schemas.response import UserResponse
from app.schemas.query import UserQuery

router = APIRouter()


@router.post("/answer", response_model=UserResponse)
async def get_answer(query: UserQuery):
    return await answer_question(query.question, query.history)

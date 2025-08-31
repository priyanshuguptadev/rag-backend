from fastapi import FastAPI
from app.routes import answer

app = FastAPI()

app.include_router(answer.router)


@app.get("/")
async def read_root():
    return {"msg": "welcome to RAG."}

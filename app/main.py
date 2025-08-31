from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import answer
from dotenv import load_dotenv
import os

load_dotenv()

FRONTEND_URL = os.getenv("FRONTEND_URL")

app = FastAPI()

origins = [FRONTEND_URL]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

app.include_router(answer.router)


@app.get("/")
async def read_root():
    return {"msg": "welcome to RAG."}

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from app.schemas.state import State
from langgraph.graph import StateGraph, START
from typing import List
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain import hub
from dotenv import load_dotenv
from pathlib import Path
import os

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

BASE_DIR = Path(__file__).resolve().parent.parent
RESUME_PATH = BASE_DIR / "data" / "resume.pdf"

llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

loader = PyPDFLoader(str(RESUME_PATH))
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    add_start_index=True,
)
splits = splitter.split_documents(docs)


try:
    vector_store = FAISS.load_local(
        "faiss_index", embeddings, allow_dangerous_deserialization=True
    )
except:
    vector_store = FAISS.from_documents(splits, embeddings)
    vector_store.save_local("faiss_index")

prompt = hub.pull("daekeun-ml/rag-baseline")


def retrieve(state: State):
    query = state.question
    for message in state.chat_history:
        query = f"{message.content}\n{query}"

    retrieve_docs = vector_store.similarity_search(query, k=4)
    return {"context": retrieve_docs}


def generate(state: State):
    if not state.context:
        return {"answer": "I couldn't find the info in my resume."}
    docs_content = "/n/n".join(doc.page_content for doc in state.context)
    messages = (
        state.chat_history
        + prompt.invoke(
            {"context": docs_content, "question": state.question}
        ).to_messages()
    )
    answer = llm.invoke(messages).content
    return {"answer": answer}


graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()


async def answer_question(question: str, history: List[BaseMessage] = []):
    converted_history = []
    for msg in history:
        if msg["role"] == "user":
            converted_history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            converted_history.append(AIMessage(content=msg["content"]))
    result = graph.invoke(
        {
            "question": question,
            "context": [],
            "answer": "",
            "chat_history": converted_history,
        }
    )
    return {"answer": result["answer"]}

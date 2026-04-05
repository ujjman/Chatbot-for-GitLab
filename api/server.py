from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from chatbot.rag_service import RagService
from chatbot.settings import settings


app = FastAPI(title="GitLab Handbook MCP Chatbot API (Groq)", version="2.1.0")
rag_service: RagService | None = None


class ChatTurn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(
        default=settings.firecrawl_max_pages,
        ge=1,
        le=100,
        description="Max pages Gemini should inspect via Firecrawl MCP",
    )
    site_filter: str = Field(default="all")
    chat_history: list[ChatTurn] = Field(default_factory=list)


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    retrieved_count: int


@app.on_event("startup")
def startup_event() -> None:
    global rag_service
    rag_service = RagService()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> dict[str, Any]:
    if rag_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        result = rag_service.answer(
            question=payload.question,
            top_k=payload.top_k,
            site_filter=payload.site_filter,
            chat_history=[{"role": t.role, "content": t.content} for t in payload.chat_history],
        )
        return result
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc

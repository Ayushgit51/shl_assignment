"""
main.py
--------
FastAPI application with two endpoints:
  GET  /health  -> {"status": "ok"}
  POST /chat    -> conversation + recommendations

The API is STATELESS — every /chat call carries the full conversation history.
No session state is stored on the server.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import uvicorn
import os

from agent.core import run_agent
from retrieval.engine import get_engine  # pre-load at startup


from dotenv import load_dotenv
load_dotenv()


# ─────────────────────────────────────────────
# App Setup
# ─────────────────────────────────────────────

app = FastAPI(
    title="SHL Assessment Recommender",
    description="Conversational agent for SHL assessment recommendations",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Pydantic Schemas (non-negotiable per assignment)
# ─────────────────────────────────────────────

class Message(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    messages: List[Message] = Field(..., description="Full conversation history")


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: List[Recommendation]
    end_of_conversation: bool


# ─────────────────────────────────────────────
# Startup: Pre-load retrieval engine
# ─────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """Pre-load the FAISS index and sentence transformer at startup."""
    print("[Startup] Pre-loading retrieval engine...")
    get_engine()
    print("[Startup] Ready.")


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@app.get("/health")
async def health():
    """Health check endpoint. Returns 200 OK when service is ready."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint.

    Takes full stateless conversation history.
    Returns next agent reply + optional shortlist of recommendations.
    """
    if not request.messages:
        raise HTTPException(status_code=400, detail="messages cannot be empty")

    # Validate roles
    for msg in request.messages:
        if msg.role not in ("user", "assistant"):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid role '{msg.role}'. Must be 'user' or 'assistant'."
            )

    # Ensure last message is from user
    if request.messages[-1].role != "user":
        raise HTTPException(
            status_code=400,
            detail="Last message must be from 'user'."
        )

    # Convert to plain dicts for agent
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    # Run agent
    result = run_agent(messages)

    return ChatResponse(
        reply=result["reply"],
        recommendations=[
            Recommendation(
                name=r["name"],
                url=r["url"],
                test_type=r["test_type"],
            )
            for r in result.get("recommendations", [])
        ],
        end_of_conversation=result.get("end_of_conversation", False),
    )


# ─────────────────────────────────────────────
# Run locally
# ─────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)

import os
import json

from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from retriever import retrieve_assessments
from prompts import SYSTEM_PROMPT

# -----------------------------
# LOAD ENV
# -----------------------------
load_dotenv()

app = FastAPI()

# -----------------------------
# GLOBAL VARIABLES (IMPORTANT FIX)
# -----------------------------
llm = None

# -----------------------------
# INITIALIZE MODEL SAFELY
# -----------------------------
@app.on_event("startup")
def startup_event():
    global llm

    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        print("❌ GOOGLE_API_KEY missing")
        return

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        google_api_key=api_key,
        temperature=0.3
    )

    print("✅ LLM Loaded Successfully")


# -----------------------------
# REQUEST SCHEMAS
# -----------------------------
class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]


# -----------------------------
# HEALTH CHECK
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


# -----------------------------
# HELPER FUNCTION
# -----------------------------
def is_vague_query(text):

    vague_words = [
        "assessment",
        "test",
        "hiring",
        "need assessment"
    ]

    text = text.lower()

    if len(text.split()) < 4:
        return True

    for word in vague_words:
        if word == text.strip():
            return True

    return False


# -----------------------------
# CHAT ENDPOINT
# -----------------------------
@app.post("/chat")
def chat(request: ChatRequest):

    global llm

    if llm is None:
        return {
            "reply": "Model not initialized. Check server logs.",
            "recommendations": [],
            "end_of_conversation": False
        }

    messages = request.messages

    latest_user_message = ""

    for msg in reversed(messages):
        if msg.role == "user":
            latest_user_message = msg.content
            break

    # -------------------------
    # CLARIFICATION LOGIC
    # -------------------------
    if is_vague_query(latest_user_message):
        return {
            "reply": "Can you specify the role, experience level, and whether you need technical, cognitive, or personality assessments?",
            "recommendations": [],
            "end_of_conversation": False
        }

    # -------------------------
    # CONTEXT BUILDING
    # -------------------------
    conversation_text = "\n".join([
        f"{m.role}: {m.content}"
        for m in messages
    ])

    # -------------------------
    # RETRIEVAL
    # -------------------------
    docs = retrieve_assessments(conversation_text)

    retrieved_context = "\n\n".join([
        d.page_content for d in docs
    ])

    # -------------------------
    # FINAL PROMPT
    # -------------------------
    final_prompt = f"""
{SYSTEM_PROMPT}

Conversation:
{conversation_text}

Retrieved SHL Assessments:
{retrieved_context}

Generate:
1. Conversational reply
2. Recommended assessments
"""

    response = llm.invoke(final_prompt)

    # -------------------------
    # BUILD RESPONSE
    # -------------------------
    recommendations = []

    for d in docs[:5]:
        recommendations.append({
            "name": d.metadata.get("name"),
            "url": d.metadata.get("url"),
            "test_type": d.metadata.get("test_type")
        })

    return {
        "reply": response.content,
        "recommendations": recommendations,
        "end_of_conversation": False
    }

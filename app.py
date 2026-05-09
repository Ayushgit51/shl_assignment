import os
import json

from fastapi import FastAPI
from pydantic import BaseModel

from langchain_google_genai import ChatGoogleGenerativeAI
from retriever import retrieve_assessments
from prompts import SYSTEM_PROMPT
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")


# Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    google_api_key=api_key,
    temperature=0.3
)

app = FastAPI()

# -----------------------------
# Request Schema
# -----------------------------

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[Message]

# -----------------------------
# Health Endpoint
# -----------------------------

@app.get("/health")
def health():
    return {"status": "ok"}

# -----------------------------
# Helper Function
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
# Chat Endpoint
# -----------------------------

@app.post("/chat")
def chat(request: ChatRequest):

    messages = request.messages

    latest_user_message = ""

    for msg in reversed(messages):
        if msg.role == "user":
            latest_user_message = msg.content
            break

    # Clarification handling
    if is_vague_query(latest_user_message):

        return {
            "reply": "Can you specify the role, experience level, and whether you need technical, cognitive, or personality assessments?",
            "recommendations": [],
            "end_of_conversation": False
        }

    # Build conversation context
    conversation_text = "\n".join([
        f"{m.role}: {m.content}"
        for m in messages
    ])

    # Retrieve documents
    docs = retrieve_assessments(conversation_text)

    retrieved_context = "\n\n".join([
        d.page_content
        for d in docs
    ])

    # Final prompt
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

    # Build recommendations
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

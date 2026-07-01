"""
agent/core.py
--------------
Main agent logic. Handles the full conversation flow:
  1. Classify intent from conversation history
  2. Retrieve relevant catalog items (if needed)
  3. Build prompt with catalog context
  4. Call Groq LLM
  5. Parse and validate response
  6. Return structured output

Design decisions:
- Intent classification is a separate fast LLM call (low token, quick)
- Catalog context is injected into the prompt (not hallucinated by LLM)
- All URLs in response are validated against catalog before returning
- Timeouts handled by keeping prompts lean (<30s total budget)
"""

import json
import os
import re
from typing import List, Dict, Optional, Tuple
from groq import Groq

from catalog.loader import CATALOG, get_by_name
from retrieval.engine import get_engine
from agent.prompts import SYSTEM_PROMPT, INTENT_CLASSIFIER_PROMPT

# ─────────────────────────────────────────────
# Groq Client Setup
# ─────────────────────────────────────────────

def get_groq_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set in environment")
    return Groq(api_key=api_key)


# ─────────────────────────────────────────────
# Intent Classification
# ─────────────────────────────────────────────

def classify_intent(messages: List[Dict], client: Groq) -> Dict:
    """Fast LLM call to understand what the user wants."""
    conv_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in messages
    )
    prompt = INTENT_CLASSIFIER_PROMPT.format(conversation=conv_text)

    try:
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",  # fast small model for classification
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=300,
        )
        raw = resp.choices[0].message.content.strip()
        # Strip markdown code fences if present
        raw = re.sub(r"```json|```", "", raw).strip()
        return json.loads(raw)
    except Exception as e:
        # Fallback: assume recommend intent
        print(f"[Intent] Classification failed: {e}, using fallback")
        return {
            "intent": "recommend",
            "has_enough_context": True,
            "role_or_skill": None,
            "job_level": None,
            "test_types_requested": [],
            "compare_items": [],
            "is_off_topic": False,
            "user_confirmed": False,
        }


# ─────────────────────────────────────────────
# Prompt Injection Detection
# ─────────────────────────────────────────────

INJECTION_PATTERNS = [
    r"ignore (previous|all|your) instructions",
    r"you are now",
    r"forget everything",
    r"system prompt",
    r"jailbreak",
    r"act as",
    r"pretend (you are|to be)",
    r"disregard",
]

def is_prompt_injection(text: str) -> bool:
    text_lower = text.lower()
    return any(re.search(p, text_lower) for p in INJECTION_PATTERNS)


# ─────────────────────────────────────────────
# Catalog Context Builder
# ─────────────────────────────────────────────

def build_catalog_context(items: List[Dict], max_items: int = 20) -> str:
    """Convert retrieved catalog items to a compact text block for the prompt."""
    lines = []
    for item in items[:max_items]:
        lang_str = ", ".join(item["languages"][:3]) if item["languages"] else "—"
        if len(item["languages"]) > 3:
            lang_str += f" (+{len(item['languages'])-3} more)"
        lines.append(
            f"- Name: {item['name']}\n"
            f"  URL: {item['link']}\n"
            f"  test_type: {item['test_type']}\n"
            f"  Keys: {', '.join(item['keys'])}\n"
            f"  Duration: {item['duration'] or '—'}\n"
            f"  Job Levels: {', '.join(item['job_levels']) or 'All levels'}\n"
            f"  Languages: {lang_str}\n"
            f"  Description: {item['description'][:200]}...\n"
        )
    return "\n".join(lines)


# ─────────────────────────────────────────────
# URL Validator
# ─────────────────────────────────────────────

VALID_URLS = {item["link"] for item in CATALOG}
VALID_NAMES = {item["name"]: item for item in CATALOG}

def validate_recommendations(recs: List[Dict]) -> List[Dict]:
    """
    Remove any recommendation whose URL is not in the real catalog.
    This prevents hallucinated URLs from reaching the evaluator.
    """
    validated = []
    for rec in recs:
        url = rec.get("url", "")
        name = rec.get("name", "")

        # Check URL directly
        if url in VALID_URLS:
            validated.append(rec)
            continue

        # Try to find by name match and fix URL
        catalog_item = get_by_name(name)
        if catalog_item:
            rec["url"] = catalog_item["link"]
            rec["test_type"] = catalog_item["test_type"]
            validated.append(rec)
            continue

        # Could not validate — skip
        print(f"[Validate] Dropping hallucinated item: {name} / {url}")

    return validated


# ─────────────────────────────────────────────
# Response Parser
# ─────────────────────────────────────────────

def parse_agent_response(raw: str) -> Dict:
    """Parse LLM JSON output, with fallback for malformed responses."""
    try:
        # Strip markdown code fences
        cleaned = re.sub(r"```json|```", "", raw).strip()
        data = json.loads(cleaned)

        reply = data.get("reply", "").strip()
        recs = data.get("recommendations", []) or []
        eoc = bool(data.get("end_of_conversation", False))

        # Validate all recs against real catalog
        recs = validate_recommendations(recs)

        return {
            "reply": reply,
            "recommendations": recs,
            "end_of_conversation": eoc,
        }
    except Exception as e:
        print(f"[Parse] Failed to parse LLM response: {e}\nRaw: {raw[:300]}")
        return {
            "reply": "I'm sorry, I encountered an issue. Could you rephrase your request?",
            "recommendations": [],
            "end_of_conversation": False,
        }


# ─────────────────────────────────────────────
# Main Agent Function
# ─────────────────────────────────────────────

def run_agent(messages: List[Dict]) -> Dict:
    """
    Main entry point. Takes full conversation history, returns structured response.

    Args:
        messages: list of {"role": "user"/"assistant", "content": "..."}

    Returns:
        {"reply": str, "recommendations": list, "end_of_conversation": bool}
    """
    client = get_groq_client()
    engine = get_engine()

    # --- Check turn count (max 8 turns enforced by evaluator) ---
    if len(messages) > 8:
        return {
            "reply": "We've reached the maximum conversation length. Here is my final recommendation based on our discussion.",
            "recommendations": [],
            "end_of_conversation": True,
        }

    # --- Prompt injection check on latest user message ---
    last_user_msg = ""
    for m in reversed(messages):
        if m["role"] == "user":
            last_user_msg = m["content"]
            break

    if is_prompt_injection(last_user_msg):
        return {
            "reply": "I can only help with SHL assessment recommendations. Please ask me about assessments for a specific role or skill.",
            "recommendations": [],
            "end_of_conversation": False,
        }

    # --- Classify intent ---
    intent_data = classify_intent(messages, client)
    intent = intent_data.get("intent", "recommend")
    is_off_topic = intent_data.get("is_off_topic", False)
    has_enough_context = intent_data.get("has_enough_context", False)
    job_level = intent_data.get("job_level")
    test_types = intent_data.get("test_types_requested", [])
    compare_items = intent_data.get("compare_items", [])
    role_or_skill = intent_data.get("role_or_skill")

    # --- Off-topic refusal ---
    if is_off_topic or intent == "refuse":
        return {
            "reply": "I'm here specifically to help with SHL assessment selection. I can't help with general hiring advice, legal questions, or other topics. Please describe the role you're hiring for and I'll recommend the right assessments.",
            "recommendations": [],
            "end_of_conversation": False,
        }

    # --- Build search query from conversation context ---
    # Use role + job level + last message as search query
    search_parts = []
    if role_or_skill:
        search_parts.append(role_or_skill)
    if job_level:
        search_parts.append(job_level)
    search_parts.append(last_user_msg)
    # Also include previous user messages for context
    for m in messages[:-1]:
        if m["role"] == "user":
            search_parts.append(m["content"])
    search_query = " ".join(search_parts)

    # --- Retrieve catalog items ---
    retrieved_items = engine.search(
        query=search_query,
        top_k=20,  # get more than needed, LLM will select best
        job_level_filter=job_level,
        test_type_filter=test_types if test_types else None,
    )

    # For compare intent: also fetch the specific compared items
    if compare_items:
        for item_name in compare_items:
            item = get_by_name(item_name)
            if item and item not in retrieved_items:
                retrieved_items.insert(0, item)

    # --- Build catalog context ---
    catalog_ctx = build_catalog_context(retrieved_items, max_items=20)

    # --- Build system prompt with catalog ---
    system = SYSTEM_PROMPT.format(catalog_context=catalog_ctx)

    # --- Build messages for LLM (full conversation history) ---
    llm_messages = [{"role": "system", "content": system}] + messages

    # --- LLM call (main response) ---
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # strong model for actual response
            messages=llm_messages,
            temperature=0.2,
            max_tokens=1000,
            response_format={"type": "json_object"},
        )
        raw_output = response.choices[0].message.content
    except Exception as e:
        print(f"[Agent] LLM call failed: {e}")
        return {
            "reply": "I'm experiencing a technical issue. Please try again.",
            "recommendations": [],
            "end_of_conversation": False,
        }

    # --- Parse and validate response ---
    result = parse_agent_response(raw_output)
    return result

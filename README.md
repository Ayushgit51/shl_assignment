# SHL Assessment Recommender

Conversational AI agent that recommends SHL assessments from the official product catalog.

## Folder Structure

```
shl_recommender/
├── catalog/
│   ├── shl_catalog.json     ← Full SHL product catalog (377 items)
│   ├── loader.py            ← Loads + cleans catalog, test_type mapping
│   └── __init__.py
├── retrieval/
│   ├── engine.py            ← BM25 + FAISS hybrid search engine
│   └── __init__.py
├── agent/
│   ├── core.py              ← Main agent logic (intent → retrieve → LLM → validate)
│   ├── prompts.py           ← All LLM prompts
│   └── __init__.py
├── tests/
│   └── test_basic.py        ← Basic tests
├── main.py                  ← FastAPI app (GET /health, POST /chat)
├── requirements.txt
├── .env.example
├── render.yaml              ← Render.com deploy config
└── Procfile                 ← Render/Railway process file
```

## Setup Locally

```bash
# 1. Clone / copy this folder
cd shl_recommender

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set environment variable
cp .env.example .env
# Edit .env and add your GROQ_API_KEY (free at console.groq.com)

# 4. Run tests
python tests/test_basic.py

# 5. Start server
python main.py
# Server runs at http://localhost:8000
```

## Test the API

```bash
# Health check
curl http://localhost:8000/health

# Chat
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "I am hiring a Java developer"}
    ]
  }'
```

## Deploy on Render (Free)

1. Push this folder to a GitHub repo
2. Go to render.com → New Web Service → Connect repo
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Add env variable: `GROQ_API_KEY` = your key
6. Deploy → get your public URL

## API Schema

### POST /chat

**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "I need a Java developer assessment"},
    {"role": "assistant", "content": "What seniority level?"},
    {"role": "user", "content": "Mid-level, 4 years experience"}
  ]
}
```

**Response:**
```json
{
  "reply": "Here are assessments for a mid-level Java developer...",
  "recommendations": [
    {"name": "Java 8 (New)", "url": "https://www.shl.com/...", "test_type": "K"},
    {"name": "Automata - Fix (New)", "url": "https://www.shl.com/...", "test_type": "S"}
  ],
  "end_of_conversation": false
}
```

- `recommendations` = `[]` when agent is clarifying or refusing
- `end_of_conversation` = `true` only when user confirms satisfaction

## Get Free Groq API Key

1. Go to https://console.groq.com
2. Sign up (free)
3. Create API Key
4. Paste in `.env` file

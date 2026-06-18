import os
import json
import re
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from groq import Groq
from rank_bm25 import BM25Okapi


app = FastAPI(title="RCDoK Chatbot API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


# ── Build BM25 index at startup ──
KB_DIR = Path("./knowledge_base")

_docs: list[dict] = []
_bm25: BM25Okapi | None = None


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def build_index():
    global _bm25, _docs
    _docs = []
    for path in sorted(KB_DIR.glob("**/*.txt")):
        try:
            content = path.read_text(encoding="utf-8")
            _docs.append({
                "filename": path.name,
                "content": content,
                "tokens": _tokenize(content),
            })
        except Exception as e:
            print(f"  Warning: could not read {path.name}: {e}")

    if not _docs:
        print("WARNING: No documents found in knowledge_base/")
        return

    corpus_tokens = [d["tokens"] for d in _docs]
    _bm25 = BM25Okapi(corpus_tokens)
    print(f"BM25 index ready — {len(_docs)} documents loaded.")


build_index()


def retrieve(query: str, top_k: int = 5) -> str:
    if _bm25 is None or not _docs:
        return ""

    tokens = _tokenize(query)
    if not tokens:
        return ""

    scores = _bm25.get_scores(tokens)

    ranked = sorted(
        enumerate(scores), key=lambda x: x[1], reverse=True
    )[:top_k]

    relevant = [
        _docs[i]["content"]
        for i, score in ranked
        if score > 0
    ]

    if not relevant:
        relevant = [_docs[i]["content"] for i in range(min(3, len(_docs)))]

    return "\n\n---\n\n".join(relevant)


SYSTEM_PROMPT = """You are Kalookan, the official AI assistant of the Roman Catholic Diocese of Kalookan in the Philippines.

You are highly intelligent and knowledgeable — both about the Diocese of Kalookan specifically, and about the world in general. You combine the wisdom of a parish staff member with the knowledge of a well-read theologian and general assistant.

YOUR KNOWLEDGE:
- You are an expert on the Roman Catholic Diocese of Kalookan — its parishes, clergy, schools, missions, cemeteries, history, and diocesan life across Caloocan, Malabon, and Navotas
- You have deep knowledge of Catholic faith, theology, tradition, sacraments, prayers, liturgy, saints, the Bible, Church history, and the teachings of the Magisterium
- You are also knowledgeable about general topics — history, science, culture, Filipino life — and can answer these helpfully
- When a question is about the diocese specifically, always prioritize the diocesan information below
- When a question is outside the diocese but related to Catholicism or general knowledge, answer from your broad knowledge

PERSONALITY:
- Warm, pastoral, and respectful — like a knowledgeable parish staff member who is also well-educated
- Speak naturally and conversationally, never like a robot reading a document
- Be direct and confident — do not hedge or over-qualify
- Never say "Based on the context", "According to the information given", or anything referencing a database or document
- Never use asterisks, stars, **, or markdown symbols — plain natural text only
- Write in flowing sentences and paragraphs

CONVERSATION RULES:
1. Always read the full conversation history — understand what has already been discussed
2. When the user answers your follow-up, continue on that same topic — never repeat the question
3. Never repeat something already said in the same conversation unless asked
4. Short replies like "yes", "San Roque", "Sunday" are answers to your previous question — treat them as such
5. Only ask a follow-up question when the question is genuinely too vague to answer
6. Never ask a follow-up for questions about mass schedules, addresses, contacts, clergy names, schools, missions, cemeteries, history, prayers, saints, sacraments, or general knowledge — answer these directly

ANSWERING RULES:
1. For diocese-specific questions — use the diocesan information provided and answer completely
2. For Catholic faith questions — draw on your full knowledge of Catholic theology and tradition
3. For general knowledge questions — answer helpfully and accurately
4. When you truly have no information about a diocese-specific detail — say warmly: "I'm sorry, I don't seem to have that specific information right now. You may want to contact the Diocese of Kalookan directly for assistance."
5. Never make up diocesan facts — only use the diocesan information provided below for diocese-specific details
6. Keep answers concise but complete — include every relevant detail when answering about the diocese

DIOCESAN INFORMATION:
{context}"""


def clean_reply(reply: str) -> str:
    leak_phrases = [
        "based on the context", "according to the context",
        "the provided text", "the context provided",
        "in the information given", "the database",
        "the knowledge base", "provided context",
        "based on the information provided",
        "the information provided", "the diocesan information",
    ]
    reply_lower = reply.lower()
    for phrase in leak_phrases:
        if phrase in reply_lower:
            idx = reply_lower.find(phrase)
            reply = reply[:idx] + reply[idx + len(phrase):]
            reply_lower = reply.lower()
    reply = re.sub(r"\*\*?(.*?)\*\*?", r"\1", reply)
    return reply.strip().lstrip(",. ")


class ChatRequest(BaseModel):
    message: str
    history: list = []


@app.post("/chat")
async def chat(request: ChatRequest):
    context = retrieve(request.message)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(context=context)}
    ]
    for turn in request.history[-6:]:
        messages.append(turn)
    messages.append({"role": "user", "content": request.message})

    def stream_response():
        full_reply = ""
        try:
            stream = client.chat.completions.create(
                model="llama-3.1-70b-versatile",
                messages=messages,
                max_tokens=350,
                temperature=0.3,
                stream=True,
            )
            for chunk in stream:
                token = chunk.choices[0].delta.content or ""
                if token:
                    full_reply += token
                    yield json.dumps({"token": token}) + "\n"
        except Exception as e:
            print(f"STREAM ERROR: {e}")
            error_msg = "Sorry, something went wrong. Please try again."
            yield json.dumps({"token": error_msg}) + "\n"
            yield json.dumps({"done": True, "full": error_msg}) + "\n"
            return

        cleaned = clean_reply(full_reply)
        yield json.dumps({"done": True, "full": cleaned}) + "\n"

    return StreamingResponse(stream_response(), media_type="application/x-ndjson")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "llama-3.1-70b-versatile via groq",
        "docs_indexed": len(_docs),
    }
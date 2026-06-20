from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq, RateLimitError
import os
import re
import json
import time
import math
import glob
import requests as req_lib

# ── BM25 ──────────────────────────────────────────────────────────────────────
from rank_bm25 import BM25Okapi

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(title="RCDoK Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Groq client ───────────────────────────────────────────────────────────────
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
MODEL  = "gemma2-9b-it"

# ── Token budget (Groq free tier: 15,000 TPM for gemma2-9b-it) ───────────────
# Keep each request well under limit to avoid rate errors.
# System prompt ~ 500 tokens, history ~ 300, response ~ 250 → budget 800 for context
MAX_CONTEXT_CHARS = 2400   # ~600 tokens at ~4 chars/token
MAX_TOKENS_OUT    = 250
MAX_HISTORY_TURNS = 4      # last 4 turns (2 exchanges)

# ── BM25 index — load all knowledge_base/*.txt at startup ─────────────────────
print("Loading BM25 index...")

KB_DIR = os.path.join(os.path.dirname(__file__), "knowledge_base")
_docs: list[dict] = []   # {"text": str, "name": str}

for path in sorted(glob.glob(os.path.join(KB_DIR, "*.txt"))):
    try:
        with open(path, encoding="utf-8") as f:
            raw = f.read().strip()
        name = os.path.splitext(os.path.basename(path))[0]
        # Split into ~300-char chunks with 50-char overlap
        step = 250
        for i in range(0, max(1, len(raw) - 50), step):
            chunk = raw[i : i + 300].strip()
            if len(chunk) > 40:
                _docs.append({"text": chunk, "name": name})
    except Exception as e:
        print(f"  Warning: could not load {path}: {e}")

_tokenized = [re.findall(r"\w+", d["text"].lower()) for d in _docs]
_bm25      = BM25Okapi(_tokenized)

print(f"BM25 index ready — {len(_docs)} chunks from {len(set(d['name'] for d in _docs))} documents.")

# ── Retrieval ──────────────────────────────────────────────────────────────────
STOP = {
    "the","a","an","is","are","what","who","where","when","how","does","do",
    "of","in","at","for","and","or","to","can","tell","me","about","please",
    "its","it","my","our","your","their","i","was","be","been","has","have",
    "had","will","would","could","should","give","list","show","find","get",
    "know","need","want","po","ba","ang","mga","ang","na","ng","sa","si","ni",
}

def retrieve(query: str, history: list, top_k: int = 3) -> str:
    # Enrich query with last 2 history turns for follow-up awareness
    history_tail = " ".join(
        t.get("content", "") for t in history[-4:]
    )
    enriched = (query + " " + history_tail).strip()

    tokens = [w for w in re.findall(r"\w+", enriched.lower()) if w not in STOP and len(w) > 2]
    if not tokens:
        return ""

    scores  = _bm25.get_scores(tokens)
    indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    seen, chunks = set(), []
    total_chars  = 0

    for idx in indices:
        if scores[idx] < 0.01:
            break
        text = _docs[idx]["text"]
        key  = text[:60]
        if key in seen:
            continue
        seen.add(key)
        if total_chars + len(text) > MAX_CONTEXT_CHARS:
            break
        chunks.append(text)
        total_chars += len(text)
        if len(chunks) >= top_k:
            break

    return "\n\n---\n\n".join(chunks)

# ── Groq call with exponential backoff ────────────────────────────────────────
def groq_chat(messages: list, stream: bool = False, max_retries: int = 4):
    """
    Call Groq with exponential backoff on RateLimitError.
    Delays: 2s, 4s, 8s, 16s before giving up.
    """
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(
                model=MODEL,
                messages=messages,
                max_tokens=MAX_TOKENS_OUT,
                temperature=0.3,
                stream=stream,
            )
        except RateLimitError:
            if attempt == max_retries - 1:
                raise
            wait = 2 ** (attempt + 1)   # 2, 4, 8, 16
            print(f"Rate limit hit — retrying in {wait}s (attempt {attempt+1}/{max_retries})")
            time.sleep(wait)
        except Exception:
            raise

# ── Prompt cleaning ────────────────────────────────────────────────────────────
_LEAK_PHRASES = [
    "based on the context","according to the context","the provided text",
    "the context provided","in the information given","the database",
    "the knowledge base","provided context","based on the information provided",
    "the information provided","the diocesan information","based on the provided",
]

def clean_reply(reply: str) -> str:
    low = reply.lower()
    for phrase in _LEAK_PHRASES:
        if phrase in low:
            idx   = low.find(phrase)
            reply = reply[:idx] + reply[idx + len(phrase):]
            low   = reply.lower()
    reply = re.sub(r"\*\*?(.*?)\*\*?", r"\1", reply)   # strip markdown bold/italic
    reply = re.sub(r"#{1,6}\s*",        "",    reply)   # strip markdown headers
    return reply.strip().lstrip(",. ")

# ── System prompt ──────────────────────────────────────────────────────────────
# Deliberately kept tight — every token here costs against the 15K TPM budget.
SYSTEM_PROMPT = """\
You are Kalookan, the AI assistant of the Roman Catholic Diocese of Kalookan, Philippines.

IDENTITY: Warm, pastoral, direct — like a knowledgeable parish staff member.

RULES:
- Answer diocese questions using ONLY the CONTEXT below. Never invent diocesan facts.
- Answer Catholic faith and general knowledge questions from your own knowledge.
- Never reference "the context", "the database", or any system internals.
- Never use **, *, or markdown. Write in plain flowing sentences.
- Short follow-up replies ("yes", "San Roque", "Sunday") are answers to your last question — respond accordingly, never repeat the question.
- Only ask a follow-up when the question is genuinely too vague (e.g. no parish specified).
- Keep answers concise. For diocese details, include all relevant facts you have.
- If a diocese-specific detail is missing, say: "I don't have that detail right now. Please contact the Diocese of Kalookan directly."
- You may respond in Filipino/Tagalog if the user writes in Filipino.

CONTEXT:
{context}\
"""

# ── Request model ──────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    history: list = []

# ── Messenger state (in-memory, resets on restart — acceptable for free tier) ──
_messenger_history: dict[str, list] = {}

# ── Web chat endpoint (streaming) ─────────────────────────────────────────────
@app.post("/chat")
async def chat(request: ChatRequest):
    context = retrieve(request.message, request.history)

    messages = [{"role": "system", "content": SYSTEM_PROMPT.format(context=context)}]
    for turn in request.history[-(MAX_HISTORY_TURNS * 2):]:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": request.message})

    print(f"USER: {request.message!r}  |  ctx_chars={len(context)}  |  history={len(request.history)}")

    def stream_response():
        full_reply = ""
        try:
            stream = groq_chat(messages, stream=True)
            for chunk in stream:
                token = chunk.choices[0].delta.content or ""
                if token:
                    full_reply += token
                    yield json.dumps({"token": token}) + "\n"
        except RateLimitError:
            yield json.dumps({"token": " The assistant is busy right now. Please try again in a moment."}) + "\n"
        except Exception as e:
            print(f"Stream error: {e}")
            yield json.dumps({"token": " Sorry, something went wrong. Please try again."}) + "\n"

        cleaned = clean_reply(full_reply)
        yield json.dumps({"done": True, "full": cleaned}) + "\n"

    return StreamingResponse(stream_response(), media_type="application/x-ndjson")

# ── Messenger webhook ──────────────────────────────────────────────────────────
PAGE_ACCESS_TOKEN = os.environ.get("PAGE_ACCESS_TOKEN", "")
VERIFY_TOKEN      = os.environ.get("VERIFY_TOKEN", "")

def _send_messenger(recipient_id: str, text: str) -> None:
    if not PAGE_ACCESS_TOKEN:
        return
    req_lib.post(
        f"https://graph.facebook.com/v19.0/me/messages?access_token={PAGE_ACCESS_TOKEN}",
        json={"recipient": {"id": recipient_id}, "message": {"text": text}, "messaging_type": "RESPONSE"},
        timeout=10,
    )

def _typing_on(recipient_id: str) -> None:
    if not PAGE_ACCESS_TOKEN:
        return
    req_lib.post(
        f"https://graph.facebook.com/v19.0/me/messages?access_token={PAGE_ACCESS_TOKEN}",
        json={"recipient": {"id": recipient_id}, "sender_action": "typing_on"},
        timeout=5,
    )

@app.get("/webhook")
async def verify_webhook(
    hub_mode: str = None,
    hub_verify_token: str = None,
    hub_challenge: str = None,
):
    if hub_mode == "subscribe" and hub_verify_token == VERIFY_TOKEN:
        print("Webhook verified.")
        return int(hub_challenge)
    return {"error": "Verification failed"}, 403

@app.post("/webhook")
async def receive_message(req: Request):
    body = await req.json()
    for entry in body.get("entry", []):
        for event in entry.get("messaging", []):
            sender_id = event["sender"]["id"]
            msg       = event.get("message", {})
            text      = msg.get("text", "").strip()
            if not text:
                continue

            _typing_on(sender_id)

            history = _messenger_history.get(sender_id, [])
            context = retrieve(text, history)

            messages = [{"role": "system", "content": SYSTEM_PROMPT.format(context=context)}]
            for turn in history[-(MAX_HISTORY_TURNS * 2):]:
                messages.append(turn)
            messages.append({"role": "user", "content": text})

            try:
                response = groq_chat(messages, stream=False)
                reply    = clean_reply(response.choices[0].message.content)
            except RateLimitError:
                reply = "The assistant is busy right now. Please try again in a moment."
            except Exception as e:
                print(f"Messenger error: {e}")
                reply = "Sorry, something went wrong. Please try again later."

            history.append({"role": "user",      "content": text})
            history.append({"role": "assistant",  "content": reply})
            _messenger_history[sender_id] = history[-(MAX_HISTORY_TURNS * 2):]

            _send_messenger(sender_id, reply)

    return {"status": "ok"}

# ── Health ─────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status":       "ok",
        "model":        MODEL,
        "docs_indexed": len(_docs),
    }
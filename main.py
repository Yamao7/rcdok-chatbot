from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from groq import Groq, RateLimitError
from rank_bm25 import BM25Okapi
import os, re, json, time, glob, requests as req_lib

# ── app setup ──────────────────────────────────────
app = FastAPI(title="RCDoK Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
MODEL  = "openai/gpt-oss-120b"

MAX_CONTEXT_CHARS = 3000
MAX_CHARS_PER_DOC = 900
MAX_DOCS_RETURNED = 4
MAX_TOKENS_OUT    = 700
MAX_HISTORY_TURNS = 4

# ── knowledge base / bm25 index ───────────────────
print("Loading knowledge base...")
KB_DIR = os.path.join(os.path.dirname(__file__), "cleaned_knowledge_base")
_docs: list[dict] = []

for path in sorted(glob.glob(os.path.join(KB_DIR, "*.txt"))):
    try:
        with open(path, encoding="utf-8") as f:
            raw = f.read().strip()
        name = os.path.splitext(os.path.basename(path))[0]
        if raw:
            _docs.append({"text": raw, "name": name})
    except Exception as e:
        print(f"  skipped {path}: {e}")

_tokenized = [re.findall(r"\w+", d["text"].lower()) for d in _docs]
_bm25      = BM25Okapi(_tokenized)
print(f"BM25 ready — {len(_docs)} whole documents indexed.")

STOP = {
    "the","a","an","is","are","what","who","where","when","how","does","do",
    "of","in","at","for","and","or","to","can","tell","me","about","please",
    "its","it","my","our","your","their","i","was","be","been","has","have",
    "had","will","would","could","should","give","list","show","find","get",
    "know","need","want","po","ba","ang","mga","na","ng","sa","si","ni",
}

QUERY_EXPAND = {
    "priests":          "clergy fr father bishop vicar rector diocesan",
    "diocesan priests": "clergy fr father bishop vicar rector diocesan",
    "parish priests":   "clergy fr father bishop vicar rector parish",
    "mission centers":  "missions center location address headquarters station",
    "mission":          "missions center location address headquarters station",
    "mission stations": "missions station chapel location",
    "schools":          "school member education college academy",
    "cemeteries":       "cemetery columbary ossuary burial",
    "coat of arms":     "coat arms crest emblem heraldry symbol",
    "history":          "history founded established year diocese",
}

# queries that are meaningless without a specific parish name —
# if no parish name is present, let the model ask instead of guessing
VAGUE_WITHOUT_PARISH = {
    "mass schedule", "mass schedules", "schedule of mass", "confession hours",
    "parish office", "parish priest", "parochial vicar", "contact number",
    "church schedule",
}

def is_vague_parish_query(query: str) -> bool:
    q = query.lower()
    if not any(phrase in q for phrase in VAGUE_WITHOUT_PARISH):
        return False
    for doc in _docs:
        parish_words = [w for w in re.findall(r"\w+", doc["name"].lower()) if len(w) > 3]
        if parish_words and any(w in q for w in parish_words):
            return False
    return True

def retrieve(query: str, history: list) -> str:
    if is_vague_parish_query(query):
        return ""

    query_lower = query.lower()
    expanded    = query
    for key, expansion in QUERY_EXPAND.items():
        if key in query_lower:
            expanded = query + " " + expansion
            break

    tail     = " ".join(t.get("content", "") for t in history[-4:])
    enriched = (expanded + " " + tail).strip()
    tokens   = [w for w in re.findall(r"\w+", enriched.lower()) if w not in STOP and len(w) > 2]
    if not tokens:
        return ""

    scores  = _bm25.get_scores(tokens)
    indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    chunks, total = [], 0
    for idx in indices:
        if scores[idx] < 0.001:
            break
        text = _docs[idx]["text"]
        if len(text) > MAX_CHARS_PER_DOC:
            text = text[:MAX_CHARS_PER_DOC]
        if total + len(text) > MAX_CONTEXT_CHARS:
            remaining = MAX_CONTEXT_CHARS - total
            if remaining > 300:
                chunks.append(text[:remaining])
            break
        chunks.append(text)
        total += len(text)
        if len(chunks) >= MAX_DOCS_RETURNED:
            break

    return "\n\n---\n\n".join(chunks)

# ── groq call ──────────────────────────────────────
def groq_call(messages: list, stream: bool = False):
    for attempt in range(4):
        try:
            return client.chat.completions.create(
                model=MODEL,
                messages=messages,
                max_tokens=MAX_TOKENS_OUT,
                temperature=0.3,
                stream=stream,
            )
        except RateLimitError:
            if attempt == 3:
                raise
            wait = 2 ** (attempt + 1)
            print(f"Rate limit — waiting {wait}s")
            time.sleep(wait)
        except Exception:
            raise

# ── reply cleaning ─────────────────────────────────
_LEAK_RE = re.compile(
    r'[^.!?]*(?:based on the context|according to the context|the provided text|'
    r'the context provided|in the information given|the knowledge base|provided context|'
    r'based on the information provided|the information provided|the diocesan information|'
    r'based on the provided|in the context|from the context|the context does not|'
    r'the context only|not listed in|not mentioned in|not included in|not specified in|'
    r'not found in|not available in)[^.!?]*[.!?]?',
    re.IGNORECASE
)

def clean(reply: str) -> str:
    reply = _LEAK_RE.sub("", reply)
    reply = re.sub(r"\*\*?(.*?)\*\*?", r"\1", reply)
    reply = re.sub(r"#{1,6}\s*", "", reply)
    reply = re.sub(r"(?<!\n)[ \t]*[•\-–]\s+(?=[A-Z0-9])", r"\n- ", reply)
    reply = re.sub(r"(?<!\n)(\d+\.)\s+", r"\n\1 ", reply)
    reply = re.sub(r"\n{3,}", "\n\n", reply)
    return reply.strip().lstrip(",. ")

SYSTEM_PROMPT = """\
You are Kalookan, the official AI assistant of the Roman Catholic Diocese of Kalookan, Philippines.
You speak like a warm, knowledgeable parish staff member — direct, pastoral, never robotic.

ABSOLUTE RULES — follow these without exception:
1. Never say "the context", "the database", "the provided information", "not listed in", "not mentioned in", or any phrase that reveals you are working from a document. You simply know this or you don't.
2. Never use **, *, #, or markdown of any kind. Plain text only.
3. When listing priests, parishes, schools, or any named items — put each item on its own line, prefixed with a number like "1. ". Never put multiple items on the same line. Never truncate a list. Never say "and more" or "among others" — list everything given to you.
4. Never cut off mid-sentence. Complete every thought.
5. Short replies like "yes", "San Roque", or "Sunday" are follow-up answers — treat them as such.
6. If asked about mass schedule, confession hours, parish priest, parochial vicar, contact number, or church schedule WITHOUT a specific parish named, always ask which parish first. Never guess or pick a parish yourself.
7. Only ask a clarifying question when the query is genuinely impossible to answer without it — for anything else, answer directly.
8. If you truly have no information on a diocese-specific detail, say exactly: "I don't have that detail right now. You can reach the Diocese of Kalookan directly through their Facebook page or website for the most up-to-date information."
9. Never add that fallback phrase unless you genuinely have nothing. If partial information exists, give it in full.
10. Never invent specific times, numbers, addresses, or schedules. If exact figures are not in the DIOCESE INFORMATION below, say you don't have that detail — never estimate or guess numbers to sound helpful.
11. You may respond in Filipino or Tagalog if the user writes in Filipino.
12. For Catholic faith and general knowledge questions not specific to the diocese, answer from your own knowledge.

DIOCESE INFORMATION:
{context}"""

class ChatRequest(BaseModel):
    message: str
    history: list = []

_messenger_history: dict[str, list] = {}

# ── web chat ───────────────────────────────────────
@app.post("/chat")
async def chat(request: ChatRequest):
    context  = retrieve(request.message, request.history)
    messages = [{"role": "system", "content": SYSTEM_PROMPT.format(context=context)}]
    for turn in request.history[-(MAX_HISTORY_TURNS * 2):]:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": request.message})

    print(f"USER: {request.message!r} | ctx={len(context)}chars")

    def stream_response():
        full_reply = ""
        try:
            for chunk in groq_call(messages, stream=True):
                token = chunk.choices[0].delta.content or ""
                if token:
                    full_reply += token
                    yield json.dumps({"token": token}) + "\n"
        except RateLimitError:
            yield json.dumps({"token": "The assistant is busy right now. Please try again in a moment."}) + "\n"
        except Exception as e:
            print(f"Stream error: {e}")
            yield json.dumps({"token": "Sorry, something went wrong. Please try again."}) + "\n"
        yield json.dumps({"done": True, "full": clean(full_reply)}) + "\n"

    return StreamingResponse(stream_response(), media_type="application/x-ndjson")

# ── messenger webhook ──────────────────────────────
PAGE_ACCESS_TOKEN = os.environ.get("PAGE_ACCESS_TOKEN", "")
VERIFY_TOKEN      = os.environ.get("VERIFY_TOKEN", "")

def messenger_send(recipient_id: str, text: str):
    if not PAGE_ACCESS_TOKEN:
        return
    req_lib.post(
        f"https://graph.facebook.com/v19.0/me/messages?access_token={PAGE_ACCESS_TOKEN}",
        json={"recipient": {"id": recipient_id}, "message": {"text": text}, "messaging_type": "RESPONSE"},
        timeout=10,
    )

def messenger_typing(recipient_id: str):
    if not PAGE_ACCESS_TOKEN:
        return
    req_lib.post(
        f"https://graph.facebook.com/v19.0/me/messages?access_token={PAGE_ACCESS_TOKEN}",
        json={"recipient": {"id": recipient_id}, "sender_action": "typing_on"},
        timeout=5,
    )

@app.get("/webhook")
async def verify_webhook(hub_mode: str = None, hub_verify_token: str = None, hub_challenge: str = None):
    if hub_mode == "subscribe" and hub_verify_token == VERIFY_TOKEN:
        return int(hub_challenge)
    return {"error": "Verification failed"}

@app.post("/webhook")
async def receive_message(req: Request):
    body = await req.json()
    for entry in body.get("entry", []):
        for event in entry.get("messaging", []):
            sender_id = event["sender"]["id"]
            text      = event.get("message", {}).get("text", "").strip()
            if not text:
                continue

            messenger_typing(sender_id)
            history  = _messenger_history.get(sender_id, [])
            context  = retrieve(text, history)
            messages = [{"role": "system", "content": SYSTEM_PROMPT.format(context=context)}]
            for turn in history[-(MAX_HISTORY_TURNS * 2):]:
                messages.append(turn)
            messages.append({"role": "user", "content": text})

            try:
                response = groq_call(messages, stream=False)
                reply    = clean(response.choices[0].message.content)
            except RateLimitError:
                reply = "The assistant is busy right now. Please try again in a moment."
            except Exception as e:
                print(f"Messenger error: {e}")
                reply = "Sorry, something went wrong. Please try again later."

            history.append({"role": "user",     "content": text})
            history.append({"role": "assistant", "content": reply})
            _messenger_history[sender_id] = history[-(MAX_HISTORY_TURNS * 2):]
            messenger_send(sender_id, reply)

    return {"status": "ok"}

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL, "docs_indexed": len(_docs)}

# ── static frontend ────────────────────────────────
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/")
    def index():
        return FileResponse(os.path.join(static_dir, "index.html"))
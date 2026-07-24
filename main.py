from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from groq import Groq, RateLimitError
from rank_bm25 import BM25Okapi
import os, re, json, time, glob, unicodedata, requests as req_lib

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

MAX_CONTEXT_CHARS = 3500
MAX_DOCS_RETURNED = 4
MAX_TOKENS_OUT    = 1100
MAX_HISTORY_TURNS = 4
TITLE_BOOST_WEIGHT = 2.0

# ── text normalization helpers ────────────────────
def normalize(text: str) -> str:
    """Strip accents for tokenizing/matching only — never used for display text."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))

def tokenize(text: str) -> list:
    return re.findall(r"\w+", normalize(text).lower())

HEADER_RE = re.compile(r"^PAGE:.*\n?SOURCE:.*\n*", re.IGNORECASE)

# ── knowledge base / bm25 index ───────────────────
print("Loading knowledge base...")
KB_DIR = os.path.join(os.path.dirname(__file__), "cleaned_knowledge_base")
_docs: list[dict] = []

# roster-style files get split into one chunk per person instead of
# being indexed as one giant block — otherwise a single-priest query
# returns the entire 40-person directory
ROSTER_FILES = {"(cleaned) Clergy - Diocesan", "(cleaned) Clergy - Religious"}

def chunk_roster(text: str, doc_name: str) -> list:
    entries = re.split(r"\n\n(?=\S)", text.strip())
    chunks = []
    for entry in entries:
        entry = entry.strip()
        if len(entry) > 20:
            chunks.append({"text": entry, "name": doc_name})
    return chunks

for path in sorted(glob.glob(os.path.join(KB_DIR, "*.txt"))):
    try:
        with open(path, encoding="utf-8") as f:
            raw = f.read().strip()
        raw = HEADER_RE.sub("", raw)  # strip PAGE:/SOURCE: header lines
        name = os.path.splitext(os.path.basename(path))[0]
        if not raw:
            continue
        if name in ROSTER_FILES:
            _docs.extend(chunk_roster(raw, name))
        else:
            _docs.append({"text": raw, "name": name})
    except Exception as e:
        print(f"  skipped {path}: {e}")

_tokenized = [tokenize(d["text"]) for d in _docs]
# b lowered from default 0.75 — our docs vary a lot in length (mission
# station blurbs vs. full parish histories) and length here reflects
# real content richness, not padding, so we don't want to penalize it hard
_bm25 = BM25Okapi(_tokenized, k1=1.2, b=0.4)
print(f"BM25 ready — {len(_docs)} documents indexed ({len(ROSTER_FILES)} roster files chunked).")

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

# words too generic to count as identifying a specific parish
GENERIC_PARISH_WORDS = {
    "cleaned","parish","parishes","quasi","vicariate","diocesan","shrine",
    "of","and","the","san","sta","sto","de","los","las","our","lady",
    "clergy","religious",  # exclude roster file names from parish matching
}

def short_name(doc_name: str) -> str:
    parts = doc_name.split(" - ")
    return parts[-1] if parts else doc_name

def significant_words(name: str) -> list:
    words = re.findall(r"\w+", normalize(name).lower())
    return [w for w in words if w not in GENERIC_PARISH_WORDS and len(w) > 2]

# roster files are excluded here — they aren't named parishes, and letting
# "clergy" or "diocesan" match would incorrectly short-circuit into
# find_named_parish and return only one priest instead of the full context
_parish_sig = [
    (doc, significant_words(short_name(doc["name"])))
    for doc in _docs
    if doc["name"] not in ROSTER_FILES
]

def find_named_parish(query: str):
    """direct lookup — matches a parish by name regardless of conversation noise"""
    q_words = set(tokenize(query))
    best, best_score = None, 0
    for doc, sig in _parish_sig:
        if not sig:
            continue
        overlap = len(set(sig) & q_words)
        ratio   = overlap / len(sig)
        if overlap >= 1 and (ratio >= 0.6 or overlap >= 2):
            if overlap > best_score:
                best_score, best = overlap, doc
    return best

# phrases that need a named parish to mean anything
VAGUE_WITHOUT_PARISH = {
    "mass schedule", "mass schedules", "schedule of mass", "confession hours",
    "parish office", "parish priest", "parochial vicar", "contact number",
    "church schedule",
}

def is_vague_parish_query(query: str) -> bool:
    q = query.lower()
    return any(phrase in q for phrase in VAGUE_WITHOUT_PARISH)

ANY_FALLBACK_TRIGGERS = {"any", "kalookan", "cathedral", "main parish", "any parish", "some parish"}

def title_boosted_scores(tokens: list) -> list:
    """Base BM25 scores, boosted when query terms also appear in the doc's own title."""
    scores = list(_bm25.get_scores(tokens))
    token_set = set(tokens)
    for i, doc in enumerate(_docs):
        title_words = set(tokenize(doc["name"]))
        overlap = len(token_set & title_words)
        if overlap:
            scores[i] += overlap * TITLE_BOOST_WEIGHT
    return scores

def retrieve(query: str, history: list) -> str:
    # 1. try a direct named-parish match first — bypasses history noise entirely
    matched = find_named_parish(query)
    if matched:
        return matched["text"][:MAX_CONTEXT_CHARS]

    # 2. vague schedule/contact-type question with no parish named
    if is_vague_parish_query(query):
        q_lower = query.lower()
        if any(trigger in q_lower for trigger in ANY_FALLBACK_TRIGGERS):
            cathedral = next((d for d in _docs if "cathedral" in d["name"].lower()), None)
            if cathedral:
                return cathedral["text"][:MAX_CONTEXT_CHARS]
        return ""  # let the model ask which parish

    # 3. normal bm25 retrieval for everything else
    query_lower = query.lower()
    expanded    = query
    for key, expansion in QUERY_EXPAND.items():
        if key in query_lower:
            expanded = query + " " + expansion
            break

    tokens = [w for w in tokenize(expanded) if w not in STOP and len(w) > 2]
    if not tokens:
        return ""

    scores  = title_boosted_scores(tokens)
    indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    chunks, total, seen_names = [], 0, set()
    for idx in indices:
        if scores[idx] < 0.001:
            break
        doc = _docs[idx]
        if doc["name"] in ROSTER_FILES:
            key = doc["name"]
            count = sum(1 for n in seen_names if n == key)
            if count >= 2:
                continue
            seen_names.add(key)

        text = doc["text"]
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
You speak like a warm, patient parish staff member helping any visitor, including elderly or unfamiliar users — direct, pastoral, never robotic, never curt.

ABSOLUTE RULES — follow these without exception:
1. Never say "the context", "the database", "the provided information", "not listed in", "not mentioned in", or any phrase that reveals you are working from a document. You simply know this or you don't.
2. Never use **, *, #, or markdown of any kind. Plain text only.
3. When listing priests, parishes, schools, or any named items — put each item on its own line, prefixed with a number like "1. ". Never truncate a list, never say "and more" — write out every single one given to you, no matter how long the list is.
4. Never cut off mid-sentence or mid-list. Complete every thought and every list fully.
5. Short replies like "yes", "San Roque", or "Sunday" are follow-up answers to your last question — treat them as such, never repeat the question.
6. If asked about mass schedule, confession hours, parish priest, or contact info without a parish named, and you have no specific parish information provided below, kindly ask which parish they mean, and mention as an example that you can share San Roque Cathedral's schedule if they are not sure which parish serves their area.
7. Be forgiving of vague, casual, or imprecise questions — never refuse or give up after one unclear reply. Gently guide the person toward an answer instead of repeating the same clarifying question.
8. If you truly have no information on a diocese-specific detail even after trying to help, say: "I don't have that detail right now. You can reach the Diocese of Kalookan directly through their Facebook page or website for the most up-to-date information." Use this only as a last resort, never as a first response to a vague question.
9. Never invent specific times, numbers, addresses, or schedules. If exact figures are not in the DIOCESE INFORMATION below, say so plainly rather than guessing.
10. You may respond in Filipino or Tagalog if the user writes in Filipino.
11. For Catholic faith and general knowledge questions not specific to the diocese, answer from your own knowledge.

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
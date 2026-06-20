from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq, RateLimitError
from rank_bm25 import BM25Okapi
import os, re, json, time, glob, requests as req_lib

app = FastAPI(title="RCDoK Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
MODEL  = "llama-3.1-8b-instant"

MAX_CONTEXT_CHARS = 2400
MAX_TOKENS_OUT    = 250
MAX_HISTORY_TURNS = 4

print("Loading knowledge base...")
KB_DIR = os.path.join(os.path.dirname(__file__), "knowledge_base")
_docs: list[dict] = []

for path in sorted(glob.glob(os.path.join(KB_DIR, "*.txt"))):
    try:
        with open(path, encoding="utf-8") as f:
            raw = f.read().strip()
        name = os.path.splitext(os.path.basename(path))[0]
        for i in range(0, max(1, len(raw) - 50), 250):
            chunk = raw[i : i + 300].strip()
            if len(chunk) > 40:
                _docs.append({"text": chunk, "name": name})
    except Exception as e:
        print(f"  skipped {path}: {e}")

_tokenized = [re.findall(r"\w+", d["text"].lower()) for d in _docs]
_bm25      = BM25Okapi(_tokenized)
print(f"BM25 ready — {len(_docs)} chunks from {len(set(d['name'] for d in _docs))} documents.")

STOP = {
    "the","a","an","is","are","what","who","where","when","how","does","do",
    "of","in","at","for","and","or","to","can","tell","me","about","please",
    "its","it","my","our","your","their","i","was","be","been","has","have",
    "had","will","would","could","should","give","list","show","find","get",
    "know","need","want","po","ba","ang","mga","na","ng","sa","si","ni",
}

def retrieve(query: str, history: list) -> str:
    tail     = " ".join(t.get("content", "") for t in history[-4:])
    enriched = (query + " " + tail).strip()
    tokens   = [w for w in re.findall(r"\w+", enriched.lower()) if w not in STOP and len(w) > 2]
    if not tokens:
        return ""

    scores  = _bm25.get_scores(tokens)
    indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    seen, chunks, total = set(), [], 0
    for idx in indices:
        if scores[idx] < 0.01:
            break
        text = _docs[idx]["text"]
        key  = text[:60]
        if key in seen:
            continue
        seen.add(key)
        if total + len(text) > MAX_CONTEXT_CHARS:
            break
        chunks.append(text)
        total += len(text)
        if len(chunks) >= 3:
            break

    return "\n\n---\n\n".join(chunks)

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

LEAK_PHRASES = [
    "based on the context","according to the context","the provided text",
    "the context provided","in the information given","the knowledge base",
    "provided context","based on the information provided",
    "the information provided","the diocesan information","based on the provided",
]

def clean(reply: str) -> str:
    low = reply.lower()
    for phrase in LEAK_PHRASES:
        if phrase in low:
            idx   = low.find(phrase)
            reply = reply[:idx] + reply[idx + len(phrase):]
            low   = reply.lower()
    reply = re.sub(r"\*\*?(.*?)\*\*?", r"\1", reply)
    reply = re.sub(r"#{1,6}\s*", "", reply)
    return reply.strip().lstrip(",. ")

SYSTEM_PROMPT = """\
You are Kalookan, the AI assistant of the Roman Catholic Diocese of Kalookan, Philippines.

You are warm, pastoral, and direct — like a knowledgeable parish staff member.

RULES:
- Answer diocese questions using ONLY the CONTEXT below. Never invent diocesan facts.
- Answer Catholic faith and general knowledge questions from your own knowledge.
- Never reference "the context", "the database", or any system internals.
- Never use **, *, #, or any markdown. Write in plain flowing sentences.
- Short follow-up replies like "yes", "San Roque", or "Sunday" are answers to your last question — respond accordingly, never repeat the question.
- Only ask a follow-up when the question is genuinely too vague to answer.
- Keep answers concise. For diocese details, include all relevant facts you have.
- If a diocese-specific detail is missing say: "I don't have that detail right now. Please contact the Diocese of Kalookan directly."
- You may respond in Filipino or Tagalog if the user writes in Filipino.

CONTEXT:
{context}"""

class ChatRequest(BaseModel):
    message: str
    history: list = []

_messenger_history: dict[str, list] = {}

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
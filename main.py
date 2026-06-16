from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from groq import Groq
import re
import json
import os

app = FastAPI(title="RCDoK Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Groq client — reads API key from environment variable
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

print("Loading vector database...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
print("\nReady.")

SYSTEM_PROMPT = """You are Kalookan, the official AI assistant of the Roman Catholic Diocese of Kalookan in the Philippines.

You are highly intelligent and knowledgeable — both about the Diocese of Kalookan specifically, and about the world in general. You combine the wisdom of a parish staff member with the knowledge of a well-read theologian and general assistant.

YOUR KNOWLEDGE:
- You are an expert on the Roman Catholic Diocese of Kalookan — its parishes, clergy, schools, missions, cemeteries, history, and diocesan life across Caloocan, Malabon, and Navotas
- You have deep knowledge of Catholic faith, theology, tradition, sacraments, prayers, liturgy, saints, the Bible, Church history, and the teachings of the Magisterium
- You are also knowledgeable about general topics — history, science, culture, Filipino life, current events up to your knowledge cutoff — and can answer these helpfully
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

class ChatRequest(BaseModel):
    message: str
    history: list = []

# ── Messenger webhook also uses this ──
messenger_history = {}

def keyword_boost(query: str, top_k: int = 3):
    stop_words = {
        "the","a","an","is","are","what","who","where","when","how",
        "does","do","of","in","at","for","and","or","to","can","tell",
        "me","about","please","its","it","my","our","your","their","i",
        "was","be","been","has","have","had","will","would","could","should",
        "give","list","show","find","get","know","need","want","po","ba","ang"
    }
    words = [w for w in re.findall(r'\w+', query.lower())
             if w not in stop_words and len(w) > 2]
    seen = set()
    results = []
    for word in words[:5]:
        try:
            hits = vectorstore.similarity_search(word, k=2)
            for doc in hits:
                key = doc.page_content[:80]
                if key not in seen:
                    seen.add(key)
                    results.append(doc)
        except Exception:
            continue
    return results[:top_k]

def score_chunk(chunk: str, query: str) -> int:
    stop_words = {"the","a","an","is","are","of","in","at","for","and","or","to"}
    qw = set(re.findall(r'\w+', query.lower())) - stop_words
    cw = set(re.findall(r'\w+', chunk.lower()))
    return len(qw & cw)

def build_context_from_history(history: list) -> str:
    if not history:
        return ""
    recent = history[-4:]
    return " ".join([turn.get("content", "") for turn in recent])

def clean_reply(reply: str) -> str:
    leak_phrases = [
        "based on the context","according to the context",
        "the provided text","the context provided",
        "in the information given","the database",
        "the knowledge base","provided context",
        "based on the information provided",
        "the information provided","the diocesan information",
    ]
    reply_lower = reply.lower()
    for phrase in leak_phrases:
        if phrase in reply_lower:
            idx = reply_lower.find(phrase)
            reply = reply[:idx] + reply[idx + len(phrase):]
            reply_lower = reply.lower()
    reply = re.sub(r'\*\*?(.*?)\*\*?', r'\1', reply)
    return reply.strip().lstrip(",. ")

def retrieve_context(message: str, history: list) -> str:
    history_context = build_context_from_history(history)
    enriched_query  = message + " " + history_context
    semantic_docs   = retriever.invoke(enriched_query)
    keyword_docs    = keyword_boost(enriched_query)
    seen = set()
    all_docs = []
    for doc in (semantic_docs + keyword_docs):
        key = doc.page_content[:100]
        if key not in seen:
            seen.add(key)
            all_docs.append(doc)
    all_docs.sort(key=lambda d: score_chunk(d.page_content, enriched_query), reverse=True)
    top_docs = all_docs[:5]
    return "\n\n---\n\n".join([d.page_content for d in top_docs])

# ── Web chat endpoint (streaming) ──
@app.post("/chat")
async def chat(request: ChatRequest):
    context = retrieve_context(request.message, request.history)

    print(f"\n{'='*50}")
    print(f"USER: '{request.message}'")
    print(f"{'='*50}\n")

    messages = [{"role": "system", "content": SYSTEM_PROMPT.format(context=context)}]
    for turn in request.history[-6:]:
        messages.append(turn)
    messages.append({"role": "user", "content": request.message})

    def stream_response():
        full_reply = ""
        try:
            stream = client.chat.completions.create(
                model="gemma2-9b-it",
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
            yield json.dumps({"token": " Sorry, something went wrong. Please try again."}) + "\n"

        cleaned = clean_reply(full_reply)
        yield json.dumps({"done": True, "full": cleaned}) + "\n"

    return StreamingResponse(stream_response(), media_type="application/x-ndjson")

# ── Messenger webhook ──
import requests as req_lib

PAGE_ACCESS_TOKEN = os.environ.get("PAGE_ACCESS_TOKEN", "")
VERIFY_TOKEN      = os.environ.get("VERIFY_TOKEN", "")

def send_messenger_reply(recipient_id: str, text: str):
    url = f"https://graph.facebook.com/v19.0/me/messages?access_token={PAGE_ACCESS_TOKEN}"
    req_lib.post(url, json={
        "recipient": {"id": recipient_id},
        "message":   {"text": text},
        "messaging_type": "RESPONSE"
    })

def send_typing(recipient_id: str):
    url = f"https://graph.facebook.com/v19.0/me/messages?access_token={PAGE_ACCESS_TOKEN}"
    req_lib.post(url, json={
        "recipient":     {"id": recipient_id},
        "sender_action": "typing_on"
    })

@app.get("/webhook")
async def verify_webhook(
    hub_mode: str = None,
    hub_verify_token: str = None,
    hub_challenge: str = None
):
    if hub_mode == "subscribe" and hub_verify_token == VERIFY_TOKEN:
        return int(hub_challenge)
    return {"error": "Verification failed"}

@app.post("/webhook")
async def receive_message(request_obj):
    from fastapi import Request
    body = await request_obj.json()
    for entry in body.get("entry", []):
        for event in entry.get("messaging", []):
            sender_id = event["sender"]["id"]
            if "message" in event and "text" in event["message"]:
                user_text = event["message"]["text"]
                send_typing(sender_id)
                history = messenger_history.get(sender_id, [])
                context = retrieve_context(user_text, history)
                messages = [{"role": "system", "content": SYSTEM_PROMPT.format(context=context)}]
                for turn in history[-6:]:
                    messages.append(turn)
                messages.append({"role": "user", "content": user_text})
                try:
                    response = client.chat.completions.create(
                        model="gemma2-9b-it",
                        messages=messages,
                        max_tokens=350,
                        temperature=0.3,
                    )
                    reply = clean_reply(response.choices[0].message.content)
                except Exception:
                    reply = "Sorry, something went wrong. Please try again later."
                history.append({"role": "user",      "content": user_text})
                history.append({"role": "assistant",  "content": reply})
                messenger_history[sender_id] = history[-12:]
                send_messenger_reply(sender_id, reply)
    return {"status": "ok"}

@app.get("/health")
def health():
    return {"status": "ok", "model": "gemma2-9b-it via groq"}
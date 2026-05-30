from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import ollama

app = FastAPI(title="RCDoK Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading vector database...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
print("Ready.")

SYSTEM_PROMPT = """You are the official AI assistant of the Roman Catholic Diocese of Kalookan (RCDoK), Philippines.
You are warm, pastoral, and helpful — like a knowledgeable parish secretary or diocesan staff member.

Answer questions using ONLY the context provided below. If the answer isn't in the context, say so honestly
and suggest the user visit https://dioceseofkalookan.github.io/rcdok-website-new/ or contact the diocese
via Facebook at https://www.facebook.com/RomanCatholicBishopofKalookan

For broader Catholic theology or doctrine questions not specific to the diocese, briefly answer what you can
and suggest they visit https://www.magisterium.com/ for deeper resources.

You may respond in Filipino/Tagalog if the user writes in Filipino.

RESPONSE STYLE:
- Keep answers short and direct. Match the length to the question — simple questions get 1-3 sentences.
- Only give detailed bullet points if the user is clearly asking for a list or full details.
- Do NOT add closing lines like "May God bless you!" or "Do you have any other questions?" after every message — only use them occasionally when it feels natural.
- Do NOT repeat contact info, website links, or Facebook links unless the user specifically asks for them.

CONTEXT FROM THE DIOCESE WEBSITE:
{context}
"""

class ChatRequest(BaseModel):
    message: str
    history: list = []

@app.post("/chat")
async def chat(request: ChatRequest):
    docs = retriever.invoke(request.message)
    context = "\n\n---\n\n".join([doc.page_content for doc in docs])
    
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT.format(context=context)
        }
    ]
    
    for turn in request.history[-6:]:
        messages.append(turn)
    
    messages.append({
        "role": "user",
        "content": request.message
    })
    
    response = ollama.chat(
        model="gemma4:e4b",
        messages=messages,
        options={
            "num_predict": 250,   # max reply length
            "temperature": 0.3,   # more focused, less rambling
            "num_gpu": 99,        # use all available GPU layers
        }
    )
    
    reply = response["message"]["content"]
    return {"reply": reply}

@app.get("/health")
def health():
    return {"status": "ok", "model": "gemma4:e4b"}
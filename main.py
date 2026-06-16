from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import re
import json
import os

print("==> imports done")

from groq import Groq
print("==> groq imported")

from langchain_chroma import Chroma
print("==> chroma imported")

from langchain_huggingface import HuggingFaceEmbeddings
print("==> huggingface imported")

import requests as req_lib
print("==> requests imported")

app = FastAPI(title="RCDoK Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
print("==> groq client created")

print("==> loading embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
print("==> embeddings loaded")

print("==> loading chroma...")
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)
print("==> chroma loaded")

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
print("==> Ready.")
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import shutil, os

# Always wipe old DB — stale/mismatched chunks cause wrong retrieval
if os.path.exists("./chroma_db"):
    shutil.rmtree("./chroma_db")
    print("Cleared old chroma_db")

print("Loading documents from knowledge_base/...")
loader = DirectoryLoader(
    "knowledge_base/",
    glob="**/*.txt",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"}
)
documents = loader.load()
print(f"Loaded {len(documents)} documents")

print("Splitting into chunks...")
splitter = RecursiveCharacterTextSplitter(
    # 600 chars = each chunk covers roughly one topic section
    # (one parish's schedule, one priest's info, one school's details)
    # This is the sweet spot for Gemma 2b — specific enough to retrieve
    # exactly what's needed, small enough to not confuse the model.
    chunk_size=600,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " "]
)
chunks = splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks")

print("Loading embedding model (first run downloads ~90MB)...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Building vector database...")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print(f"\nDone! {len(chunks)} chunks indexed from {len(documents)} files.")
print("Saved to chroma_db/ — re-run only when knowledge_base files change.")

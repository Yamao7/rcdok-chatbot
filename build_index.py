from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

print("Loading documents from knowledge_base/...")
loader = DirectoryLoader(
    "knowledge_base/",
    glob="**/*.txt",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"}
)
documents = loader.load()
print(f"Loaded {len(documents)} documents")

# Split into chunks (300-500 words each)
print("Splitting into chunks...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,  # overlap helps context not get cut off
    separators=["\n\n", "\n", ". ", " "]
)
chunks = splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks")

# Create embeddings (this downloads ~90MB model first time)
print("Loading embedding model (first run downloads ~90MB)...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Store in ChromaDB (saved locally in ./chroma_db folder)
print("Building vector database...")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print("Vector database built and saved to chroma_db/")
print("You only need to run this once (or when the website updates).")
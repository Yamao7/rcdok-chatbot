import os, json, glob
from huggingface_hub import InferenceClient

HF_TOKEN = os.environ.get("HF_TOKEN")
MODEL    = "sentence-transformers/all-MiniLM-L6-v2"

client = InferenceClient(provider="hf-inference", api_key=HF_TOKEN)

KB_DIR      = "cleaned_knowledge_base"
OUTPUT_FILE = "embeddings_cache.json"

embeddings_cache = {}

for path in sorted(glob.glob(os.path.join(KB_DIR, "*.txt"))):
    name = os.path.splitext(os.path.basename(path))[0]
    with open(path, encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        continue

    print(f"Embedding: {name}")
    try:
        vector = client.feature_extraction(text[:2000], model=MODEL)
        embeddings_cache[name] = vector.tolist() if hasattr(vector, "tolist") else vector
    except Exception as e:
        print(f"  failed: {e}")

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(embeddings_cache, f)

print(f"\nSaved {len(embeddings_cache)} embeddings to {OUTPUT_FILE}")
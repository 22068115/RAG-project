import json
import ollama
import chromadb

# Load chunked handbook data
with open("wsu_data_science_handbook_chunks.json", "r", encoding="utf-8") as f:
    handbook_chunks = json.load(f)

# Extract only the text field from each chunk
documents = [chunk["text"] for chunk in handbook_chunks]

# Create ChromaDB client and collection
client = chromadb.Client()
collection = client.create_collection(name="wsu_docs")

# Store each handbook chunk in the vector database
for i, d in enumerate(documents):
    response = ollama.embed(
        model="mxbai-embed-large",
        input=d
    )
    embedding = response["embeddings"][0]

    collection.add(
        ids=[str(i)],
        embeddings=[embedding],
        documents=[d]
    )

# User query
query = "What are the core subjects in the Bachelor of Data Science?"

# Convert query into embedding
response = ollama.embed(
    model="mxbai-embed-large",
    input=query
)

# Retrieve the most relevant chunk
results = collection.query(
    query_embeddings=[response["embeddings"][0]],
    n_results=3
)

# Combine top-k retrieved chunks
data = " ".join(results["documents"][0])

# Generate answer
output = ollama.generate(
    model="gemma3:1b",
    prompt=f"Using this data: {data}. Respond to this prompt: {query}"
)

print(output["response"])
import ollama
from data import cos_collection

question = "	Which subjects are offered in Autumn and Spring?"

response = ollama.embed(
    model="nomic-embed-text-v2-moe",
    input=question
)

results = cos_collection.query(
    query_embeddings=[response["embeddings"][0]],
    n_results=3
)

print("Closest documents found:")

for i in range(len(results["metadatas"][0])):
    similarity = 1 - results["distances"][0][i]

    print("File:", results["metadatas"][0][i]["file_name"])
    print("Cosine similarity:", similarity)
    print()

data = results["documents"][0][0]

output = ollama.generate(
    model="qwen3:4b",
    prompt=f"""Answer the question using only the provided documentation.
If the answer is not in the documentation, say: I don't have the answer for your question.

Using this data:
{data}

Respond to this prompt:
{question}"""
)

print(output["response"])
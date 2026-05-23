import ollama
import csv
from dataequalchunking import cos_collection


question = "		Do I need to give a presentation in Project Management?      "

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

data = "\n\n".join(results["documents"][0])

csv_path = r"C:\Users\lon09\New folder\Indirectq2.csv"

old_rows = []

with open(csv_path, "r", newline="", encoding="utf-8") as csvfile:
    csvreader = csv.reader(csvfile)

    for row in csvreader:
        old_rows.append(row)


result = {}
for i in range(10):
    output = ollama.generate(
        model="qwen3:4b",
        prompt=f"""Answer the question using only the provided documentation.
If the answer is not in the documentation, say: I don't have the answer.

Using this data:
{data}

Respond to this prompt:
{question}"""
    )

    result[f"Run {i+1}"] = output["response"]

 

with open(r"C:\Users\lon09\New folder\Indirectq13.csv", "w", newline="", encoding="utf-8") as csvfile:
    csvwriter = csv.writer(csvfile)

    for row in result.items():
        csvwriter.writerow(row)

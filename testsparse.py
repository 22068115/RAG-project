import ollama
import chromadb
from rank_bm25 import BM25Okapi

file_names = [
    r"C:\Users\lon09\New folder\3769 - Bachelor of Data Science.txt",
    r"C:\Users\lon09\New folder\COMP1005 - Programming Fundamentals.txt",
    r"C:\Users\lon09\New folder\COMP1013 - Analytics Programming.txt",
    r"C:\Users\lon09\New folder\COMP1014 - Thinking About Data.txt",
    r"C:\Users\lon09\New folder\COMP2023 - Mathematical Programming.txt",
    r"C:\Users\lon09\New folder\COMP2025 - Introduction to Data Science.txt",
    r"C:\Users\lon09\New folder\COMP2026 - Visual Analytics.txt",
    r"C:\Users\lon09\New folder\COMP3002 - Applications of Big Data.txt",
    r"C:\Users\lon09\New folder\COMP3020 - Social Web Analytics.txt",
    r"C:\Users\lon09\New folder\COMP3032 - Machine Learning.txt",
    r"C:\Users\lon09\New folder\COMP3035 - Discovery Project.txt",
    r"C:\Users\lon09\New folder\INFO3019 - Project Management.txt",
    r"C:\Users\lon09\New folder\INFS2001 - Database Design and Development.txt",
    r"C:\Users\lon09\New folder\MATH1006 - Discrete Mathematics.txt",
    r"C:\Users\lon09\New folder\MATH1014 - Mathematics 1A.txt",
    r"C:\Users\lon09\New folder\MATH3011 - Probabilistic Models and Inference.txt",
    r"C:\Users\lon09\New folder\NATS1019 - Scientific Literacy.txt"
]

documents = []
for name in file_names:
    with open(name, "r", encoding="utf-8") as f:
        text = f.read()
        documents.append(text[:9000])

client = chromadb.Client()
collection = client.create_collection(name="docs")

for i, d in enumerate(documents):
    collection.add(
        ids=[str(i)],
        documents=[d],
        metadatas=[{"file_name": file_names[i]}]
    )

tokenized_documents = []
for d in documents:
    words = d.lower().split()
    tokenized_documents.append(words)

bm25 = BM25Okapi(tokenized_documents)

input = "#What are the core subjects in the Bachelor of Data Science?"

tokenized_query = input.lower().split()

scores = bm25.get_scores(tokenized_query)

best_index = scores.argmax()

print("Best document:", documents[best_index])
print("Score:", scores[best_index])

data = documents[best_index]

output = ollama.generate(
    model="qwen3:4b",
    prompt=f"""Answer the question using only the provided documentation.
If the answer is not in the documentation, say: I don't have the answer for your question.
Using this data: {data}
Respond to this prompt: {input}"""
)

print(output["response"])
import ollama
import chromadb

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

cos_collection = client.get_or_create_collection(
    name="docs",
    metadata={"hnsw:space": "cosine"}
)

    
for i, d in enumerate(documents):
    response = ollama.embed(model="nomic-embed-text-v2-moe", input=d)
    embeddings = response["embeddings"][0]
    cos_collection.add(
        ids=[str(i)],
        embeddings=[embeddings],
        documents=[d],
        metadatas=[{"file_name": file_names[i]}]
    )



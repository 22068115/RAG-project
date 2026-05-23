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
    r"C:\Users\lon09\New folder\MATH1006 - Discrete Mathematics.txt",
    r"C:\Users\lon09\New folder\NATS1019 - Scientific Literacy.txt",
]


def manual_chunking(text):
    chunks = []

    overview_text = text.find("Overview")
    requisites_text = text.find("Requisites")
    assessments_text = text.find("Assessments")

    if overview_text != -1 and requisites_text != -1:
        chunk1 = text[overview_text:requisites_text]
        chunks.append(chunk1)

    if requisites_text != -1 and assessments_text != -1:
        chunk2 = text[requisites_text:assessments_text]
        chunks.append(chunk2)

    if assessments_text != -1:
        chunk3 = text[assessments_text:]
        chunks.append(chunk3)

    if len(chunks) == 0:
        chunks.append(text)

    return chunks


documents = []
metadatas = []
ids = []

for file_index, file_name in enumerate(file_names):

    with open(file_name, "r", encoding="utf-8") as file:
        text = file.read()

    chunks = manual_chunking(text)

    for chunk_index, chunk in enumerate(chunks):
        documents.append(chunk)
        metadatas.append({"file_name": file_name})
        ids.append(str(file_index) + "_" + str(chunk_index))


client = chromadb.Client()

cos_collection = client.get_or_create_collection(
    name="docs_v1",
    metadata={"hnsw:space": "cosine"}
)


for i, d in enumerate(documents):

    response = ollama.embed(
        model="nomic-embed-text-v2-moe",
        input=d
    )

    cos_collection.upsert(
        ids=[ids[i]],
        embeddings=[response["embeddings"][0]],
        documents=[d],
        metadatas=[metadatas[i]]
    )
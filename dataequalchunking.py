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
    r"C:\Users\lon09\New folder\MATH1014 - Mathematics 1A.txt",
    r"C:\Users\lon09\New folder\MATH3011 - Probabilistic Models and Inference.txt",
    r"C:\Users\lon09\New folder\INFO3019 - Project Management.txt",
    r"C:\Users\lon09\New folder\INFS2001 - Database Design and Development.txt",
]


documents = []
metadatas = []
ids = []


def split_equal(text):
    words = text.split()
    chunks = []

    total_words = len(words)
    chunk_size = total_words // 3

    chunk1 = " ".join(words[0:chunk_size])
    chunk2 = " ".join(words[chunk_size:chunk_size * 2])
    chunk3 = " ".join(words[chunk_size * 2:])

    chunks.append(chunk1)
    chunks.append(chunk2)
    chunks.append(chunk3)

    return chunks


for file_index, file_name in enumerate(file_names):

    with open(file_name, "r", encoding="utf-8") as file:
        text = file.read()

    chunks = split_equal(text)

    print("File:", file_name)
    print("Original word count:", len(text.split()))
    print("Chunk 1 word count:", len(chunks[0].split()))
    print("Chunk 2 word count:", len(chunks[1].split()))
    print("Chunk 3 word count:", len(chunks[2].split()))
    print(
        "Total chunk words:",
        len(chunks[0].split()) + len(chunks[1].split()) + len(chunks[2].split())
    )
  

    for chunk_index, chunk in enumerate(chunks):
        documents.append(chunk)
        metadatas.append({"file_name": file_name})
        ids.append(str(file_index) + "_" + str(chunk_index))


client = chromadb.Client()

cos_collection = client.get_or_create_collection(
    name="docs_v2",
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
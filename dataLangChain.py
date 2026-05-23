import ollama
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter


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


def langchain_no_overlap(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=0
    )

    chunks = splitter.split_text(text)

    return chunks


for file_index, file_name in enumerate(file_names):

    with open(file_name, "r", encoding="utf-8") as file:
        text = file.read()

    chunks = langchain_no_overlap(text)

    print("File:", file_name)
    print("Original character count:", len(text))
    print("Number of chunks:", len(chunks))

    for chunk_index, chunk in enumerate(chunks):
        print("Chunk:", chunk_index)
        print("Chunk character count:", len(chunk))
        print(chunk)
     

        documents.append(chunk)
        metadatas.append({"file_name": file_name})
        ids.append(str(file_index) + "_" + str(chunk_index))

    


client = chromadb.Client()

cos_collection = client.get_or_create_collection(
    name="docs_v4",
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
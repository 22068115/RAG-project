import ollama
import chromadb
import os

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

doc = []
doc_ids = []

for file_name in file_names:
    with open(file_name, "r", encoding="utf-8") as file:
        text = file.read()
        if text != "":
            doc.append(text)
            doc_ids.append(os.path.basename(file_name).replace(".txt", ""))

database = chromadb.Client()
collection = database.create_collection(name="docs")

# Step 1: Embed and store
for i, d in enumerate(doc):
    try:
        response = ollama.embed(
            model="mxbai-embed-large",
            input=d
        )

        embeddings = response["embeddings"][0]

        collection.add(
            ids=[str(i)],
            embeddings=[embeddings],
            documents=[d]
        )

        print("Stored:", doc_ids[i])

    except Exception as e:
        print("Error on:", doc_ids[i])
        print(e)

# Step 2: Retrieve
question = "What are the core subjects in the Bachelor of Data Science?"

response = ollama.embed(
    model="mxbai-embed-large",
    input=question
)

results = collection.query(
    query_embeddings=[response["embeddings"][0]],
    n_results=3
)

retrieved_docs = results["documents"][0]

data = ""
for d in retrieved_docs:
    data = data + d + " "

# Step 3: Generate
output = ollama.generate(
    model="gemma3:1b",
    prompt=f"Using this data: {data}. Respond to this prompt: {question}"
)

print(output["response"])
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
for file_name in file_names:
    with open(file_name, "r", encoding="utf-8") as file:
        text = file.read()
        documents.append(text[:9000])#make sure this number covers all text in the doc. you should write another code to count the length of text, find the max, and put that value in.

client = chromadb.Client()
collection = client.create_collection(name="docs")

for i, d in enumerate(documents):
    response = ollama.embed(model="nomic-embed-text-v2-moe", input=d)
    embeddings = response["embeddings"][0]
    collection.add(
        ids=[str(i)],
        embeddings=[embeddings],
        documents=[d],
        metadatas=[{"file_name": file_names[i]}]
    )
# make sure which embedding you are using here. Find out which embedding is the best and stick with it. Maybe Nomic Embed (embedding) and FAISS for indexing

# Retrieve
input = "What are the core subjects in the Bachelor of Data Science?"

response = ollama.embed(
    model="nomic-embed-text-v2-moe",
    input=input
)

results = collection.query(
    query_embeddings=[response["embeddings"][0]],
    n_results=3
)
#try to find out printing the similarity score and the doc number that it finds the closest
print("Closest documents found:")

for i in range(len(results["metadatas"][0])):
    print("File name:", results["metadatas"][0][i]["file_name"])
    print("Similarity score:", results["distances"][0][i])
    print()

data = results["documents"][0][0] 

#  Generate
#add a fixed prompt, saying only use the dcumentation if answer is not there say I don't know"

output = ollama.generate(
    model=" qwen3:4b",
    prompt=f"""Answer the question using only the provided documentation.
If the answer is not in the documentation, say: I don't have the answer for your question.
Using this data: {data}
Respond to this prompt: {input}"""
)

print(output["response"])
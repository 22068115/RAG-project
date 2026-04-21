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

input = "Who is the coordinator of COMP2025 Introduction to Data Science?"

response = ollama.embed(
    model="nomic-embed-text-v2-moe",
    input=input
)

results = collection.query(
    query_embeddings=[response["embeddings"][0]],
    n_results=3
)# find how this ends up with 3 docs, how it identifies whch 3. Which similarity it uses, and print their values.

print("Closest documents found:")

for i in range(len(results["metadatas"][0])):
    print("File:", results["metadatas"][0][i]["file_name"])
    print("Similarity score:", 1 - results["distances"][0][i])
    print()
#make sure you use cosine similarty , then you need a high score
# if there us built in function for cosine distance, then cos sim is 1-cosdis. But I am sure there is cosine similarity availabile
data = results["documents"][0][0]

output = ollama.generate(
    model="qwen3:4b",
    prompt=f"""Answer the question using only the provided documentation.
If the answer is not in the documentation, say: I don't have the answer for your question.
Using this data: {data}
Respond to this prompt: {input}"""
)

print(output["response"])

#What are the core subjects in the Bachelor of Data Science?
#What is the description of COMP1005 Programming Fundamentals?
#How many credit points is MATH1014 Mathematics 1A?
#What level is INFS2001 Database Design and Development?
#Who is the coordinator of COMP2025 Introduction to Data Science?
#What is the assumed knowledge for MATH1006 Discrete Mathematics?
#In which semester is COMP3032 Machine Learning offered?
#What teaching periods does NATS1019 Scientific Literacy operate in?
#Is COMP3002 Applications of Big Data available in Autumn or Spring?
#When can students take COMP2026 Visual Analytics?
#Which subjects are offered in Spring semester?
#What are the assessment tasks for NATS1019 Scientific Literacy?
#What percentage is the final exam in COMP3032 Machine Learning?
#Which subjects require programming knowledge?
#Which subjects are Level 1 in the degree?
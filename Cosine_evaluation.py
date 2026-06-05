import csv
import ollama
from sklearn.metrics.pairwise import cosine_similarity

folder_path = r"C:\Users\lon09\New folder"
file_name = "Indirectq2.csv"
embed_model = "nomic-embed-text-v2-moe"


def get_similarity(rag_answer, ground_truth):

    response1 = ollama.embed(
        model=embed_model,
        input=rag_answer
    )

    response2 = ollama.embed(
        model=embed_model,
        input=ground_truth
    )

    embedding1 = response1["embeddings"][0]
    embedding2 = response2["embeddings"][0]

    similarity = cosine_similarity(
        [embedding1],
        [embedding2]
    )[0][0]

    return similarity

input_path = folder_path + "\\" + file_name
output_name = file_name.replace(".csv", "_cosine_evaluation.csv")
output_path = folder_path + "\\" + output_name

rows = []

with open(input_path, "r", newline="", encoding="utf-8") as csvfile:
    csvreader = csv.reader(csvfile)

    for row in csvreader:
        rows.append(row)


evaluation_rows = []

evaluation_rows.append([
    "Run time",
    "Non chunking cosine",
    "Equal chunking cosine",
    "Manual chunking overlap cosine",
    "LangChain cosine",
    "LangChain overlap cosine"
])


for row in rows[1:]:

    run_time = row[0]
    ground_truth = row[6]

    score1 = get_similarity(row[1], ground_truth)
    score2 = get_similarity(row[2], ground_truth)
    score3 = get_similarity(row[3], ground_truth)
    score4 = get_similarity(row[4], ground_truth)
    score5 = get_similarity(row[5], ground_truth)

    evaluation_rows.append([
        run_time,
        score1,
        score2,
        score3,
        score4,
        score5
    ])


with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
    csvwriter = csv.writer(csvfile)

    for row in evaluation_rows:
        csvwriter.writerow(row)


print("Created:", output_name)
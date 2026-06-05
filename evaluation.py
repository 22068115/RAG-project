import csv
import ollama

folder_path = r"C:\Users\lon09\New folder"
file_name = "Indirectq4.csv"
llm_model = "gemma3:1b"

def llm_evaluate(rag_answer, ground_truth):

    prompt = f"""
You are evaluating a RAG generated answer against a ground truth answer.

Ground truth answer:
{ground_truth}

RAG generated answer:
{rag_answer}

Evaluate the answer based on meaning and important information, not writing style.

Scoring rules:
1 = correct. The generated answer gives the same main meaning as the ground truth and includes the important information.
0.5 = partially correct. The generated answer has some correct information, but misses important details, is incomplete, or includes minor mistakes.
0 = incorrect. The generated answer is wrong, unrelated, too vague, or does not answer the question.

Do not give 1 just because the answer sounds confident or academic.
Do not judge based on opening words like "Based on" or "Upon".
Focus only on whether the answer matches the ground truth.

Only output one number: 1, 0.5, or 0.
"""

    response = ollama.generate(
        model=llm_model,
        prompt=prompt
    )

    score_text = response["response"].strip()

    if "0.5" in score_text:
        return 0.5
    elif "1" in score_text:
        return 1
    else:
        return 0


input_path = folder_path + "\\" + file_name
output_name = file_name.replace(".csv", "_evaluation_llm.csv")
output_path = folder_path + "\\" + output_name

rows = []

with open(input_path, "r", newline="", encoding="utf-8") as csvfile:
    csvreader = csv.reader(csvfile)

    for row in csvreader:
        rows.append(row)


evaluation_rows = []

evaluation_rows.append([
    "Run time",
    "Non chunking score",
    "Equal chunking score",
    "Manual chunking overlap score",
    "LangChain score",
    "LangChain overlap score"
])


for row in rows[1:]:

    run_time = row[0]
    ground_truth = row[6]

    score1 = llm_evaluate(row[1], ground_truth)
    score2 = llm_evaluate(row[2], ground_truth)
    score3 = llm_evaluate(row[3], ground_truth)
    score4 = llm_evaluate(row[4], ground_truth)
    score5 = llm_evaluate(row[5], ground_truth)

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
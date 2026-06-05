import csv
import os

folder_path = r"C:\Users\lon09\New folder"

direct_files = [
    "Directq1_evaluation_llm.csv",
    "Directq2_evaluation_llm.csv",
    "Directq3_evaluation_llm.csv",
    "Directq4_evaluation_llm.csv",
    "Directq5_evaluation_llm.csv",
    "Directq6_evaluation_llm.csv"
]

indirect_files = [
    "Indirectq1_evaluation_llm.csv",
    "Indirectq2_evaluation_llm.csv",
    "Indirectq3_evaluation_llm.csv",
    "Indirectq4_evaluation_llm.csv",
    "Indirectq5_evaluation_llm.csv",
    "Indirectq6_evaluation_llm.csv"
]

columns = [
    "Non chunking score",
    "Equal chunking score",
    "Manual chunking overlap score",
    "LangChain score",
    "LangChain overlap score"
]

summary_rows = []

for question_type, file_list in [
    ("Direct", direct_files),
    ("Indirect", indirect_files)
]:

    row = {}
    row["Question Type"] = question_type

    for column in columns:

        total_ones = 0
        total_scores = 0

        for file_name in file_list:

            file_path = os.path.join(folder_path, file_name)

            with open(file_path, "r", encoding="utf-8-sig") as file:
                reader = csv.DictReader(file)

                run_count = 0

                for line in reader:

                    # Only count Run 1 to Run 10
                    if line["Run time"].startswith("Run"):

                        score = line[column].strip()

                        if score == "1" or score == "1.0":
                            total_ones = total_ones + 1

                        total_scores = total_scores + 1
                        run_count = run_count + 1

                        # Stop after 10 runs for each question file
                        if run_count == 10:
                            break

        percentage = total_ones / 60 * 100

        row[column] = (
            str(total_ones)
            + "/60 = "
            + str(round(percentage, 2))
            + "%"
        )

    summary_rows.append(row)

output_file = os.path.join(folder_path, "llm_1_score_summary_table.csv")

with open(output_file, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "Question Type",
            "Non chunking score",
            "Equal chunking score",
            "Manual chunking overlap score",
            "LangChain score",
            "LangChain overlap score"
        ]
    )

    writer.writeheader()
    writer.writerows(summary_rows)

for row in summary_rows:
    print(row)

print("Saved to:", output_file)
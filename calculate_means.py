import csv

FILE_PATH = "Indirectq1_cosine_evaluation.csv"

with open(FILE_PATH, newline="") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

# Get all version columns (everything except 'Run time')
versions = [col for col in rows[0].keys() if col != "Run time"]

# Calculate mean for each version
mean_row = {"Run time": "Mean"}
for version in versions:
    values = [float(row[version]) for row in rows]
    mean_row[version] = round(sum(values) / len(values), 6)

# Append mean row to the CSV file
with open(FILE_PATH, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["Run time"] + versions)
    writer.writerow(mean_row)

print("Mean row successfully added to the bottom of the CSV file!")
print(mean_row)

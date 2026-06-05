import csv

FILE_PATH = "Directq1_cosine_evaluation.csv"

with open(FILE_PATH, newline="") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

# Get all version columns (everything except 'Run time')
versions = [col for col in rows[0].keys() if col != "Run time"]

print(f"{'Version':<35} {'Mean Cosine Score':>18}")
print("-" * 55)

for version in versions:
    values = [float(row[version]) for row in rows]
    mean = sum(values) / len(values)
    print(f"{version:<35} {mean:>18.6f}")
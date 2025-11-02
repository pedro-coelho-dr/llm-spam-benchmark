import json

input_path = "data/batches/batch_input_04.jsonl"

with open(input_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

mid = len(lines) // 2

with open("batch_input_04_a.jsonl", "w", encoding="utf-8") as f:
    f.writelines(lines[:mid])

with open("batch_input_04_b.jsonl", "w", encoding="utf-8") as f:
    f.writelines(lines[mid:])

import json
import pandas as pd
from pathlib import Path

# Paths
data_path = Path("data/enron_spam_dataset/enron_dataset.csv")
results_dir = Path("results/enron")
output_path = results_dir / "predictions.csv"

# 1. Load ground truth
df_labels = pd.read_csv(data_path, names=["id", "label", "text"], header=0)
df_labels["id"] = df_labels["id"].astype(str)

# 2. Collect all batch output files
jsonl_files = sorted(results_dir.glob("batch_*_output.jsonl"))

predictions = []
for file_path in jsonl_files:
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            if data.get("error"):  # Skip failed requests
                continue
            custom_id = str(data["custom_id"])
            content = (
                data["response"]["body"]["choices"][0]["message"]["content"]
                .strip()
                .lower()
            )
            predictions.append({"id": custom_id, "prediction": content})

df_preds = pd.DataFrame(predictions)

# 3. Merge ground truth with predictions
df_merged = df_labels.merge(df_preds, on="id", how="inner")

# 4. Save unified CSV
output_path.parent.mkdir(parents=True, exist_ok=True)
df_merged.to_csv(output_path, index=False)

print(f"✅ Saved merged predictions to {output_path}")
print(df_merged.head())

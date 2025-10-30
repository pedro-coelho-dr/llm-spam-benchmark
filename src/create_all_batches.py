import pandas as pd
import json
from pathlib import Path
from config import DATA_FILE, MODELS, SYSTEM_PROMPT, DATA_DIR

BATCHES_DIR = DATA_DIR / "batches"
BATCHES_DIR.mkdir(parents=True, exist_ok=True)

def create_batch_for_model(model_name: str):
    """Creates a batch input JSONL file for one model."""
    df = pd.read_csv(DATA_FILE)
    output_path = BATCHES_DIR / f"batch_input_{model_name}.jsonl"

    with open(output_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            item = {
                "custom_id": str(row["id"]),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": row["text"]},
                    ],
                },
            }
            f.write(json.dumps(item) + "\n")

    print(f"Created batch input for {model_name}: {output_path}")

def create_all_batches():
    """Generates JSONL batch input files for all configured models."""
    print(f"Creating batch input files in: {BATCHES_DIR.resolve()}")
    for model in MODELS:
        create_batch_for_model(model)
    print("All batch files generated successfully.")

if __name__ == "__main__":
    create_all_batches()

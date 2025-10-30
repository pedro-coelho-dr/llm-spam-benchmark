import pandas as pd
import json
from pathlib import Path
from config import DATA_FILE, BATCH_INPUT_FILE, DEFAULT_MODEL, SYSTEM_PROMPT

def create_batch_input(model_name=DEFAULT_MODEL):
    df = pd.read_csv(DATA_FILE)

    with open(BATCH_INPUT_FILE, "w") as f:
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

    print(f"Batch input file created for {model_name}: {BATCH_INPUT_FILE}")

if __name__ == "__main__":
    create_batch_input()

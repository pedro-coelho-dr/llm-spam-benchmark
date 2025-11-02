import pandas as pd
import json
from pathlib import Path

# Config
DATA_FILE = Path("data/enron_spam_dataset/enron_dataset.csv")     # adjust path
OUTPUT_DIR = Path("data/batches/enron")              # directory to write batch files
MODEL_NAME = "gpt-5-nano"                      # your chosen model
SYSTEM_PROMPT = (
    "You are an expert email classifier. "
    "Classify each email message as exactly one of the following labels:\n"
    "- ham: normal, work-related, personal, or legitimate correspondence\n"
    "- spam: unsolicited, promotional, fraudulent, or irrelevant content\n\n"
    "Respond with only the label — 'ham' or 'spam' — without explanation or punctuation."
)
BATCH_COUNT = 4

def create_batches():
    df = pd.read_csv(DATA_FILE)
    total = len(df)
    batch_size = (total + BATCH_COUNT - 1) // BATCH_COUNT  # ceil division

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for i in range(BATCH_COUNT):
        start = i * batch_size
        end = min(start + batch_size, total)
        slice_df = df.iloc[start:end]
        out_path = OUTPUT_DIR / f"batch_input_{i+1:02d}.jsonl"

        with open(out_path, "w", encoding="utf-8") as f:
            for _, row in slice_df.iterrows():
                item = {
                    "custom_id": str(row["id"]),
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": MODEL_NAME,
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": row["text"]}
                        ],
                        # you can add other params like temperature/max_tokens if needed
                    }
                }
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        print(f"Created batch file {out_path} with {len(slice_df)} requests")

if __name__ == "__main__":
    create_batches()

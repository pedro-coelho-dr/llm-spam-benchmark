import json
import pandas as pd
from pathlib import Path

# === FILES & MODEL CONFIG ===
ERROR_FILE = Path("data/multi_model_errors_3.csv")  # mensagens erradas pelos 7 modelos
BATCHES_DIR = Path("data/batches_explain")
BATCHES_DIR.mkdir(parents=True, exist_ok=True)
MODEL_NAME = "gpt-5-mini"

# === EXPLAINABLE SYSTEM PROMPT ===
SYSTEM_PROMPT_EXPLAIN = (
    "You are a binary text classifier for SMS messages. "
    "Classify each message as exactly one of the following labels:\n"
    "- ham: legitimate, personal, or non-promotional content\n"
    "- spam: promotional, fraudulent, or unsolicited content\n\n"
    "For each message, decide the correct label ('ham' or 'spam') "
    "and explain your reasoning in this format:\n"
    "Label: <ham|spam>\n"
    "Reasoning:\n"
    "1. Linguistic cues: <words, tone, or phrases that influenced the decision>\n"
    "2. Structural cues: <formatting, punctuation, or style patterns>\n"
    "3. Contextual interpretation: <intent or purpose inferred from message>\n"
    "4. Decision summary: <short justification connecting cues to final label>"
)

# === BUILD BATCH FROM CONSISTENT ERRORS (7 models) ===
def create_batch_explain_all_models():
    if not ERROR_FILE.exists():
        raise FileNotFoundError(f"❌ File not found: {ERROR_FILE.resolve()}")

    df = pd.read_csv(ERROR_FILE)
    if "text" not in df.columns:
        raise ValueError("❌ The CSV must contain a 'text' column with messages.")

    output_path = BATCHES_DIR / f"batch_input_{MODEL_NAME}_3errors_explain.jsonl"

    with open(output_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            text = str(row["text"]).strip()
            item = {
                "custom_id": str(row["id"]),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": MODEL_NAME,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT_EXPLAIN},
                        {"role": "user", "content": text},
                    ],
                },
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ Batch file created with {len(df)} messages (all-model errors): {output_path}")

if __name__ == "__main__":
    create_batch_explain_all_models()

import pandas as pd
import json
from pathlib import Path
from math import ceil
from config import DATA_FILE, SYSTEM_PROMPT

def create_split_batches(model_name: str, parts: int = 2):
    """
    Divide o dataset em N partes menores e gera arquivos JSONL separados para batch.
    """

    data_dir = Path(__file__).resolve().parent.parent / "data" / "batches"
    data_dir.mkdir(parents=True, exist_ok=True)


    df = pd.read_csv(DATA_FILE)
    total_rows = len(df)
    chunk_size = ceil(total_rows / parts)
    print(f"Total rows: {total_rows} | Splitting into {parts} parts (~{chunk_size} each)")


    for i in range(parts):
        start, end = i * chunk_size, min((i + 1) * chunk_size, total_rows)
        df_chunk = df.iloc[start:end]
        batch_path = data_dir / f"batch_input_{model_name}_part{i+1}.jsonl"

        with open(batch_path, "w") as f:
            for _, row in df_chunk.iterrows():
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

        print(f"Created {batch_path.name} with {len(df_chunk)} records")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python src/create_split_batches.py <model_name> [num_parts]")
    else:
        model = sys.argv[1]
        parts = int(sys.argv[2]) if len(sys.argv) > 2 else 2
        create_split_batches(model, parts)

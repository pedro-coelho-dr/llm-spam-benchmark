import json
import pandas as pd
from pathlib import Path
from llm_utils import get_openai_client
from config import get_result_dir

def download_batch_output(batch_id: str):
    client = get_openai_client()
    batch = client.batches.retrieve(batch_id)

    if batch.status != "completed" or not batch.output_file_id:
        print(f"⚠️ Batch {batch_id} not ready or has no output file.")
        return None

    output_file = client.files.content(batch.output_file_id)
    lines = output_file.text.strip().splitlines()

    if not lines:
        print(f"⚠️ Empty output for {batch_id}")
        return None

    # Extract model name from the first line
    first_data = json.loads(lines[0])
    model_name = (
        first_data.get("response", {})
        .get("body", {})
        .get("model", "unknown-model")
        .split("-202")[0] 
    )

    # Save output file to results directory
    result_dir = get_result_dir(model_name)
    output_path = result_dir / f"{batch_id}_output.jsonl"
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"✅ Output for {model_name} saved to {output_path}")

    return output_path, model_name


def parse_output_to_csv(output_path: Path, model_name: str):
    predictions = []

    with open(output_path, "r") as f:
        for line in f:
            data = json.loads(line)
            custom_id = data.get("custom_id")
            try:
                label = (
                    data["response"]["body"]["choices"][0]["message"]["content"]
                    .strip()
                    .lower()
                )
            except KeyError:
                label = "error"
            predictions.append({"id": custom_id, "prediction": label})

    df = pd.DataFrame(predictions)
    csv_path = output_path.parent / "predictions.csv"
    df.to_csv(csv_path, index=False)
    print(f"📊 Parsed predictions saved to {csv_path}")


def process_batches_from_file(batch_list_path: str):
    with open(batch_list_path, "r") as f:
        batch_ids = [line.strip() for line in f if line.strip()]

    print(f"🧾 Found {len(batch_ids)} batch IDs to process.")

    for batch_id in batch_ids:
        result = download_batch_output(batch_id)
        if result:
            output_path, model_name = result
            parse_output_to_csv(output_path, model_name)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: uv run src/parse_multiple_batches.py <batch_list.txt>")
        sys.exit(1)

    batch_list_path = sys.argv[1]
    process_batches_from_file(batch_list_path)

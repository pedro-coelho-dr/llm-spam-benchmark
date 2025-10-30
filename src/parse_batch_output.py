import json
from llm_utils import get_openai_client
from config import DEFAULT_MODEL, get_result_dir

def download_batch_output(batch_id, model_name=DEFAULT_MODEL):
    client = get_openai_client()
    batch = client.batches.retrieve(batch_id)

    if batch.status != "completed" or not batch.output_file_id:
        print("Batch not ready or output not available yet.")
        return

    output_file = client.files.content(batch.output_file_id)
    result_dir = get_result_dir(model_name)
    output_path = result_dir / "batch_output.jsonl"

    with open(output_path, "w") as f:
        f.write(output_file.text)

    print(f"Output saved to {output_path}")

def parse_output_to_csv(model_name=DEFAULT_MODEL):
    import pandas as pd
    result_dir = get_result_dir(model_name)
    output_path = result_dir / "batch_output.jsonl"
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
    csv_path = result_dir / "predictions.csv"
    df.to_csv(csv_path, index=False)
    print(f"Parsed predictions saved to {csv_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python src/parse_batch_output.py <batch_id>")
    else:
        batch_id = sys.argv[1]
        download_batch_output(batch_id)
        parse_output_to_csv()

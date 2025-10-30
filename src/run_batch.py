from llm_utils import get_openai_client
from config import BATCH_INPUT_FILE, DEFAULT_MODEL, BATCH_COMPLETION_WINDOW
from pathlib import Path

def submit_batch(model_name=DEFAULT_MODEL):
    client = get_openai_client()
    batch_path = Path(BATCH_INPUT_FILE)

    if not batch_path.exists():
        raise FileNotFoundError(f"Batch input file not found: {batch_path.resolve()}")

    # === Step 1: Upload file to OpenAI ===
    with open(batch_path, "rb") as f:
        uploaded_file = client.files.create(
            file=f,
            purpose="batch"
        )
    print(f"Uploaded batch file: {uploaded_file.id}")

    # === Step 2: Submit batch job ===
    batch = client.batches.create(
        input_file_id=uploaded_file.id,
        endpoint="/v1/chat/completions",
        completion_window=BATCH_COMPLETION_WINDOW,
    )

    print(f"\nSubmitted batch for {model_name}")
    print(f"Batch ID: {batch.id}")
    print(f"Status: {batch.status}")
    print(f"Monitor progress with:\n  python src/check_batch.py {batch.id}")

if __name__ == "__main__":
    submit_batch()

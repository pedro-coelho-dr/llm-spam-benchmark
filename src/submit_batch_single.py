from llm_utils import get_openai_client
from config import BATCH_COMPLETION_WINDOW
from pathlib import Path
import sys

def submit_single_batch(model_name: str):
    client = get_openai_client()
    batch_path = Path(f"data/batches/batch_input_{model_name}.jsonl")

    if not batch_path.exists():
        raise FileNotFoundError(f"❌ Batch input not found: {batch_path.resolve()}")

    # === Step 1: Upload file ===
    with open(batch_path, "rb") as f:
        uploaded_file = client.files.create(file=f, purpose="batch")
    print(f"✅ Uploaded batch file: {uploaded_file.id}")

    # === Step 2: Submit batch ===
    batch = client.batches.create(
        input_file_id=uploaded_file.id,
        endpoint="/v1/chat/completions",
        completion_window=BATCH_COMPLETION_WINDOW,
    )

    print(f"\n🚀 Submitted batch for {model_name}")
    print(f"Batch ID: {batch.id}")
    print(f"Status: {batch.status}")
    print(f"Monitor progress with:\n  uv run src/check_batch.py {batch.id}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run src/submit_batch_single.py <model_name>")
        sys.exit(1)

    model_name = sys.argv[1]
    submit_single_batch(model_name)

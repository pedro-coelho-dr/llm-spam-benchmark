from llm_utils import get_openai_client
from config import BATCH_COMPLETION_WINDOW, DATA_DIR
from pathlib import Path
import re

BATCHES_DIR = DATA_DIR / "batches"

def extract_model_name(filename: str) -> str:
    """Extracts model name from file name like 'batch_input_gpt-4o-mini.jsonl'."""
    match = re.search(r"batch_input_(.+)\.jsonl", filename)
    return match.group(1) if match else "unknown-model"

def submit_batch_file(client, batch_file: Path):
    """Uploads and submits one batch file."""
    model_name = extract_model_name(batch_file.name)
    print(f"\n🚀 Submitting batch for model: {model_name}")

    with open(batch_file, "rb") as f:
        uploaded_file = client.files.create(file=f, purpose="batch")
    print(f"📤 Uploaded file: {uploaded_file.id}")

    batch = client.batches.create(
        input_file_id=uploaded_file.id,
        endpoint="/v1/chat/completions",
        completion_window=BATCH_COMPLETION_WINDOW,
        metadata={"description": f"spam-detection-{model_name}"}
    )

    print(f"✅ Submitted batch for {model_name}")
    print(f"   Batch ID: {batch.id}")
    print(f"   Status: {batch.status}")
    print(f"   Monitor: python src/check_batch.py {batch.id}")

def submit_all_batches():
    """Iterates over all batch files in data/batches/ and submits them."""
    client = get_openai_client()
    batch_files = sorted(BATCHES_DIR.glob("batch_input_*.jsonl"))

    if not batch_files:
        print(f"❌ No batch files found in {BATCHES_DIR.resolve()}")
        return

    print(f"📦 Found {len(batch_files)} batch files to submit.")
    for batch_file in batch_files:
        try:
            submit_batch_file(client, batch_file)
        except Exception as e:
            print(f"⚠️ Failed to submit {batch_file.name}: {e}")

if __name__ == "__main__":
    submit_all_batches()

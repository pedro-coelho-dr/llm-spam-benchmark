from llm_utils import get_openai_client
from config import BATCH_COMPLETION_WINDOW, DATA_DIR
from pathlib import Path

BATCHES_DIR = DATA_DIR / "batches" 

# Explicitly define the files you want
BATCH_FILES = [
    BATCHES_DIR / "batch_input_04_b.jsonl",
]

def submit_batch_file(client, batch_file: Path):
    """Uploads and submits one batch file."""
    model_name = "gpt-5-mini"
    print(f"\n🚀 Submitting batch for model: {model_name}")

    with open(batch_file, "rb") as f:
        uploaded_file = client.files.create(file=f, purpose="batch")
    print(f"📤 Uploaded file: {uploaded_file.id}")

    batch = client.batches.create(
        input_file_id=uploaded_file.id,
        endpoint="/v1/chat/completions",
        completion_window=BATCH_COMPLETION_WINDOW,
        metadata={"description": f"enron-email-spam-detection-{model_name}"}
    )

    print(f"✅ Submitted batch: {batch_file.name}")
    print(f"   Batch ID: {batch.id}")
    print(f"   Status: {batch.status}")
    print(f"   Monitor: python src/check_batch.py {batch.id}")

def submit_selected_batches():
    """Submits only the specified batch files."""
    client = get_openai_client()
    existing = [f for f in BATCH_FILES if f.exists()]

    if not existing:
        print("❌ No valid batch files found.")
        return

    print(f"📦 Found {len(existing)} batch files to submit.")
    for batch_file in existing:
        try:
            submit_batch_file(client, batch_file)
        except Exception as e:
            print(f"⚠️ Failed to submit {batch_file.name}: {e}")

if __name__ == "__main__":
    submit_selected_batches()

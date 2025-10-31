from pathlib import Path
from llm_utils import get_openai_client
from config import DATA_DIR, BATCH_COMPLETION_WINDOW

def submit_explain_batch():
    """Submits the explainable  batch to the OpenAI Batch API."""

    client = get_openai_client()

    model_name = "gpt-5-mini"
    batch_path = DATA_DIR / "batches_explain" / "batch_input_gpt-5-mini_3errors_explain.jsonl"

    if not batch_path.exists():
        raise FileNotFoundError(f"❌ Batch file not found: {batch_path.resolve()}")

    print(f"\n🚀 Submitting explainable batch for model: {model_name}")

    # Step 1: Upload batch input
    with open(batch_path, "rb") as f:
        uploaded_file = client.files.create(file=f, purpose="batch")

    print(f"✅ Uploaded batch file: {uploaded_file.id}")

    # Step 2: Submit batch job
    batch = client.batches.create(
        input_file_id=uploaded_file.id,
        endpoint="/v1/chat/completions",
        completion_window=BATCH_COMPLETION_WINDOW,
        metadata={"description": f"explainable-sms-spam-{model_name}"}
    )

    print(f"✅ Batch submitted successfully!")
    print(f"   Model: {model_name}")
    print(f"   Batch ID: {batch.id}")
    print(f"   Status: {batch.status}")
    print(f"   Monitor progress with:")
    print(f"   uv run src/check_batch.py {batch.id}")

if __name__ == "__main__":
    submit_explain_batch()

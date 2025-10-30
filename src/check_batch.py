import sys
from llm_utils import get_openai_client

def check_batch(batch_id):
    client = get_openai_client()
    batch = client.batches.retrieve(batch_id)
    print(f"Batch ID: {batch.id}")
    print(f"Status: {batch.status}")
    if batch.status == "completed" and batch.output_file_id:
        print(f"Output file ID: {batch.output_file_id}")
    else:
        print("No output yet.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/check_batch.py <batch_id>")
    else:
        check_batch(sys.argv[1])

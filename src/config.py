from pathlib import Path

# === Base directories ===
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

# === Dataset ===
DATA_FILE = DATA_DIR / "smsspam_shuffled.csv"
BATCH_INPUT_FILE = DATA_DIR / "batch_input.jsonl"

# === Models to test ===
MODELS = [
    "gpt-5-pro", 
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-3.5-turbo",
]

# === Default model for initial batch ===
DEFAULT_MODEL = "gpt-3.5-turbo"

# === LLM setup ===
SYSTEM_PROMPT = (
    "You are a binary text classifier for SMS messages. "
    "Classify each message as exactly one of the following labels:\n"
    "- ham: legitimate, personal, or non-promotional content\n"
    "- spam: promotional, fraudulent, or unsolicited content\n\n"
    "Respond with only the label — 'ham' or 'spam' — without explanation or punctuation."
)

# === Batch settings ===
BATCH_COMPLETION_WINDOW = "24h"

# === Utility ===
def get_result_dir(model_name: str) -> Path:
    path = RESULTS_DIR / model_name
    path.mkdir(parents=True, exist_ok=True)
    return path

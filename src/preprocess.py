"""
preprocess.py
-------------
Prepares the SMS Spam Collection v.1 dataset for modern LLM-based experiments.

Input:
    data/raw                   # tab-separated file: <label>\t<message>

Outputs:
    data/smsspam_dataset.csv        # canonical reference: id,label,text
    data/smsspam_unlabeled.csv      # shuffled unlabeled: id,text (for inference)
    data/dataset_info.md            # metadata summary

Normalization:
    - HTML unescaped (&lt; → <, &gt; → >)
    - Unicode normalized (NFKC)
    - Collapsed whitespace
"""

import pandas as pd
from sklearn.utils import shuffle
from pathlib import Path
import html
import unicodedata
import re
import sys

def preprocess():
    base = Path("data")
    raw_path = base / "raw"

    if not raw_path.exists():
        sys.exit(f"File not found: {raw_path.resolve()}")

    print(f"Loading dataset from: {raw_path.resolve()}")

    # --- Load the dataset safely ---
    with open(raw_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [line.strip() for line in f if line.strip()]
    data = [line.split("\t", 1) for line in lines if "\t" in line]
    df = pd.DataFrame(data, columns=["label", "text"])
    df = df.dropna().drop_duplicates()

    # --- Assign unique ID ---
    df.insert(0, "id", range(1, len(df) + 1))

    # --- Normalize text ---
    def clean_text(t: str) -> str:
        t = str(t).strip()
        t = html.unescape(t)                      # decode HTML entities
        t = unicodedata.normalize("NFKC", t)      # normalize Unicode
        t = re.sub(r"\s+", " ", t)                # collapse spaces
        return t

    df["text"] = df["text"].apply(clean_text)
    df["label"] = df["label"].str.lower().str.strip()

    # --- Save canonical dataset (id,label,text) ---
    full_path = base / "smsspam_dataset.csv"
    df.to_csv(full_path, index=False)

    # --- Create unlabeled shuffled version (id,text) ---
    df_unlabeled = shuffle(df[["id", "text"]], random_state=42)
    unlabeled_path = base / "smsspam_shuffled.csv"
    df_unlabeled.to_csv(unlabeled_path, index=False)

    # --- Compute summary ---
    ham = (df["label"] == "ham").sum()
    spam = (df["label"] == "spam").sum()
    stats = f"""# SMS Spam Collection v.1 — Metadata

**Total messages:** {len(df)}
- Ham:  {ham} ({ham/len(df)*100:.2f}%)
- Spam: {spam} ({spam/len(df)*100:.2f}%)

**Columns:** id, label, text  
**Format:** TSV → CSV  
**Preprocessing:**
- HTML unescaped (&lt; → <, &gt; → >)
- Unicode normalized (NFKC)
- Whitespace collapsed
- Dropped duplicates / NaNs
- Added stable IDs
- Created shuffled unlabeled copy for LLM inference
"""
    (base / "dataset_info.md").write_text(stats.strip())

    # --- Report ---
    print("Preprocessing complete.")
    print(f"→ Total: {len(df)} messages ({ham} ham / {spam} spam)")
    print(f"Canonical dataset: {full_path}")
    print(f"Unlabeled shuffled: {unlabeled_path}")
    print(f"Metadata: {base / 'dataset_info.md'}")

if __name__ == "__main__":
    preprocess()

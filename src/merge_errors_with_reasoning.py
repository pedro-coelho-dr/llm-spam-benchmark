#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

# === Paths ===
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# Input files
OLD_FILE = DATA_DIR / "multi_model_errors_3.csv"
NEW_FILE = DATA_DIR / "gpt-5-mini-3errors-explain" / "predictions.csv"
OUTPUT_FILE = DATA_DIR / "prediction_changes.csv"

print(f"📂 BASE_DIR: {BASE_DIR}")
print(f"📄 Old results: {OLD_FILE.exists()} → {OLD_FILE}")
print(f"📄 New predictions: {NEW_FILE.exists()} → {NEW_FILE}")

# === Load data ===
df_old = pd.read_csv(OLD_FILE)
df_new = pd.read_csv(NEW_FILE)

# === Normalize ===
df_old["prediction"] = df_old["prediction"].astype(str).str.strip().str.lower()
df_new["prediction"] = df_new["prediction"].astype(str).str.strip().str.lower()
df_new["reasoning"] = df_new.get("reasoning", "").astype(str).str.strip()

# === Merge ===
df_merged = df_old.merge(df_new, on="id", how="left", suffixes=("_old", "_new"))

# === Compare predictions ===
df_merged["prediction_changed"] = df_merged["prediction_old"] != df_merged["prediction_new"]

# === Organize ===
cols = [
    "id", "label", "text",
    "prediction_old", "prediction_new",
    "prediction_changed",
    "n_models_error", "models_with_error",
    "reasoning"
]
df_merged = df_merged[cols]

# === Save ===
df_merged.to_csv(OUTPUT_FILE, index=False)

# === Summary ===
changed = df_merged["prediction_changed"].sum()
total = len(df_merged)
print(f"✅ Arquivo salvo: {OUTPUT_FILE}")
print(f"🔁 Mudaram de opinião: {changed}/{total} ({changed/total:.1%})")
print(f"🧩 Total de linhas: {len(df_merged)}")

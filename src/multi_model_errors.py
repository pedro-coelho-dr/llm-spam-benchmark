import pandas as pd
from pathlib import Path
from config import RESULTS_DIR, DATA_DIR

# === Caminho fixo do dataset ===
DATA_FILE = DATA_DIR / "smsspam_dataset.csv"

# === Carrega ground truth com texto ===
print(f"📄 Lendo dataset: {DATA_FILE}")
df_true = pd.read_csv(DATA_FILE, usecols=["id", "label", "text"])
df_true["label"] = df_true["label"].str.strip().str.lower()
df_true["text"] = df_true["text"].astype(str).str.strip()

# === Identifica modelos válidos ===
models = [p.name for p in RESULTS_DIR.iterdir() if (p / "predictions.csv").exists()]
print(f"🧠 Modelos detectados: {models}")

# === Carrega predições ===
dfs = []
for model in models:
    pred_path = RESULTS_DIR / model / "predictions.csv"
    df_pred = pd.read_csv(pred_path, usecols=["id", "prediction"])
    df_pred.rename(columns={"prediction": model}, inplace=True)
    df_pred[model] = df_pred[model].str.strip().str.lower()
    dfs.append(df_pred)

# === Junta tudo ===
df_all = df_true.copy()
for df_pred in dfs:
    df_all = df_all.merge(df_pred, on="id", how="left")

# === Marca erros ===
for m in models:
    df_all[f"{m}_error"] = df_all[m] != df_all["label"]

# === Conta quantos modelos erraram ===
err_cols = [f"{m}_error" for m in models]
df_all["n_models_error"] = df_all[err_cols].sum(axis=1)

# === Lista quais modelos erraram ===
def models_with_error(row):
    return [m for m in models if row.get(f"{m}_error", False)]

df_all["models_with_error"] = df_all.apply(models_with_error, axis=1)

# === Pega uma predição incorreta (primeiro modelo que errou) ===
def get_wrong_prediction(row):
    for m in models:
        if row.get(f"{m}_error", False):
            return row[m]
    return None

df_all["prediction"] = df_all.apply(get_wrong_prediction, axis=1)

# === Filtra apenas mensagens com erro ===
df_errors = df_all[df_all["n_models_error"] > 0][
    ["id", "label", "text", "prediction", "n_models_error", "models_with_error"]
]

# === Exporta CSV consolidado ===
output_path = DATA_DIR / "multi_model_errors.csv"
df_errors.to_csv(output_path, index=False)

# === Resumo ===
print(f"\n✅ CSV salvo em: {output_path}")
print(f"Total de mensagens com erro: {len(df_errors)}")

print("\n📊 Distribuição de mensagens erradas por número de modelos:")
print(df_errors["n_models_error"].value_counts().sort_index(ascending=False).to_string())

import json
import pandas as pd
from pathlib import Path

# Caminho para o arquivo JSONL baixado manualmente
input_path = Path("data/batch_gpt5_fail_output.jsonl")

# Nome do modelo (ajuste conforme necessário)
model_name = "gpt-5"

# Diretório de saída
output_dir = Path("results") / model_name
output_dir.mkdir(parents=True, exist_ok=True)

# Arquivo CSV de saída
csv_path = output_dir / "predictions.csv"

predictions = []

with open(input_path, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        custom_id = data.get("custom_id")
        try:
            label = (
                data["response"]["body"]["choices"][0]["message"]["content"]
                .strip()
                .lower()
            )
        except KeyError:
            label = "error"
        predictions.append({"id": custom_id, "prediction": label})

# Converte para DataFrame e salva
df = pd.DataFrame(predictions)
df.to_csv(csv_path, index=False)

print(f"Parsed predictions saved to {csv_path}")

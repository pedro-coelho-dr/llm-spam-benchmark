import json
import re
import pandas as pd
from pathlib import Path
from llm_utils import get_openai_client
from config import DATA_DIR, RESULTS_DIR

def download_and_parse_batch(batch_id: str):
    """Baixa e converte a saída de um batch explicável em CSV."""
    client = get_openai_client()

    # === Recupera o batch ===
    batch = client.batches.retrieve(batch_id)
    if batch.status != "completed" or not batch.output_file_id:
        raise RuntimeError(f"⚠️ Batch {batch_id} ainda não finalizado ou sem arquivo de saída.")

    # === Baixa o arquivo JSONL ===
    output_file = client.files.content(batch.output_file_id)
    lines = output_file.text.strip().splitlines()
    print(f"📦 {len(lines)} respostas recebidas no batch.")

    model_name = "gpt-5-mini"
    result_dir = RESULTS_DIR / f"{model_name}-3errors-explain"
    result_dir.mkdir(parents=True, exist_ok=True)
    raw_path = result_dir / f"{batch_id}_output.jsonl"

    # === Salva a versão bruta ===
    with open(raw_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"✅ Arquivo bruto salvo em: {raw_path}")

    # === Extrai campos estruturados ===
    records = []
    for line in lines:
        data = json.loads(line)
        custom_id = data.get("custom_id")
        content = data["response"]["body"]["choices"][0]["message"]["content"]

        # Extrai o label (primeira linha)
        match_label = re.search(r"Label:\s*(ham|spam)", content, re.IGNORECASE)
        label = match_label.group(1).lower() if match_label else "error"

        # Extrai reasoning (todo o resto)
        match_reason = re.search(r"Reasoning:(.*)", content, re.IGNORECASE | re.DOTALL)
        reasoning = match_reason.group(1).strip() if match_reason else ""

        records.append({
            "id": custom_id,
            "prediction": label,
            "reasoning": reasoning
        })

    # === Salva CSV ===
    df = pd.DataFrame(records)
    csv_path = result_dir / "predictions.csv"
    df.to_csv(csv_path, index=False)
    print(f"✅ Arquivo processado salvo em: {csv_path}")
    print(df.head())

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Uso: uv run src/parse_explain_batch_output.py <batch_id>")
        sys.exit(1)

    batch_id = sys.argv[1]
    download_and_parse_batch(batch_id)

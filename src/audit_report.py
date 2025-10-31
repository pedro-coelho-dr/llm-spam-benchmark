"""
audit_raw_sms.py
----------------
Verifica inconsistências no dataset SMS Spam Collection v.1
e mostra quais linhas são removidas no preprocessamento.

Analisa:
    - Linhas sem tabulação
    - Linhas duplicadas
    - Linhas vazias
    - Linhas com campos ausentes
"""

from pathlib import Path
import pandas as pd
import html
import unicodedata
import re

def normalize_text(t: str) -> str:
    t = str(t).strip()
    t = html.unescape(t)
    t = unicodedata.normalize("NFKC", t)
    t = re.sub(r"\s+", " ", t)
    return t

def audit_raw():
    raw_path = Path("data/raw")
    if not raw_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {raw_path.resolve()}")

    with open(raw_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [line.rstrip("\n") for line in f]

    print(f"Total bruto: {len(lines)} linhas")

    # --- Vazias ---
    empty_lines = [i for i, l in enumerate(lines, 1) if not l.strip()]
    print(f"Linhas vazias: {len(empty_lines)}")

    # --- Sem tabulação ---
    no_tab = [i for i, l in enumerate(lines, 1) if "\t" not in l]
    print(f"Linhas sem tabulação: {len(no_tab)}")

    # --- Carregar válidas ---
    valid = [l.split("\t", 1) for l in lines if "\t" in l and l.strip()]
    df = pd.DataFrame(valid, columns=["label", "text"])

    # Normaliza e limpa
    df["text"] = df["text"].map(normalize_text)
    df["label"] = df["label"].str.lower().str.strip()

    # --- Linhas com valores faltando ---
    missing = df[df.isna().any(axis=1)]
    print(f"Linhas com valores ausentes: {len(missing)}")

    # --- Duplicadas ---
    duplicates = df[df.duplicated()]
    print(f"Linhas duplicadas: {len(duplicates)}")

    # --- Resumo final ---
    total_final = len(df.dropna().drop_duplicates())
    print(f"\nResumo:")
    print(f"  Original: {len(lines)}")
    print(f"  Após limpeza: {total_final}")
    print(f"  Perda total: {len(lines) - total_final}")

    # --- Exemplos de inconsistências ---
    print("\n=== Exemplos ===")
    if no_tab:
        print("\nSem tabulação (primeiras 3):")
        for i in no_tab[:3]:
            print(f"  Linha {i}: {lines[i-1]!r}")

    if not duplicates.empty:
        print("\nDuplicadas (primeiras 3):")
        print(duplicates.head(3).to_string(index=False))

    if empty_lines:
        print("\nVazias (primeiras 3 linhas):", empty_lines[:3])

    if not missing.empty:
        print("\nCom valores ausentes (primeiras 3):")
        print(missing.head(3).to_string(index=False))

    print("\n--- Fim da auditoria ---")

if __name__ == "__main__":
    audit_raw()

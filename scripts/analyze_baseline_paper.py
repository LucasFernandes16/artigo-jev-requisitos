#!/usr/bin/env python3
"""Extrai estatísticas descritivas e baselines oficiais do pacote de replicação de Roßkothen et al. (RE 2026)."""

from __future__ import annotations

import ast
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from req_jev.dataset import DEFAULT_ZENODO_DIR, load_source_dataset  # noqa: E402


def main() -> None:
    items = load_source_dataset()
    print(f"Total de requisitos fonte carregados: {len(items)}")
    total_targets = sum(len(i.target_master_requirements) for i in items)
    print(f"Total de requisitos MASTeR humanos (ground truth): {total_targets}")

    rephrased_df = pd.read_csv(DEFAULT_ZENODO_DIR / "rephrased_reqs_data.csv")
    metric_cols = [
        "CorpBERTF1",
        "CorpBERTRecall",
        "SentBERTF1",
        "EmbeddingMoversDistance",
        "SophistTemplateF1",
        "SophistConditionF1",
        "SophistConditionWithTemplateF1",
        "BLEUScore",
        "best",
    ]
    grouped = rephrased_df.groupby("run_name")[metric_cols].mean()
    print("\n=== Médias Oficiais das 4 Configurações de Roßkothen et al. (n=249 por config) ===")
    print(grouped.round(3).to_string())

    # Contagem média de requisitos gerados por configuração
    rephrased_df["gen_count"] = rephrased_df["requirements"].apply(lambda x: len(ast.literal_eval(x)))
    print("\n=== Fator médio de split gerado por configuração ===")
    print(rephrased_df.groupby("run_name")["gen_count"].mean().round(2).to_string())

    # Análise por complexidade (Split Factor 1 vs 2 vs 3+)
    rephrased_df["sf_bucket"] = rephrased_df["split_factor"].apply(lambda s: "3+" if int(s) >= 3 else str(int(s)))
    by_sf = rephrased_df.groupby(["run_name", "sf_bucket"])[["CorpBERTF1", "SophistTemplateF1", "EmbeddingMoversDistance"]].mean()
    print("\n=== Desempenho por Complexidade (Split Factor) ===")
    print(by_sf.round(3).to_string())

    out_dir = REPO_ROOT / "paper" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = REPO_ROOT / "results" / "baseline_rosskothen2026_summary.csv"
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(summary_csv)
    print(f"\nResumo oficial exportado para: {summary_csv}")


if __name__ == "__main__":
    main()

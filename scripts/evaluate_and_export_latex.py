#!/usr/bin/env python3
"""Avalia arquivos JSONL gerados em `results/` e exporta tabelas LaTeX para `paper/tables/`."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from req_jev.evaluation import (  # noqa: E402
    compute_corpus_bleu_avg,
    compute_master_template_f1,
    summarize_operational_metrics,
)
from req_jev.schemas import ExperimentRunRecord  # noqa: E402


def load_run_jsonl(path: Path) -> list[ExperimentRunRecord]:
    records: list[ExperimentRunRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(ExperimentRunRecord.model_validate_json(line))
    return records


def evaluate_records(records: list[ExperimentRunRecord]) -> dict[str, float]:
    op = summarize_operational_metrics(records)
    tmpl_main: list[float] = []
    tmpl_cond: list[float] = []
    tmpl_comb: list[float] = []
    bleu_scores: list[float] = []

    accepted_main: list[float] = []
    escalated_main: list[float] = []

    for r in records:
        f1s = compute_master_template_f1(r.ground_truth_requirements, r.generated_requirements)
        tmpl_main.append(f1s["MasterTemplateF1"])
        tmpl_cond.append(f1s["MasterConditionF1"])
        tmpl_comb.append(f1s["MasterConditionWithTemplateF1"])
        bleu_scores.append(compute_corpus_bleu_avg(r.ground_truth_requirements, r.generated_requirements))
        if r.escalated_to_human:
            escalated_main.append(f1s["MasterTemplateF1"])
        else:
            accepted_main.append(f1s["MasterTemplateF1"])

    n = max(len(records), 1)
    op.update(
        {
            "MasterTemplateF1": sum(tmpl_main) / n,
            "MasterConditionF1": sum(tmpl_cond) / n,
            "MasterConditionWithTemplateF1": sum(tmpl_comb) / n,
            "BLEUScore": sum(bleu_scores) / n,
            "AcceptedSubset_MasterTemplateF1": (
                sum(accepted_main) / len(accepted_main) if accepted_main else 0.0
            ),
            "EscalatedSubset_MasterTemplateF1": (
                sum(escalated_main) / len(escalated_main) if escalated_main else 0.0
            ),
        }
    )
    return op


def main() -> None:
    parser = argparse.ArgumentParser(description="Consolida métricas dos runs experimentais e gera tabelas LaTeX.")
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--tables-dir", type=Path, default=REPO_ROOT / "paper" / "tables")
    args = parser.parse_args()

    jsonl_files = sorted(args.results_dir.glob("*.jsonl"))
    if not jsonl_files:
        print(f"Nenhum arquivo .jsonl encontrado em {args.results_dir}. Execute scripts/run_experiment.py primeiro.")
        return

    rows: list[dict[str, object]] = []
    for path in jsonl_files:
        recs = load_run_jsonl(path)
        if not recs:
            continue
        metrics = evaluate_records(recs)
        row: dict[str, object] = {
            "run_id": recs[0].run_id,
            "provider": recs[0].provider,
            "mode": recs[0].mode,
            **metrics,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    args.tables_dir.mkdir(parents=True, exist_ok=True)
    csv_out = args.results_dir / "summary_metrics.csv"
    df.to_csv(csv_out, index=False)
    print("\n=== Resumo Consolidado dos Experimentos ===")
    print(
        df[
            [
                "run_id",
                "num_requirements",
                "MasterTemplateF1",
                "MasterConditionF1",
                "BLEUScore",
                "cost_per_1000_reqs_usd",
                "mean_validation_latency_ms",
                "escalation_rate",
            ]
        ].to_string(index=False)
    )

    tex_out = args.tables_dir / "experimental_results_auto.tex"
    lines = [
        "% Tabela gerada automaticamente por scripts/evaluate_and_export_latex.py",
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "Configuração & MASTeR Main $F_1$ & Cond. $F_1$ & BLEU & Lat. Val. (ms) & Custo/1k (\\$) & Escalon. (\\%) \\\\",
        "\\midrule",
    ]
    for _, r in df.iterrows():
        name_esc = str(r["run_id"]).replace("_", "\\_")
        lines.append(
            f"{name_esc} & {r['MasterTemplateF1']:.3f} & {r['MasterConditionF1']:.3f} & "
            f"{r['BLEUScore']:.3f} & {r['mean_validation_latency_ms']:.1f} & "
            f"{r['cost_per_1000_reqs_usd']:.2f} & {r['escalation_rate'] * 100.0:.1f}\\% \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    tex_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nCSV salvo em: {csv_out}\nTabela LaTeX salva em: {tex_out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Executor assíncrono dos experimentos com checkpoint incremental em JSONL.

Exemplos de uso:
  # Teste rápido (5 requisitos) com Gemini + JEV-Cascade:
  .venv/bin/python scripts/run_experiment.py --provider gemini --mode jev_cascade --limit 5

  # Bateria completa (249 requisitos) nos 3 modos com Gemini:
  .venv/bin/python scripts/run_experiment.py --provider gemini --mode oneshot
  .venv/bin/python scripts/run_experiment.py --provider gemini --mode llm_judge
  .venv/bin/python scripts/run_experiment.py --provider gemini --mode jev_cascade

  # Comparação 1:1 com Qwen3-235B via OpenRouter + JEV-Cascade:
  .venv/bin/python scripts/run_experiment.py --provider openrouter --mode jev_cascade
"""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
import sys

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from req_jev.dataset import SourceDatasetItem, load_source_dataset  # noqa: E402
from req_jev.jev_validator import TypeSafeJEVValidator  # noqa: E402
from req_jev.llm_backends import build_llm_backend  # noqa: E402
from req_jev.pipelines import (  # noqa: E402
    run_jev_cascade_pipeline,
    run_llm_judge_pipeline,
    run_oneshot_pipeline,
)
from req_jev.schemas import ExperimentRunRecord  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Executa formalização MASTeR com OneShot, LLM-Judge ou JEV-Cascade.")
    parser.add_argument("--provider", choices=["gemini", "openrouter"], default="gemini")
    parser.add_argument("--mode", choices=["oneshot", "llm_judge", "jev_cascade"], required=True)
    parser.add_argument("--generator-model", type=str, default=None)
    parser.add_argument("--judge-model", type=str, default=None)
    parser.add_argument("--jev-model", type=str, default=None)
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--tau-accept", type=float, default=0.75)
    parser.add_argument("--tau-reject", type=float, default=0.40)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="Limita o número de requisitos (útil para smoke test).")
    parser.add_argument("--output", type=Path, default=None, help="Arquivo JSONL de saída (suporta retomada automática).")
    return parser.parse_args()


def load_completed_ids(jsonl_path: Path) -> set[str]:
    if not jsonl_path.exists():
        return set()
    done: set[str] = set()
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = ExperimentRunRecord.model_validate_json(line)
            done.add(rec.req_name)
    return done


async def main_async() -> None:
    load_dotenv(REPO_ROOT / ".env")
    args = parse_args()

    if args.provider == "gemini":
        gen_model = args.generator_model or os.environ.get("GEMINI_GENERATOR_MODEL", "gemini-2.5-flash")
        judge_model = args.judge_model or os.environ.get("GEMINI_JUDGE_MODEL", "gemini-2.5-flash")
    else:
        gen_model = args.generator_model or os.environ.get(
            "OPENROUTER_GENERATOR_MODEL", "qwen/qwen3-235b-a22b-2507"
        )
        judge_model = args.judge_model or os.environ.get("OPENROUTER_JUDGE_MODEL", "openai/gpt-oss-120b")

    safe_model_slug = gen_model.replace("/", "_").replace(":", "_")
    run_id = f"{args.provider}_{safe_model_slug}_{args.mode}"
    out_path = args.output or (REPO_ROOT / "results" / f"{run_id}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_source_dataset()
    if args.limit is not None:
        dataset = dataset[: args.limit]

    completed = load_completed_ids(out_path)
    pending = [item for item in dataset if item.req_name not in completed]
    print(
        f"[run_experiment] run_id={run_id} | total={len(dataset)} | "
        f"já_concluídos={len(completed)} | pendentes={len(pending)}"
    )
    if not pending:
        print("[run_experiment] Todos os itens solicitados já estão gravados em", out_path)
        return

    gen_backend = build_llm_backend(args.provider)
    judge_backend = build_llm_backend(args.provider) if args.mode == "llm_judge" else None
    jev_validator = TypeSafeJEVValidator(model=args.jev_model) if args.mode == "jev_cascade" else None

    sem = asyncio.Semaphore(args.concurrency)
    write_lock = asyncio.Lock()

    async def _process_one(idx: int, item: SourceDatasetItem) -> None:
        async with sem:
            if args.mode == "oneshot":
                rec = await run_oneshot_pipeline(
                    run_id=run_id,
                    backend=gen_backend,
                    generator_model=gen_model,
                    item=item,
                )
            elif args.mode == "llm_judge":
                assert judge_backend is not None
                rec = await run_llm_judge_pipeline(
                    run_id=run_id,
                    generator_backend=gen_backend,
                    judge_backend=judge_backend,
                    generator_model=gen_model,
                    judge_model=judge_model,
                    item=item,
                    max_iterations=args.max_iterations,
                )
            else:
                assert jev_validator is not None
                rec = await run_jev_cascade_pipeline(
                    run_id=run_id,
                    generator_backend=gen_backend,
                    jev_validator=jev_validator,
                    generator_model=gen_model,
                    item=item,
                    max_iterations=args.max_iterations,
                    tau_accept=args.tau_accept,
                    tau_reject=args.tau_reject,
                )

            async with write_lock:
                with out_path.open("a", encoding="utf-8") as f:
                    f.write(rec.model_dump_json() + "\n")
                print(
                    f"  [{idx}/{len(pending)}] {item.req_name:10s} | "
                    f"iters={rec.iterations_used} | route={rec.final_cascade_route:8s} | "
                    f"lat={rec.total_latency_ms:7.1f}ms | cost=${rec.total_cost_usd:.6f}"
                )

    await asyncio.gather(*(_process_one(i, it) for i, it in enumerate(pending, start=1)))
    print(f"[run_experiment] Concluído! Resultados salvos em: {out_path}")


if __name__ == "__main__":
    asyncio.run(main_async())

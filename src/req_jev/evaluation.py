"""Métricas automáticas de qualidade (MASTeR F1, EMD, BLEU, BERTScore opcional) e testes estatísticos."""

from __future__ import annotations

from collections import Counter
import math
import re
from typing import Callable

from scipy.stats import spearmanr, wasserstein_distance_nd, wilcoxon

from req_jev.schemas import (
    ExperimentRunRecord,
    MasterCondition,
    MasterTemplatePattern,
)


def classify_master_by_rules(req: str) -> tuple[MasterTemplatePattern, MasterCondition]:
    """Classificador determinístico MASTeR baseado nos padrões sintáticos de Rupp/SOPHIST e Roßkothen et al."""
    text = req.strip()

    # 1. Sub-padrão de Condição
    if text.startswith("As soon as"):
        cond = MasterCondition.Event
    elif text.startswith("As long as"):
        cond = MasterCondition.Time
    elif text.startswith("If"):
        cond = MasterCondition.Logic
    else:
        cond = MasterCondition.No

    # 2. Sub-padrão Principal (regras de alta precisão de metrics.py + heurísticas sintáticas)
    if "provide" in text and "with the ability to" in text:
        main = MasterTemplatePattern.FunctionalUser
    elif "be able to" in text:
        main = MasterTemplatePattern.FunctionalInterface
    elif "be designed in a way" in text and "can be operated" in text:
        main = MasterTemplatePattern.Environment
    elif re.search(r"\b(shall|should|will)\s+be\b", text):
        main = MasterTemplatePattern.Property
    elif re.search(
        r"\b(supplier|contractor|manufacturer|developer|provider|organization|vendor|engineer)\s+(shall|should|will)\b",
        text,
        flags=re.IGNORECASE,
    ):
        main = MasterTemplatePattern.Process
    else:
        main = MasterTemplatePattern.FunctionalIndependent

    return main, cond


def multiset_f1(base_labels: list[str], candidate_labels: list[str]) -> float:
    """Implementação idêntica ao `MasterTemplateMetric._get_f1_score` de Roßkothen et al. (`metrics.py`)."""
    if not base_labels or not candidate_labels:
        return 0.0
    base_counter = Counter(base_labels)
    cand_counter = Counter(candidate_labels)
    overlap = sum(min(base_counter[lbl], cand_counter[lbl]) for lbl in cand_counter)
    if overlap == 0:
        return 0.0
    precision = overlap / sum(base_counter.values())
    recall = overlap / sum(cand_counter.values())
    return 2.0 * precision * recall / (precision + recall)


def compute_master_template_f1(
    ground_truth_reqs: list[str],
    candidate_reqs: list[str],
    classifier: Callable[[str], tuple[MasterTemplatePattern, MasterCondition]] = classify_master_by_rules,
) -> dict[str, float]:
    """Calcula Main Template F1, Condition F1 e Combined F1 contra o ground truth."""
    base_cls = [classifier(r) for r in ground_truth_reqs]
    cand_cls = [classifier(r) for r in candidate_reqs]

    # Nota da Tabela III do artigo: no ground truth de Großer et al., os 3 subtipos FunctionMASTeR
    # são agrupados sob um único rótulo "FunctionMASTeR" para avaliação justa.
    def _collapse_fn(pat: MasterTemplatePattern) -> str:
        if pat.value.startswith("FunctionMASTeR"):
            return "FunctionMASTeR"
        return pat.value

    base_main = [_collapse_fn(c[0]) for c in base_cls]
    cand_main = [_collapse_fn(c[0]) for c in cand_cls]
    base_cond = [c[1].value for c in base_cls]
    cand_cond = [c[1].value for c in cand_cls]
    base_comb = [f"{m}|{c}" for m, c in zip(base_main, base_cond, strict=True)]
    cand_comb = [f"{m}|{c}" for m, c in zip(cand_main, cand_cond, strict=True)]

    return {
        "MasterTemplateF1": multiset_f1(base_main, cand_main),
        "MasterConditionF1": multiset_f1(base_cond, cand_cond),
        "MasterConditionWithTemplateF1": multiset_f1(base_comb, cand_comb),
    }


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", text.lower())


def sentence_bleu_smoothed(references: list[str], hypothesis: str, max_n: int = 4) -> float:
    """BLEU suavizado em [0, 1] por sentença candidata contra múltiplas referências."""
    hyp_tokens = _tokenize(hypothesis)
    if not hyp_tokens or not references:
        return 0.0
    ref_token_lists = [_tokenize(r) for r in references if r.strip()]
    if not ref_token_lists:
        return 0.0

    log_prec_sum = 0.0
    effective_orders = 0
    for n in range(1, max_n + 1):
        if len(hyp_tokens) < n:
            continue
        effective_orders += 1
        hyp_ngrams = Counter(tuple(hyp_tokens[i : i + n]) for i in range(len(hyp_tokens) - n + 1))
        max_ref_ngrams: Counter[tuple[str, ...]] = Counter()
        for ref_toks in ref_token_lists:
            ref_counts = Counter(tuple(ref_toks[i : i + n]) for i in range(max(0, len(ref_toks) - n + 1)))
            for ng, cnt in ref_counts.items():
                if cnt > max_ref_ngrams[ng]:
                    max_ref_ngrams[ng] = cnt
        clipped = sum(min(cnt, max_ref_ngrams[ng]) for ng, cnt in hyp_ngrams.items())
        total = sum(hyp_ngrams.values())
        # Suavização +1 (Chen & Cherry Method 1) para evitar log(0) em sentenças curtas
        p_n = (clipped + 1.0) / (total + 1.0) if n > 1 else max(clipped / max(total, 1), 1e-9)
        log_prec_sum += math.log(p_n)

    if effective_orders == 0:
        return 0.0
    closest_ref_len = min(ref_token_lists, key=lambda r: (abs(len(r) - len(hyp_tokens)), len(r)))
    r_len = len(closest_ref_len)
    h_len = len(hyp_tokens)
    bp = 1.0 if h_len > r_len else math.exp(1.0 - (r_len / max(h_len, 1)))
    return float(bp * math.exp(log_prec_sum / effective_orders))


def compute_corpus_bleu_avg(ground_truth_reqs: list[str], candidate_reqs: list[str]) -> float:
    """Média do BLEU por candidato conforme Seção IV-D1a de Roßkothen et al."""
    if not candidate_reqs:
        return 0.0
    return sum(sentence_bleu_smoothed(ground_truth_reqs, c) for c in candidate_reqs) / len(candidate_reqs)


def compute_embedding_movers_distance(
    base_embeddings: list[list[float]],
    candidate_embeddings: list[list[float]],
) -> float:
    """Calcula EMD normalizado em [0, 1] (`1 / (1 + wasserstein_distance_nd)`), idêntico a `metrics.py`."""
    if not base_embeddings or not candidate_embeddings:
        return 0.0
    dist = float(wasserstein_distance_nd(base_embeddings, candidate_embeddings))
    return 1.0 / (1.0 + dist)


def summarize_operational_metrics(records: list[ExperimentRunRecord]) -> dict[str, float]:
    """Agrega métricas de custo, consumo de tokens, latência, iterações e taxa de escalonamento."""
    if not records:
        return {}
    n = len(records)
    total_cost = sum(r.total_cost_usd for r in records)
    return {
        "num_requirements": float(n),
        "total_cost_usd": total_cost,
        "mean_cost_per_req_usd": total_cost / n,
        "cost_per_1000_reqs_usd": (total_cost / n) * 1000.0,
        "mean_input_tokens": sum(r.total_input_tokens for r in records) / n,
        "mean_output_tokens": sum(r.total_output_tokens for r in records) / n,
        "mean_generator_latency_ms": sum(r.generator_latency_ms for r in records) / n,
        "mean_validation_latency_ms": sum(r.validation_latency_ms for r in records) / n,
        "mean_total_latency_ms": sum(r.total_latency_ms for r in records) / n,
        "mean_iterations": sum(r.iterations_used for r in records) / n,
        "fast_accept_rate": sum(1 for r in records if r.final_cascade_route == "accept" and r.iterations_used == 0) / n,
        "accepted_total_rate": sum(1 for r in records if not r.escalated_to_human) / n,
        "escalation_rate": sum(1 for r in records if r.escalated_to_human) / n,
    }


def paired_wilcoxon(scores_a: list[float], scores_b: list[float]) -> tuple[float, float]:
    """Teste pareado de Wilcoxon signed-rank entre duas configurações sobre os mesmos requisitos."""
    diffs = [a - b for a, b in zip(scores_a, scores_b, strict=True)]
    if all(abs(d) < 1e-12 for d in diffs):
        return 0.0, 1.0
    stat, p_val = wilcoxon(scores_a, scores_b, zero_method="wilcox")
    return float(stat), float(p_val)


def spearman_corr(x: list[float], y: list[float]) -> tuple[float, float]:
    """Correlação de postos de Spearman (rho, p-valor)."""
    res = spearmanr(x, y)
    return float(res.statistic), float(res.pvalue)

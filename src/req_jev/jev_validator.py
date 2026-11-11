"""Validador System One baseado no JEV (`typesafe-sdk`) + Roteador Cascata Calibrado (Accept/Rewrite/Escalate)."""

from __future__ import annotations

import os
import re
import time
from typing import Protocol

from typesafe_sdk import AsyncTypeSafeClient, Choice, Noul, Score

from req_jev.prompts_loader import estimate_call_cost_usd
from req_jev.schemas import CallTelemetry, CascadeRoute, JEVRequirementAudit

MODAL_VERB_REGEX = re.compile(r"\b(shall|should|will)\b")

# Mapeamento determinístico entre cada primitiva Noul acionável e a instrução de reparo enviada ao gerador.
# Note que TODAS as regras abaixo são estritamente deriváveis do requisito fonte S ou da gramática MASTeR,
# eliminando por construção a necessidade de um LLM extra para Feedback Filtering.
DETERMINISTIC_REPAIR_INSTRUCTIONS: dict[str, str] = {
    "has_modal_verb": (
        "Include exactly one mandatory MASTeR modal verb ('shall', 'should', or 'will') "
        "in its proper grammatical slot."
    ),
    "conforms_to_master_syntax": (
        "Restructure the sentence so it strictly follows one of the MASTeR main patterns "
        "(FunctionMASTeR, PropertyMASTeR, EnvironmentMASTeR, or ProcessMASTeR) and valid "
        "condition prefixes ('If ...', 'As soon as ...', or 'As long as ...') if conditional."
    ),
    "is_singular_atomic": (
        "Ensure singularity: split compound actions, multiple main verbs, or independent "
        "capabilities into separate atomic MASTeR requirements."
    ),
    "has_explicit_actor_or_system": (
        "Explicitly name the target <system>, <actor>, or <subject matter> and <object> "
        "from the source requirement instead of ambiguous pronouns or agentless passive voice."
    ),
    "preserves_source_details": (
        "Retain all relevant conditions, qualifiers, and constraints explicitly stated in "
        "the source requirement without omitting semantics."
    ),
    "no_hallucinated_values": (
        "Remove any invented numbers, thresholds, or units that do not appear in the source "
        "requirement; use only information grounded in the source text."
    ),
}


class JEVValidatorProtocol(Protocol):
    """Contrato assíncrono do validador JEV System One."""

    model_name: str

    async def audit_candidate(
        self,
        *,
        source_text: str,
        candidate_text: str,
        tau_accept: float = 0.75,
        tau_reject: float = 0.40,
    ) -> tuple[JEVRequirementAudit, CallTelemetry]:
        ...


def build_jev_questions() -> dict[str, Noul | Choice | Score]:
    """Constrói o conjunto de primitivas tipadas avaliadas em paralelo num único passe System One."""
    return {
        "main_template": Choice(
            instructions=(
                "Classify which MASTeR main template pattern best matches `candidate_requirement`, "
                "or select `NonConforming` if it violates all MASTeR main patterns."
            ),
            criteria={
                "FunctionMASTeR - Independent System Activity": (
                    "[<condition>] the system <system> {shall|should|will} <process verb> <object> [...]."
                ),
                "FunctionMASTeR - User Interaction": (
                    "[<condition>] the system <system> {shall|should|will} provide <actor> with the ability to <process verb> <object> [...]."
                ),
                "FunctionMASTeR - Interface requirement": (
                    "[<condition>] the system <system> {shall|should|will} be able to <process verb> <object> [...]."
                ),
                "PropertyMASTeR": (
                    "[<condition>] <characteristic> <subject matter> {shall|should|will} be [<qualifying expression>] <value>."
                ),
                "EnvironmentMASTeR": (
                    "<subject matter> {shall|should|will} be designed in a way [<condition>] <object> can be operated [...]."
                ),
                "ProcessMASTeR": (
                    "[<condition>] <actor> {shall|should|will} <process verb> <object> <value>."
                ),
                "NonConforming": (
                    "Does not conform to any valid MASTeR main template structure."
                ),
            },
        ),
        "condition_template": Choice(
            instructions="Identify the MASTeR condition sub-pattern at the beginning of `candidate_requirement`.",
            criteria={
                "LogicMASTeR": "Starts with 'If ...' logical condition (or valid chained conditions).",
                "EventMASTeR": "Starts with 'As soon as ...' event trigger.",
                "TimeMASTeR": "Starts with 'As long as ...' state/duration condition.",
                "No Condition": "Unconditional MASTeR requirement with no condition clause.",
                "InvalidCondition": "Uses non-MASTeR conditional keywords such as 'When', 'While', 'Whenever', or 'In case' without proper MASTeR syntax.",
            },
        ),
        "has_modal_verb": Noul(
            instructions=(
                "Does `candidate_requirement` properly use one of the mandatory MASTeR modal verbs "
                "('shall', 'should', or 'will')?"
            )
        ),
        "conforms_to_master_syntax": Noul(
            instructions=(
                "Does `candidate_requirement` strictly adhere to the syntactic slot order of its "
                "MASTeR main and optional condition template (including repeating condition keywords after 'and'/'or')?"
            )
        ),
        "is_singular_atomic": Noul(
            instructions=(
                "Is `candidate_requirement` singular and atomic (expressing a single capability, "
                "property, or constraint rather than bundling multiple independent obligations)?"
            )
        ),
        "has_explicit_actor_or_system": Noul(
            instructions=(
                "Does `candidate_requirement` clearly state the subject (<system>, <actor>, or <characteristic>) "
                "without vague pronouns ('it', 'they') given `source_requirement`?"
            )
        ),
        "preserves_source_details": Noul(
            instructions=(
                "Does `candidate_requirement` preserve the relevant conditions and qualifiers present in "
                "`source_requirement` for this atomic aspect?"
            )
        ),
        "no_hallucinated_values": Noul(
            instructions=(
                "Is `candidate_requirement` free from invented numbers, units, or thresholds that do NOT "
                "exist in `source_requirement`?"
            )
        ),
        "quality_rubric": Score(
            instructions=(
                "Rate how ready `candidate_requirement` is as a MASTeR formalization of `source_requirement` "
                "(from C4 unhelpful to C1 ready without edits)."
            ),
            criteria=[
                "C4 - Unhelpful: severe template violation, hallucination, or distorted meaning",
                "C3 - Useful draft: captures partial intent but needs substantial structural edits",
                "C2 - Minor changes needed: mostly conforms to MASTeR with small phrasing issues",
                "C1 - No update needed: accurate, atomic, and fully conforming MASTeR requirement",
            ],
        ),
    }


def route_from_probabilities(
    *,
    candidate_text: str,
    main_template: str,
    main_confidence: float,
    condition_template: str,
    condition_confidence: float,
    noul_scores: dict[str, float],
    quality_score: float,
    quality_confidence: float,
    tau_accept: float = 0.75,
    tau_reject: float = 0.40,
) -> JEVRequirementAudit:
    """Aplica o roteamento cascata calibrado (Accept / Rewrite / Escalate) sobre as saídas do JEV."""
    # Pre-check determinístico rápido para verbo modal obrigatório
    if not MODAL_VERB_REGEX.search(candidate_text):
        noul_scores = dict(noul_scores)
        noul_scores["has_modal_verb"] = min(noul_scores.get("has_modal_verb", 0.0), 0.05)

    failed_checks: list[str] = []
    uncertain_checks: list[str] = []

    if main_template == "NonConforming" and main_confidence >= (1.0 - tau_reject):
        if "conforms_to_master_syntax" not in failed_checks:
            failed_checks.append("conforms_to_master_syntax")
    elif main_template == "NonConforming":
        uncertain_checks.append("main_template")

    if condition_template == "InvalidCondition" and condition_confidence >= (1.0 - tau_reject):
        if "conforms_to_master_syntax" not in failed_checks:
            failed_checks.append("conforms_to_master_syntax")
    elif condition_template == "InvalidCondition":
        uncertain_checks.append("condition_template")

    for check_key, prob in noul_scores.items():
        if prob < tau_reject:
            if check_key not in failed_checks:
                failed_checks.append(check_key)
        elif prob < tau_accept:
            uncertain_checks.append(check_key)

    deterministic_feedback = [
        DETERMINISTIC_REPAIR_INSTRUCTIONS[k]
        for k in failed_checks
        if k in DETERMINISTIC_REPAIR_INSTRUCTIONS
    ]

    # Lógica de decisão em cascata:
    # 1. Se há falha clara acionável (p < tau_reject) -> REWRITE determinístico.
    # 2. Se não há falha clara, mas existe incerteza epistêmica (tau_reject <= p < tau_accept)
    #    ou qualidade estimada abaixo de C2 (< 1.8 na escala 0..3) -> ESCALATE humano.
    # 3. Caso contrário (todos os checks >= tau_accept e qualidade >= 1.8) -> FAST ACCEPT.
    if failed_checks:
        route: CascadeRoute = "rewrite"
    elif uncertain_checks or quality_score < 1.8:
        route = "escalate"
    else:
        route = "accept"

    return JEVRequirementAudit(
        candidate_text=candidate_text,
        main_template=main_template,
        main_template_confidence=main_confidence,
        condition_template=condition_template,
        condition_template_confidence=condition_confidence,
        noul_scores=noul_scores,
        quality_score=quality_score,
        quality_confidence=quality_confidence,
        route=route,
        failed_checks=failed_checks,
        uncertain_checks=uncertain_checks,
        deterministic_feedback=deterministic_feedback,
    )


class TypeSafeJEVValidator:
    """Integração oficial assíncrona com TypeSafe AI (`AsyncTypeSafeClient.system_one`)."""

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        key = api_key or os.environ.get("TYPESAFE_API_KEY")
        if not key:
            raise ValueError("Defina TYPESAFE_API_KEY no arquivo .env para usar a API oficial do JEV.")
        self.api_key = key
        self.model_name = model or os.environ.get("TYPESAFE_DEFAULT_MODEL", "jev-latest")
        self._questions = build_jev_questions()

    async def audit_candidate(
        self,
        *,
        source_text: str,
        candidate_text: str,
        tau_accept: float = 0.75,
        tau_reject: float = 0.40,
    ) -> tuple[JEVRequirementAudit, CallTelemetry]:
        state_payload = {
            "source_requirement": source_text,
            "candidate_requirement": candidate_text,
        }
        t0 = time.perf_counter()
        async with AsyncTypeSafeClient(api_key=self.api_key) as client:
            response = await client.system_one(
                model=self.model_name,
                state=state_payload,
                questions=self._questions,
            )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        main_ans = response.answers["main_template"]
        cond_ans = response.answers["condition_template"]
        qual_ans = response.answers["quality_rubric"]

        noul_keys = list(DETERMINISTIC_REPAIR_INSTRUCTIONS.keys())
        noul_scores = {k: float(response.answers[k].noul) for k in noul_keys}  # type: ignore[union-attr]

        audit = route_from_probabilities(
            candidate_text=candidate_text,
            main_template=str(main_ans.choice),  # type: ignore[union-attr]
            main_confidence=float(main_ans.confidence),  # type: ignore[union-attr]
            condition_template=str(cond_ans.choice),  # type: ignore[union-attr]
            condition_confidence=float(cond_ans.confidence),  # type: ignore[union-attr]
            noul_scores=noul_scores,
            quality_score=float(qual_ans.score),  # type: ignore[union-attr]
            quality_confidence=float(qual_ans.confidence),  # type: ignore[union-attr]
            tau_accept=tau_accept,
            tau_reject=tau_reject,
        )

        in_tok = int(response.usage.input_tokens or 0)
        out_tok = int(response.usage.output_tokens or 0)
        telemetry = CallTelemetry(
            stage="jev_system_one",
            model=response.model or self.model_name,
            input_tokens=in_tok,
            output_tokens=out_tok,
            latency_ms=latency_ms,
            estimated_cost_usd=estimate_call_cost_usd(self.model_name, in_tok, out_tok),
        )
        return audit, telemetry

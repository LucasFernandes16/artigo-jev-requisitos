"""Testes automatizados offline (smoke tests) validando dataset, prompts, roteador JEV e pipelines."""

from __future__ import annotations

import asyncio
from pathlib import Path
import sys
from typing import Literal, TypeVar

from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from req_jev.dataset import load_source_dataset  # noqa: E402
from req_jev.evaluation import compute_master_template_f1, summarize_operational_metrics  # noqa: E402
from req_jev.jev_validator import route_from_probabilities  # noqa: E402
from req_jev.pipelines import (  # noqa: E402
    run_jev_cascade_pipeline,
    run_llm_judge_pipeline,
    run_oneshot_pipeline,
)
from req_jev.prompts_loader import load_prompt  # noqa: E402
from req_jev.schemas import (  # noqa: E402
    ActionabilityCheckOutput,
    ActionabilityCheckResult,
    CallTelemetry,
    FormalizationLLMOutput,
    FormalizedRequirement,
    IncoseEvaluationOutput,
    JEVRequirementAudit,
    LLMScoreValidator,
    TemplateValidatorOutput,
)

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class DummyLLMBackend:
    """Backend determinístico em memória para testar os 3 pipelines sem consumo de API."""

    provider_name: Literal["gemini", "openrouter"] = "gemini"

    def __init__(self) -> None:
        self.gen_calls = 0

    async def generate_structured(
        self,
        *,
        stage: str,
        model: str,
        system_prompt: str | None,
        messages: list[dict[str, str]],
        response_schema: type[SchemaT],
        temperature: float = 0.0,
    ) -> tuple[SchemaT, CallTelemetry]:
        tel = CallTelemetry(
            stage=stage,  # type: ignore[arg-type]
            model=model,
            input_tokens=1500,
            output_tokens=120,
            latency_ms=850.0,
            estimated_cost_usd=0.00075,
        )
        if response_schema is FormalizationLLMOutput:
            self.gen_calls += 1
            if self.gen_calls == 1:
                # Primeira geração propositalmente sem verbo modal para acionar 1 ciclo de reescrita
                out = FormalizationLLMOutput(
                    action="reformulate",
                    requirements=[
                        FormalizedRequirement(
                            name="Emergency Safe Modes",
                            content="The AOCS defines Emergency and Safe modes.",
                        )
                    ],
                )
            else:
                out = FormalizationLLMOutput(
                    action="simple_split",
                    requirements=[
                        FormalizedRequirement(
                            name="Include Emergency Mode",
                            content="The system AOCS shall include an Emergency mode.",
                        ),
                        FormalizedRequirement(
                            name="Include Safe Mode",
                            content="The system AOCS shall include a Safe mode.",
                        ),
                    ],
                )
            return out, tel  # type: ignore[return-value]

        if response_schema is TemplateValidatorOutput:
            return TemplateValidatorOutput(reason="Missing modal verb shall.", valid=False), tel  # type: ignore[return-value]

        if response_schema is IncoseEvaluationOutput:
            ok = LLMScoreValidator(comment=None, score=0.95)
            bad = LLMScoreValidator(comment="Split into singular requirements.", score=0.4)
            return (
                IncoseEvaluationOutput(
                    necessary=ok,
                    appropriate=ok,
                    unambiguous=ok,
                    complete=ok,
                    singular=bad,
                    feasible=ok,
                    verifiable=ok,
                    correct=ok,
                    conforming=bad,
                ),
                tel,
            )  # type: ignore[return-value]

        if response_schema is ActionabilityCheckOutput:
            return (
                ActionabilityCheckOutput(
                    actionability_check_results=[
                        ActionabilityCheckResult(name="TemplateValidator", comment="ok", is_actionable=True),
                        ActionabilityCheckResult(name="singular", comment="ok", is_actionable=True),
                    ]
                ),
                tel,
            )  # type: ignore[return-value]

        raise AssertionError(f"Schema inesperado: {response_schema}")


class DummyJEVValidator:
    """Simulador System One do JEV para validar o fluxo Accept / Rewrite / Escalate."""

    model_name = "jev-mock"

    async def audit_candidate(
        self,
        *,
        source_text: str,
        candidate_text: str,
        tau_accept: float = 0.75,
        tau_reject: float = 0.40,
    ) -> tuple[JEVRequirementAudit, CallTelemetry]:
        has_shall = " shall " in f" {candidate_text} "
        audit = route_from_probabilities(
            candidate_text=candidate_text,
            main_template=(
                "FunctionMASTeR - Independent System Activity" if has_shall else "NonConforming"
            ),
            main_confidence=0.92,
            condition_template="No Condition",
            condition_confidence=0.96,
            noul_scores={
                "has_modal_verb": 0.98 if has_shall else 0.05,
                "conforms_to_master_syntax": 0.91 if has_shall else 0.20,
                "is_singular_atomic": 0.90 if has_shall else 0.30,
                "has_explicit_actor_or_system": 0.94,
                "preserves_source_details": 0.89,
                "no_hallucinated_values": 0.97,
            },
            quality_score=2.7 if has_shall else 0.8,
            quality_confidence=0.88,
            tau_accept=tau_accept,
            tau_reject=tau_reject,
        )
        tel = CallTelemetry(
            stage="jev_system_one",
            model=self.model_name,
            input_tokens=320,
            output_tokens=0,
            latency_ms=110.0,
            estimated_cost_usd=0.000032,
        )
        return audit, tel


def test_dataset_and_prompts() -> None:
    items = load_source_dataset()
    assert len(items) == 249
    assert sum(len(x.target_master_requirements) for x in items) == 410
    sys_prompt = load_prompt("formalization_task_prompt.jinja")
    assert "FunctionMASTeR" in sys_prompt
    in_prompt = load_prompt("formalization_input_prompt.jinja", target_requirement=items[0].to_requirement_content())
    assert items[0].req_name in in_prompt


def test_cascade_routing_modes() -> None:
    # 1. Caso de incerteza epistêmica (probabilidade ~0.55) -> deve escalar para humano
    uncertain_audit = route_from_probabilities(
        candidate_text="As soon as the event fault happens, the system AOCS shall log the fault.",
        main_template="FunctionMASTeR - Independent System Activity",
        main_confidence=0.85,
        condition_template="EventMASTeR",
        condition_confidence=0.90,
        noul_scores={
            "has_modal_verb": 0.99,
            "conforms_to_master_syntax": 0.88,
            "is_singular_atomic": 0.55,  # Zona de incerteza entre 0.40 e 0.75!
            "has_explicit_actor_or_system": 0.90,
            "preserves_source_details": 0.82,
            "no_hallucinated_values": 0.95,
        },
        quality_score=2.1,
        quality_confidence=0.70,
    )
    assert uncertain_audit.route == "escalate"
    assert "is_singular_atomic" in uncertain_audit.uncertain_checks


def test_all_three_pipelines_end_to_end() -> None:
    item = load_source_dataset()[0]

    async def _run() -> None:
        r_one = await run_oneshot_pipeline(
            run_id="t_one",
            backend=DummyLLMBackend(),
            generator_model="gemini-2.5-flash",
            item=item,
        )
        assert r_one.iterations_used == 0

        r_llm = await run_llm_judge_pipeline(
            run_id="t_llm",
            generator_backend=DummyLLMBackend(),
            judge_backend=DummyLLMBackend(),
            generator_model="gemini-2.5-flash",
            judge_model="gemini-2.5-flash",
            item=item,
            max_iterations=2,
        )
        assert r_llm.iterations_used >= 1

        r_jev = await run_jev_cascade_pipeline(
            run_id="t_jev",
            generator_backend=DummyLLMBackend(),
            jev_validator=DummyJEVValidator(),
            generator_model="gemini-2.5-flash",
            item=item,
            max_iterations=2,
        )
        assert r_jev.iterations_used == 1
        assert r_jev.final_cascade_route == "accept"
        assert r_jev.validation_latency_ms < r_llm.validation_latency_ms
        assert r_jev.total_cost_usd < r_llm.total_cost_usd

        f1 = compute_master_template_f1(item.target_master_requirements, r_jev.generated_requirements)
        assert 0.0 <= f1["MasterTemplateF1"] <= 1.0
        summary = summarize_operational_metrics([r_jev])
        assert summary["accepted_total_rate"] == 1.0

    asyncio.run(_run())

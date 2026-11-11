"""Implementação controlada dos três pipelines experimentais:
1. `oneshot`: geração direta sem loop de validação.
2. `llm_judge`: Reflection Loop original de Roßkothen et al. (Template + INCOSE + Feedback Filter via LLM).
3. `jev_cascade`: Arquitetura proposta combinando validação System One (JEV) + Roteamento Cascata Calibrado.
"""

from __future__ import annotations

from req_jev.dataset import SourceDatasetItem
from req_jev.jev_validator import JEVValidatorProtocol
from req_jev.llm_backends import StructuredLLMBackend
from req_jev.prompts_loader import load_prompt
from req_jev.schemas import (
    ActionabilityCheckOutput,
    CallTelemetry,
    CascadeRoute,
    ExperimentRunRecord,
    FormalizationLLMOutput,
    IncoseEvaluationOutput,
    JEVRequirementAudit,
    TemplateValidatorOutput,
)

HYBRID_MASTER_CHECKER_USER_PROMPT = "Please validate the following requirement:\n\n{text_content}"
LOOP_FEEDBACK_WRAPPER_USER_PROMPT = "The formalization was not successful.\n\n{feedback}\n\nPlease try again."
INCOSE_THRESHOLD = 0.7


def _check_grammar_fast(req_text: str) -> bool:
    """Pré-checagem sintática rápida (espelhando o pré-filtro híbrido da Seção III-C1 de Roßkothen et al.)."""
    has_modal = any(f" {m} " in f" {req_text} " for m in ("shall", "should", "will"))
    if not has_modal:
        return False
    lowered = req_text.strip()
    if any(lowered.startswith(bad) for bad in ("When ", "While ", "Whenever ", "In case ")):
        return False
    return True


def _finalize_record(
    *,
    run_id: str,
    mode: str,
    backend: StructuredLLMBackend,
    generator_model: str,
    validator_model: str,
    item: SourceDatasetItem,
    result: FormalizationLLMOutput,
    iterations_used: int,
    calls: list[CallTelemetry],
    final_route: CascadeRoute = "accept",
    jev_audits: list[JEVRequirementAudit] | None = None,
) -> ExperimentRunRecord:
    gen_reqs = (
        [item.source_text]
        if result.action == "no_update" or not result.requirements
        else [r.content for r in result.requirements]
    )
    gen_lat = sum(c.latency_ms for c in calls if c.stage == "generator")
    val_lat = sum(c.latency_ms for c in calls if c.stage != "generator")
    return ExperimentRunRecord(
        run_id=run_id,
        mode=mode,  # type: ignore[arg-type]
        provider=backend.provider_name,
        generator_model=generator_model,
        validator_model=validator_model,
        req_name=item.req_name,
        project_set=item.project_set,
        source_length=item.source_length,
        ground_truth_split_factor=item.split_factor,
        source_requirement=item.source_text,
        ground_truth_requirements=item.target_master_requirements,
        final_action=result.action,
        generated_requirements=gen_reqs,
        iterations_used=iterations_used,
        final_cascade_route=final_route,
        escalated_to_human=(final_route == "escalate"),
        total_input_tokens=sum(c.input_tokens for c in calls),
        total_output_tokens=sum(c.output_tokens for c in calls),
        generator_latency_ms=gen_lat,
        validation_latency_ms=val_lat,
        total_latency_ms=gen_lat + val_lat,
        total_cost_usd=sum(c.estimated_cost_usd for c in calls),
        calls=calls,
        jev_audits=jev_audits or [],
    )


async def run_oneshot_pipeline(
    *,
    run_id: str,
    backend: StructuredLLMBackend,
    generator_model: str,
    item: SourceDatasetItem,
) -> ExperimentRunRecord:
    """Executa formalização em passo único sem validadores no loop (baseline de custo mínimo)."""
    system_prompt = load_prompt("formalization_task_prompt.jinja")
    user_prompt = load_prompt(
        "formalization_input_prompt.jinja",
        target_requirement=item.to_requirement_content(),
    )
    result, tel = await backend.generate_structured(
        stage="generator",
        model=generator_model,
        system_prompt=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
        response_schema=FormalizationLLMOutput,
    )
    return _finalize_record(
        run_id=run_id,
        mode="oneshot",
        backend=backend,
        generator_model=generator_model,
        validator_model="none",
        item=item,
        result=result,
        iterations_used=0,
        calls=[tel],
        final_route="accept",
    )


async def run_llm_judge_pipeline(
    *,
    run_id: str,
    generator_backend: StructuredLLMBackend,
    judge_backend: StructuredLLMBackend,
    generator_model: str,
    judge_model: str,
    item: SourceDatasetItem,
    max_iterations: int = 3,
    incose_threshold: float = INCOSE_THRESHOLD,
) -> ExperimentRunRecord:
    """Executa o Reflection Loop original (Roßkothen et al., Seção III) com LLM-as-a-Judge + Feedback Filter."""
    req_content = item.to_requirement_content()
    system_prompt = load_prompt("formalization_task_prompt.jinja")
    template_validator_sys = load_prompt("master_validator_prompt.md.jinja")
    user_prompt = load_prompt("formalization_input_prompt.jinja", target_requirement=req_content)

    calls: list[CallTelemetry] = []
    result, gen_tel = await generator_backend.generate_structured(
        stage="generator",
        model=generator_model,
        system_prompt=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
        response_schema=FormalizationLLMOutput,
    )
    calls.append(gen_tel)

    if result.action == "no_update" or not result.requirements:
        return _finalize_record(
            run_id=run_id,
            mode="llm_judge",
            backend=generator_backend,
            generator_model=generator_model,
            validator_model=judge_model,
            item=item,
            result=result,
            iterations_used=0,
            calls=calls,
        )

    conversation: list[dict[str, str]] = [
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": result.model_dump_json()},
    ]
    iterations_used = 0

    for _ in range(max_iterations):
        all_feedback: list[dict[str, str]] = []

        for cand in result.requirements:
            # 1. Template Validator (pré-filtro sintático + confirmação por LLM-as-a-Judge)
            if not _check_grammar_fast(cand.content):
                tmpl_out, tmpl_tel = await judge_backend.generate_structured(
                    stage="llm_template_validator",
                    model=judge_model,
                    system_prompt=template_validator_sys,
                    messages=[
                        {
                            "role": "user",
                            "content": HYBRID_MASTER_CHECKER_USER_PROMPT.format(text_content=cand.content),
                        }
                    ],
                    response_schema=TemplateValidatorOutput,
                )
                calls.append(tmpl_tel)
                if not tmpl_out.valid:
                    all_feedback.append(
                        {
                            "name": "TemplateValidator",
                            "comment": tmpl_out.reason or "Requirement violates MASTeR template syntax.",
                        }
                    )

            # 2. INCOSE Validator (avalia C1-C9)
            incose_prompt = load_prompt(
                "incose_validator_prompt.md.jinja",
                requirement_type=req_content.requirement_type,
                requirement=cand.content,
            )
            incose_out, incose_tel = await judge_backend.generate_structured(
                stage="llm_incose_validator",
                model=judge_model,
                system_prompt=None,
                messages=[{"role": "user", "content": incose_prompt}],
                response_schema=IncoseEvaluationOutput,
            )
            calls.append(incose_tel)
            for cat_name, score_obj in incose_out.as_dict().items():
                if score_obj.score < incose_threshold and score_obj.comment:
                    all_feedback.append({"name": cat_name, "comment": score_obj.comment})

        if not all_feedback:
            break

        # 3. Feedback Filter (descarta pedidos não suportados pelo requisito fonte S)
        filter_prompt = load_prompt(
            "validator_result_filter_prompt.md.jinja",
            source_requirements=[req_content],
            requirement=result.requirements[0],
            feedback=all_feedback,
        )
        filter_out, filter_tel = await judge_backend.generate_structured(
            stage="llm_feedback_filter",
            model=judge_model,
            system_prompt=None,
            messages=[{"role": "user", "content": filter_prompt}],
            response_schema=ActionabilityCheckOutput,
        )
        calls.append(filter_tel)

        actionable = [
            fb
            for fb, chk in zip(all_feedback, filter_out.actionability_check_results, strict=False)
            if chk.is_actionable
        ]
        if not actionable:
            break

        iterations_used += 1
        feedback_text = "\n".join(f"- [{fb['name']}]: {fb['comment']}" for fb in actionable)
        conversation.append(
            {
                "role": "user",
                "content": LOOP_FEEDBACK_WRAPPER_USER_PROMPT.format(feedback=feedback_text),
            }
        )
        result, regen_tel = await generator_backend.generate_structured(
            stage="generator",
            model=generator_model,
            system_prompt=system_prompt,
            messages=conversation,
            response_schema=FormalizationLLMOutput,
        )
        calls.append(regen_tel)
        conversation.append({"role": "assistant", "content": result.model_dump_json()})

    return _finalize_record(
        run_id=run_id,
        mode="llm_judge",
        backend=generator_backend,
        generator_model=generator_model,
        validator_model=judge_model,
        item=item,
        result=result,
        iterations_used=iterations_used,
        calls=calls,
        final_route="accept",
    )


async def run_jev_cascade_pipeline(
    *,
    run_id: str,
    generator_backend: StructuredLLMBackend,
    jev_validator: JEVValidatorProtocol,
    generator_model: str,
    item: SourceDatasetItem,
    max_iterations: int = 3,
    tau_accept: float = 0.75,
    tau_reject: float = 0.40,
) -> ExperimentRunRecord:
    """Executa a arquitetura proposta JEV-Cascade Reflection Loop (Abordagens 1 + 2)."""
    req_content = item.to_requirement_content()
    system_prompt = load_prompt("formalization_task_prompt.jinja")
    user_prompt = load_prompt("formalization_input_prompt.jinja", target_requirement=req_content)

    calls: list[CallTelemetry] = []
    latest_audits: list[JEVRequirementAudit] = []

    result, gen_tel = await generator_backend.generate_structured(
        stage="generator",
        model=generator_model,
        system_prompt=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
        response_schema=FormalizationLLMOutput,
    )
    calls.append(gen_tel)

    if result.action == "no_update" or not result.requirements:
        return _finalize_record(
            run_id=run_id,
            mode="jev_cascade",
            backend=generator_backend,
            generator_model=generator_model,
            validator_model=jev_validator.model_name,
            item=item,
            result=result,
            iterations_used=0,
            calls=calls,
            final_route="accept",
        )

    conversation: list[dict[str, str]] = [
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": result.model_dump_json()},
    ]
    iterations_used = 0
    final_route: CascadeRoute = "accept"

    for attempt in range(max_iterations + 1):
        latest_audits = []
        for cand in result.requirements:
            audit, jev_tel = await jev_validator.audit_candidate(
                source_text=item.source_text,
                candidate_text=cand.content,
                tau_accept=tau_accept,
                tau_reject=tau_reject,
            )
            calls.append(jev_tel)
            latest_audits.append(audit)

        # Agrega decisão do conjunto de candidatos gerados para o requisito fonte S
        has_rewrite = any(a.route == "rewrite" for a in latest_audits)
        has_escalate = any(a.route == "escalate" for a in latest_audits)

        if not has_rewrite and not has_escalate:
            # PATH A: Fast Accept — todos os candidatos aprovados com alta confiança calibrada
            final_route = "accept"
            break

        if has_rewrite and attempt < max_iterations:
            # PATH B: Deterministic Guided Rewrite — mapeia falhas claras para instruções concisas
            feedback_lines: list[str] = []
            for idx, audit in enumerate(latest_audits, start=1):
                if audit.deterministic_feedback:
                    joined = " ".join(audit.deterministic_feedback)
                    feedback_lines.append(f"- [Requirement {idx} ('{audit.candidate_text}')]: {joined}")
            if not feedback_lines:
                final_route = "escalate"
                break

            iterations_used += 1
            conversation.append(
                {
                    "role": "user",
                    "content": LOOP_FEEDBACK_WRAPPER_USER_PROMPT.format(
                        feedback="\n".join(feedback_lines)
                    ),
                }
            )
            result, regen_tel = await generator_backend.generate_structured(
                stage="generator",
                model=generator_model,
                system_prompt=system_prompt,
                messages=conversation,
                response_schema=FormalizationLLMOutput,
            )
            calls.append(regen_tel)
            conversation.append({"role": "assistant", "content": result.model_dump_json()})
            if not result.requirements:
                final_route = "escalate"
                break
        else:
            # PATH C: Escalate — zona de incerteza epistêmica ou persistência de falha após M ciclos
            final_route = "escalate"
            break

    return _finalize_record(
        run_id=run_id,
        mode="jev_cascade",
        backend=generator_backend,
        generator_model=generator_model,
        validator_model=jev_validator.model_name,
        item=item,
        result=result,
        iterations_used=iterations_used,
        calls=calls,
        final_route=final_route,
        jev_audits=latest_audits,
    )

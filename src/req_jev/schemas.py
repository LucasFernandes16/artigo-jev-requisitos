"""Schemas Pydantic compartilhados para geração, validação (LLM-Judge e JEV) e telemetria."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class IsoLLMValidatorCategory(StrEnum):
    """Nove características individuais de qualidade de requisitos do guia INCOSE."""

    NECESSARY = "necessary"
    APPROPRIATE = "appropriate"
    UNAMBIGUOUS = "unambiguous"
    COMPLETE = "complete"
    SINGULAR = "singular"
    FEASIBLE = "feasible"
    VERIFIABLE = "verifiable"
    CORRECT = "correct"
    CONFORMING = "conforming"


class MasterTemplatePattern(StrEnum):
    """Padrões principais do sistema de templates MASTeR/SOPHIST."""

    FunctionalIndependent = "FunctionMASTeR - Independent System Activity"
    FunctionalUser = "FunctionMASTeR - User Interaction"
    FunctionalInterface = "FunctionMASTeR - Interface requirement"
    Property = "PropertyMASTeR"
    Environment = "EnvironmentMASTeR"
    Process = "ProcessMASTeR"


class MasterCondition(StrEnum):
    """Sub-padrões de condição do sistema MASTeR/SOPHIST."""

    Logic = "LogicMASTeR"
    Event = "EventMASTeR"
    Time = "TimeMASTeR"
    No = "No Condition"


class MasterTemplateClassification(BaseModel):
    """Classificação estruturada de um requisito MASTeR."""

    main_template: MasterTemplatePattern
    condition_template: MasterCondition


class RequirementContent(BaseModel):
    """Representação de um requisito fonte de entrada."""

    id: str
    requirement_type: str = "Requirement"
    is_heading: bool = False
    name: str
    text_content: str


class FormalizedRequirement(BaseModel):
    """Requisito atômico formalizado gerado pelo LLM."""

    name: str = Field(description="Short semantic name (max 5 words) of the requirement")
    content: str = Field(description="Textual content of the MASTeR requirement")


class SplitHeading(BaseModel):
    """Cabeçalho opcional criado quando a ação é 'heading'."""

    heading_name: str = Field(description="Name of the new heading.")
    heading_text_content: str = Field(description="Text of the new heading.")


class FormalizationLLMOutput(BaseModel):
    """Schema de saída estruturada da etapa de formalização (Figura 3 de Roßkothen et al.)."""

    action: Literal["no_update", "reformulate", "simple_split", "heading"] = Field(
        description="The selected action for the formalization."
    )
    requirements: list[FormalizedRequirement] = Field(
        description="List of the output MASTeR requirements."
    )
    heading: SplitHeading | None = Field(
        default=None,
        description="Definition of the new heading, required if the action is 'heading'.",
    )


class TemplateValidatorOutput(BaseModel):
    """Saída do Template Validator tradicional (LLM-as-a-Judge)."""

    reason: str | None = Field(default=None, description="Reasoning behind the validation result.")
    valid: bool = Field(description="Whether the requirement adheres to MASTeR templates.")


class LLMScoreValidator(BaseModel):
    """Nota e comentário por critério do INCOSE Validator tradicional."""

    comment: str | None = Field(
        default=None,
        description="Short concise feedback on how to improve the requirement.",
    )
    score: float = Field(
        description="Score between 0 and 1 indicating adherence to the criterion.",
        ge=0.0,
        le=1.0,
    )

    @field_validator("score", mode="before")
    @classmethod
    def clip_score_value(cls, value: Any) -> Any:
        if isinstance(value, (int, float)):
            return min(1.0, max(0.0, float(value)))
        if isinstance(value, str):
            try:
                return min(1.0, max(0.0, float(value)))
            except ValueError:
                return value
        return value


class IncoseEvaluationOutput(BaseModel):
    """Avaliação completa das 9 características INCOSE (compatível com Gemini e OpenAI)."""

    necessary: LLMScoreValidator
    appropriate: LLMScoreValidator
    unambiguous: LLMScoreValidator
    complete: LLMScoreValidator
    singular: LLMScoreValidator
    feasible: LLMScoreValidator
    verifiable: LLMScoreValidator
    correct: LLMScoreValidator
    conforming: LLMScoreValidator

    def as_dict(self) -> dict[str, LLMScoreValidator]:
        return {cat.value: getattr(self, cat.value) for cat in IsoLLMValidatorCategory}


class ActionabilityCheckResult(BaseModel):
    """Resultado individual do filtro de acionabilidade."""

    name: str = Field(description="Name of the validator")
    comment: str | None = Field(default=None, description="Justification regarding source S.")
    is_actionable: bool = Field(description="True if actionable from source S alone.")


class ActionabilityCheckOutput(BaseModel):
    """Lista de resultados do filtro de feedback acionável."""

    actionability_check_results: list[ActionabilityCheckResult]


CascadeRoute = Literal["accept", "rewrite", "escalate"]


class JEVRequirementAudit(BaseModel):
    """Resultado detalhado da auditoria System One do JEV sobre um candidato R_i."""

    candidate_text: str
    main_template: str
    main_template_confidence: float
    condition_template: str
    condition_template_confidence: float
    noul_scores: dict[str, float]
    quality_score: float
    quality_confidence: float
    route: CascadeRoute
    failed_checks: list[str]
    uncertain_checks: list[str]
    deterministic_feedback: list[str]


class CallTelemetry(BaseModel):
    """Telemetria individual de custo, tokens e latência de uma chamada de inferência."""

    stage: Literal[
        "generator",
        "llm_template_validator",
        "llm_incose_validator",
        "llm_feedback_filter",
        "jev_system_one",
    ]
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    estimated_cost_usd: float = 0.0


class ExperimentRunRecord(BaseModel):
    """Registro completo de execução experimental para um requisito fonte."""

    run_id: str
    mode: Literal["oneshot", "llm_judge", "jev_cascade"]
    provider: Literal["gemini", "openrouter"]
    generator_model: str
    validator_model: str
    req_name: str
    project_set: str
    source_length: int
    ground_truth_split_factor: int
    source_requirement: str
    ground_truth_requirements: list[str]
    final_action: str
    generated_requirements: list[str]
    iterations_used: int
    final_cascade_route: CascadeRoute = "accept"
    escalated_to_human: bool = False
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    generator_latency_ms: float = 0.0
    validation_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    total_cost_usd: float = 0.0
    calls: list[CallTelemetry] = Field(default_factory=list)
    jev_audits: list[JEVRequirementAudit] = Field(default_factory=list)

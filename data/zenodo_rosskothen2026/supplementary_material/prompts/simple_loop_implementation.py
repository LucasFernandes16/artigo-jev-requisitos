"""Implementation of the reflective formalization loop described in Section 3 (Approach)."""

from __future__ import annotations

import logging
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from jinja2 import Environment, FileSystemLoader
from openai import AsyncOpenAI
from pydantic import BaseModel, Field, create_model, field_validator

_logger = logging.getLogger(__name__)

PROMPT_DIR = Path(__file__).parent
LLM_MODEL_NAME = "TODO"

class IsoLLMValidatorCategory(StrEnum):
    """Categories for the LLM Validator."""

    NECESSARY = "necessary"
    APPROPRIATE = "appropriate"
    UNAMBIGUOUS = "unambiguous"
    COMPLETE = "complete"
    SINGULAR = "singular"
    FEASIBLE = "feasible"
    VERIFIABLE = "verifiable"
    CORRECT = "correct"
    CONFORMING = "conforming"


class FormalizationLLMOutput(BaseModel):
    """Output for the formalization of a requirement."""

    action: Literal["no_update", "reformulate", "simple_split", "heading"] = Field(
        description="The selected action for the formalization."
    )
    requirements: list[FormalizedRequirement] = Field(description="List of the output requirements.")
    heading: SplitHeading | None = Field(
        description="Definition of the new heading, required if the action is 'heading'.", default=None
    )


class ActionabilityCheckResult(BaseModel):
    """Result returned by actionability checker."""

    name: str = Field(description="Name of the validator")
    comment: str | None = Field(description="Comment about is_actionable decision.", default=None)
    is_actionable: bool = Field(description="Whether or not the feedback is actionable, given the source requirements.")


class ActionabilityCheckOutput(BaseModel):
    """Output of the validator feedback filter checker."""

    actionability_check_results: list[ActionabilityCheckResult] = Field(
        description="Results of the validation actionability check. None if no feedback was given."
    )


class RequirementContent(BaseModel):
    """Model representing the content of a Requirement."""

    id: str
    requirement_type: str
    is_heading: bool = False
    name: str
    text_content: str


class FormalizedRequirement(BaseModel):
    """Formalized requirement."""

    content: str = Field(description="Textual content of the requirement")
    name: str = Field(description="Name of the requirement")


class SplitHeading(BaseModel):
    """Information about the heading."""

    heading_text_content: str = Field(description="Text of the new heading.")
    heading_name: str = Field(description="Name of the new heading.")


class LLMScoreValidator(BaseModel):
    """Result returned by an LLM scoring validator."""

    comment: str | None = Field(description="Feedback on the requirement on how to improve it.", default=None)
    score: float = Field(description="Score between 0 and 1 indicating the validity of the requirement.", le=1, ge=0)

    @field_validator("score", mode="before")
    @classmethod
    def clip_score_value(cls, value: Any) -> Any:
        """Clip the score value between 0 and 1."""
        if isinstance(value, (int, float)):
            # Clip numeric values directly
            return min(1, max(0, value))
        if isinstance(value, str):
            try:
                num = float(value)
                return min(1, max(0, num))
            except ValueError:
                _logger.debug("Value %s not parsable to float", value)
        return value


class TemplateValidatorOutput(BaseModel):
    """Boolean response model."""

    reason: str | None = Field(default=None, description="Reasoning behind the validation result.")
    valid: bool = Field(description="Whether the requirement is valid.")


HYBRID_MASTER_CHECKER_USER_PROMPT = "Please validate the following requirement:\n\n{text_content}"
LOOP_FEEDBACK_WRAPPER_USER_PROMPT = "The formalization was not successful.\n\n{feedback}\n\nPlease try again."

INCOSE_THRESHOLD = 0.7


# ---------------------------------------------------------------------------
# Jinja2 prompt loader
# ---------------------------------------------------------------------------


def load_prompt(template_name: str, **kwargs: Any) -> str:
    """Load and render a Jinja2 prompt template from the prompts directory."""
    env = Environment(loader=FileSystemLoader(str(PROMPT_DIR)), keep_trailing_newline=True)
    template = env.get_template(template_name)
    return template.render(**kwargs)


# ---------------------------------------------------------------------------
# Individual LLM call helpers
# ---------------------------------------------------------------------------


async def _call_rephrasing_llm(
    client: AsyncOpenAI,
    system_prompt: str,
    user_prompt: str,
    previous_messages: list[dict[str, str]] | None = None,
) -> FormalizationLLMOutput:
    """Call the rephrasing LLM with structured output."""
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    if previous_messages:
        messages.extend(previous_messages)
    else:
        messages.append({"role": "user", "content": user_prompt})

    response = await client.beta.chat.completions.parse(
        model=LLM_MODEL_NAME,
        response_format=FormalizationLLMOutput,
        messages=messages,
    )
    return response.choices[0].message.parsed


async def _run_template_validator(
    client: AsyncOpenAI,
    system_prompt: str,
    requirement_text: str,
) -> TemplateValidatorOutput:
    """Run the LLM-as-a-judge template validator on a single requirement."""
    response = await client.beta.chat.completions.parse(
        model=LLM_MODEL_NAME,
        response_format=TemplateValidatorOutput,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": HYBRID_MASTER_CHECKER_USER_PROMPT.format(text_content=requirement_text)},
        ],
    )
    return response.choices[0].message.parsed


async def _run_incose_validator(
    client: AsyncOpenAI,
    requirement: FormalizedRequirement,
    requirement_type: str,
) -> dict[str, LLMScoreValidator]:
    """Run the INCOSE LLM-as-a-judge validator on a single requirement."""
    incose_prompt = load_prompt(
        "incose_validator_prompt.md.jinja",
        requirement_type=requirement_type,
        requirement=requirement.content,
    )
    # Build a dynamic response model with one LLMScoreValidator field per category
    response_model = create_model(
        "RequirementEvaluation",
        __doc__="Evaluation of a Requirement",
        **{
            cat.value: LLMScoreValidator
            for cat in IsoLLMValidatorCategory
        },
    )
    response = await client.beta.chat.completions.parse(
        model=LLM_MODEL_NAME,
        response_format=response_model,
        messages=[{"role": "user", "content": incose_prompt}],
    )
    parsed = response.choices[0].message.parsed
    return {cat.value: getattr(parsed, cat.value) for cat in IsoLLMValidatorCategory}


async def _run_feedback_filter(
    client: AsyncOpenAI,
    source_requirements: list[RequirementContent],
    requirement: FormalizedRequirement,
    feedback: list[dict[str, str]],
) -> ActionabilityCheckOutput:
    """Filter validator feedback for actionability given the source requirements."""
    filter_prompt = load_prompt(
        "validator_result_filter_prompt.md.jinja",
        source_requirements=source_requirements,
        requirement=requirement,
        feedback=feedback,
    )
    response = await client.beta.chat.completions.parse(
        model=LLM_MODEL_NAME,
        response_format=ActionabilityCheckOutput,
        messages=[{"role": "user", "content": filter_prompt}],
    )
    return response.choices[0].message.parsed


# ---------------------------------------------------------------------------
# Main formalization loop
# ---------------------------------------------------------------------------


async def formalization_loop(
    client: AsyncOpenAI,
    requirement: RequirementContent,
    max_iterations: int = 10,
    incose_threshold: float = INCOSE_THRESHOLD,
) -> FormalizationLLMOutput:
    """Reflective formalization loop as described in Section 3 (Approach).

    1. Rephrase the requirement into MASTeR template syntax via LLM.
    2. Validate with Template Validator (LLM-as-a-judge) and INCOSE Validator.
    3. Filter feedback for actionability given the source requirements.
    4. Re-prompt the LLM with actionable feedback.
    5. Repeat until validators are satisfied or max_iterations is reached.
    """
    # --- Load prompts ----------------------------------------------------------
    system_prompt = load_prompt("formalization_task_prompt.jinja")
    template_validator_system_prompt = load_prompt("master_validator_prompt.md.jinja")
    user_prompt = load_prompt(
        "formalization_input_prompt.jinja",
        target_requirement=requirement,
    )

    # --- Step 1: Initial rephrasing --------------------------------------------
    result = await _call_rephrasing_llm(client, system_prompt, user_prompt)

    if result.action == "no_update":
        return result

    # --- Step 2-5: Reflective validation loop ----------------------------------
    source_requirements = [requirement]
    conversation: list[dict[str, str]] = [
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": result.model_dump_json()},
    ]

    for _iteration in range(max_iterations):
        all_feedback: list[dict[str, str]] = []

        for req in result.requirements:
            # --- Template Validator ---------------------------
            # TODO: Run grammar checks here and pass the result to the llm validator, if they fail.
            template_result = await _run_template_validator(
                client,
                template_validator_system_prompt,
                req.content,
            )
            if not template_result.valid:
                all_feedback.append(
                    {
                        "name": "TemplateValidator",
                        "comment": template_result.reason or "Template syntax violation.",
                    }
                )

            # --- INCOSE Validator ----------------------------------------------
            incose_scores = await _run_incose_validator(
                client,
                req,
                requirement.requirement_type,
            )
            for category, score_result in incose_scores.items():
                if score_result.score < incose_threshold and score_result.comment:
                    all_feedback.append(
                        {
                            "name": category,
                            "comment": score_result.comment,
                        }
                    )

        # No issues found — validators are satisfied
        if not all_feedback:
            break

        # --- Feedback Filter ---------------------------------------------------
        actionability = await _run_feedback_filter(
            client,
            source_requirements,
            result.requirements[0],
            all_feedback,
        )
        actionable_feedback = [
            fb
            for fb, check in zip(all_feedback, actionability.actionability_check_results, strict=False)
            if check.is_actionable
        ]

        # No actionable feedback — stop to avoid forcing hallucination
        if not actionable_feedback:
            break

        # --- Re-prompt with feedback -------------------------------------------
        feedback_text = "\n".join(f"- [{fb['name']}]: {fb['comment']}" for fb in actionable_feedback)
        conversation.append(
            {
                "role": "user",
                "content": LOOP_FEEDBACK_WRAPPER_USER_PROMPT.format(feedback=feedback_text),
            }
        )

        result = await _call_rephrasing_llm(
            client,
            system_prompt,
            user_prompt,
            previous_messages=conversation,
        )
        conversation.append({"role": "assistant", "content": result.model_dump_json()})

    return result

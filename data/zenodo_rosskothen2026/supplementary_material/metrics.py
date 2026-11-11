"""Evaluation metrics for comparing LLM-generated MASTeR formalizations against human-rephrased ground truth.

Metrics implemented:
- MasterTemplateMetric: Template classification F1 (CSV columns: SophistTemplateF1, SophistConditionF1,
  SophistConditionWithTemplateF1)
- EmbeddingMoversDistanceMetric: Earth Mover's Distance over embeddings (CSV column: EmbeddingMoversDistance)
"""

from __future__ import annotations

from collections import Counter
from enum import StrEnum
from pathlib import Path
from scipy.stats import wasserstein_distance_nd
from openai import AsyncOpenAI
from pydantic import BaseModel

_PROMPT_DIR = Path(__file__).parent
LLM_MODEL_NAME = "TODO"
EMBEDDING_MODEL_NAME = "TODO"

class MetricScore(BaseModel):
    """The score object of a metric."""

    name: str
    value: float | str
    comment: str | None = None


class MasterTemplatePattern(StrEnum):
    """MasterTemplatePattern."""

    FunctionalIndependent = "FunctionMASTeR - Independent System Activity"
    FunctionalUser = "FunctionMASTeR - User Interaction"
    FunctionalInterface = "FunctionMASTeR - Interface requirement"
    Property = "PropertyMASTeR"
    Environment = "EnvironmentMASTeR"
    Process = "ProcessMASTeR"


class MasterCondition(StrEnum):
    """MasterCondition."""

    Logic = "LogicMASTeR"
    Event = "EventMASTeR"
    Time = "TimeMASTeR"
    No = "No Condition"


class MasterTemplateClassification(BaseModel):
    """Classification of a Master Template."""

    main_template: MasterTemplatePattern
    condition_template: MasterCondition


MASTER_SYSTEM_PROMPT = (_PROMPT_DIR / "prompts" / "master_format_prompt.md").read_text()

MASTER_TEMPLATE_CHECKER_PROMPT = """Your Job is to classify the following Requirement.
Determine which template is used and which condition is used, if any.

The output format is a the following JSON scheme:
{output_scheme}

The Requirement to classify is: {req}"""


class MasterTemplateMetric:
    """MasterTemplateMetric.

    Checks if the candidate uses the same templates as the base. Either with string match or with LLM as a judge.
    Calculates Precision, Recall and F1 for the template, the conditions and the combination of both.
    """

    def __init__(self, llm: AsyncOpenAI) -> None:
        """Init."""
        self.llm = llm
        self.system_prompt = MASTER_SYSTEM_PROMPT

    async def __call__(self, base_req_texts: list[str], candidate_req_texts: list[str]) -> list[MetricScore]:
        """Calculate Precision, Recall and F1."""
        base_req_classifications = [await self._get_master_template(req) for req in base_req_texts]
        candidate_req_classifications = [await self._get_master_template(req) for req in candidate_req_texts]
        main_f1 = self._get_f1_score(
            [str(v[0]) for v in base_req_classifications], [str(v[0]) for v in candidate_req_classifications]
        )
        condition_f1 = self._get_f1_score(
            [str(v[1]) for v in base_req_classifications], [str(v[1]) for v in candidate_req_classifications]
        )
        combi_f1 = self._get_f1_score(
            [str(v[0]) + str(v[1]) for v in base_req_classifications],
            [str(v[0]) + str(v[1]) for v in candidate_req_classifications],
        )

        return [
            MetricScore(
                name="MasterTemplateF1",
                value=main_f1,
            ),
            MetricScore(
                name="MasterConditionF1",
                value=condition_f1,
            ),
            MetricScore(
                name="MasterConditionWithTemplateF1",
                value=combi_f1,
            ),
        ]

    @staticmethod
    def _get_f1_score(base_reqs: list[str], candidate_reqs: list[str]) -> float:
        """Calculate f1 score.

        Reimplemented, because i couldn't find an implementation, which ignores order and counts multi labels.
        """
        base_req_counter = Counter(base_reqs)
        candidate_req_counter = Counter(candidate_reqs)
        sum_equal_classifications = sum(
            min(base_req_counter[label], candidate_req_counter[label]) for label in candidate_req_counter
        )
        if sum_equal_classifications == 0:
            return 0
        precision = sum_equal_classifications / sum(base_req_counter.values())
        recall = sum_equal_classifications / sum(candidate_req_counter.values())
        return 2 * precision * recall / (precision + recall)

    async def _get_master_template(self, req: str) -> tuple[MasterTemplatePattern, MasterCondition]:
        """Get master template for a requirement."""
        master_main, master_addon = self._get_master_template_by_string_match(req)
        # FunctionalIndependent, Process and Property cant be reliably determined by string matching
        return (master_main, master_addon) if master_main is not None else await self._get_master_template_by_llm(req)

    def _get_master_template_by_string_match(self, req: str) -> tuple[MasterTemplatePattern | None, MasterCondition]:
        master_main = None
        if "provide" in req and "with the ability to" in req:
            master_main = MasterTemplatePattern.FunctionalUser
        elif "be able to" in req:
            master_main = MasterTemplatePattern.FunctionalInterface
        elif "be designed in a way" in req and "can be operated" in req:
            master_main = MasterTemplatePattern.Environment

        if req.startswith("As soon as"):
            master_addon = MasterCondition.Event
        elif req.startswith("As long as"):
            master_addon = MasterCondition.Time
        elif req.startswith("If"):
            master_addon = MasterCondition.Logic
        else:
            master_addon = MasterCondition.No
        return (master_main, master_addon)

    async def _get_master_template_by_llm(self, req: str) -> tuple[MasterTemplatePattern, MasterCondition]:
        llm_classification = await self.llm.beta.chat.completions.parse(
            model=LLM_MODEL_NAME,
            response_format=MasterTemplateClassification,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": MASTER_TEMPLATE_CHECKER_PROMPT.format(
                        req=req, output_scheme=str(MasterTemplateClassification.model_json_schema())
                    ),
                },
            ],
        )
        return (llm_classification.main_template, llm_classification.condition_template)

class EmbeddingMoversDistanceMetric:
    """Embedding Movers Distance.

    This Metric is similar to the Word Mover's Distance.
    But instead of moving relevant words, we move the whole requirements.
    This is done by computing the embeddings of each requirement and apply Earth Mover's Distance to them.
    The distance between two requirements is the vector distance in the embedding space.
    https://en.wikipedia.org/wiki/Earth_mover%27s_distance
    """

    def __init__(self, llm: AsyncOpenAI) -> None:
        """Init."""
        self.llm = llm

    async def __call__(self, base_req_texts: list[str], candidate_req_texts: list[str]) -> list[MetricScore]:
        """Compute the metric between two formalizations of the same base requirement."""
        base_embeddings = [
            x.embedding
            for x in (
                await self.llm.embeddings.create(
                    model=EMBEDDING_MODEL_NAME,
                    input=base_req_texts,
                    encoding_format="float",
                )
            ).data
        ]
        candidate_embeddings = [
            x.embedding
            for x in (
                await self.llm.embeddings.create(
                    model=EMBEDDING_MODEL_NAME,
                    input=candidate_req_texts,
                    encoding_format="float",
                )
            ).data
        ]
        distance = wasserstein_distance_nd(base_embeddings, candidate_embeddings)
        value = 1 / (1 + distance)  # distance: 0.0 -> similarity 1.0, distance: inf -> similarity 0.0
        return [MetricScore(name="EmbeddingMoversDistance", value=value)]
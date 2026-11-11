"""Carregamento dos datasets oficiais de Großer et al. / Roßkothen et al. (RE 2026)."""

from __future__ import annotations

import ast
import csv
import re
from dataclasses import dataclass
from pathlib import Path

from req_jev.prompts_loader import REPO_ROOT
from req_jev.schemas import RequirementContent

DEFAULT_ZENODO_DIR = REPO_ROOT / "data" / "zenodo_rosskothen2026" / "supplementary_material"


@dataclass(frozen=True)
class SourceDatasetItem:
    """Item individual do conjunto de 249 requisitos fonte e ground truth humano."""

    req_name: str
    project_set: str
    source_length: int
    split_factor: int
    human_survey_ratings: list[str]
    source_text: str
    target_master_requirements: list[str]

    def to_requirement_content(self) -> RequirementContent:
        return RequirementContent(
            id=self.req_name,
            requirement_type="Requirement",
            is_heading=False,
            name=self.req_name,
            text_content=self.source_text,
        )


def infer_project_set(req_name: str) -> str:
    """Infere o subconjunto de requisitos (FLEX, ECSS, CSE, TSS, EVS) pelo prefixo do ID."""
    prefix = re.match(r"^[A-Za-z_]+", req_name)
    raw = prefix.group(0).upper() if prefix else "UNKNOWN"
    for canonical in ("FLEX", "ECSS", "CSE", "CS_E", "TSS", "EVS"):
        if raw.startswith(canonical):
            return "CSE" if canonical == "CS_E" else canonical
    return raw


def _parse_list_literal(raw: str) -> list[str]:
    if not raw or not raw.strip():
        return []
    parsed = ast.literal_eval(raw)
    if isinstance(parsed, list):
        return [str(x) for x in parsed]
    return [str(parsed)]


def load_source_dataset(zenodo_dir: Path = DEFAULT_ZENODO_DIR) -> list[SourceDatasetItem]:
    """Carrega os 249 requisitos fonte e as 410 reescritas humanas MASTeR."""
    csv_path = zenodo_dir / "survey_solution_data.csv"
    items: list[SourceDatasetItem] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            req_name = row["req_name"].strip()
            items.append(
                SourceDatasetItem(
                    req_name=req_name,
                    project_set=infer_project_set(req_name),
                    source_length=int(row["source_length"]),
                    split_factor=int(row["split_factor"]),
                    human_survey_ratings=_parse_list_literal(row.get("survey_ratings", "")),
                    source_text=row["source_requirements"].strip(),
                    target_master_requirements=_parse_list_literal(row["target_requirements"]),
                )
            )
    return items

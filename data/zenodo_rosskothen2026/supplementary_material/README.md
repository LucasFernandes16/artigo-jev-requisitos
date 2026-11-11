# Supplementary Material

**Paper:** *On Converting Natural Language Requirements into Semi-Formal Templates Using LLMs*

**Conference:** 2026 34th IEEE International Requirements Engineering Conference (RE)

This supplementary material contains the data used in the evaluation of our LLM-based approach for converting natural language requirements into the MASTeR semi-formal template syntax.

If you use this artifact in your research, please cite:

```
@inproceedings{rosskothen_2026_convertReqsToMaster,
  title        = {On Converting Natural Language Requirements into Semi-Formal Templates Using LLMs},
  author       = {Roßkothen, Julian and Fuchß, Dominik and Erdösi, Florian and Floruß, Maria and Keim, Jan and Hey, Tobias},
  booktitle    = {2026 34th IEEE International Requirements Engineering Conference (RE)},
  year         = {2026},
  organization = {IEEE}
}
```
## Contents

| File | Description |
|------|-------------|
| `rephrased_reqs_data.csv` | LLM-rephrased requirements with all computed metric scores and survey ratings for each configuration |
| `survey_solution_data.csv` | Source requirements with human-rephrased ground truth (MASTeR) and survey ratings |
| `limesurvey_survey_export.lss` | LimeSurvey export file containing the complete survey structure and questions |
| `metrics.py` | Implementation of the MASTeR Template Metric and Embedding Mover's Distance metric used for evaluation |
| `prompts/` | Jinja2 prompt templates and reference implementation of the reflective formalization loop |

## Dataset Origin

The source requirements and human-rephrased ground truth originate from Großer et al. [[1]](#references). 

## File Descriptions

### `survey_solution_data.csv`

Contains the 260 source requirements together with their human-rephrased MASTeR ground truth and aggregated survey ratings (261 rows including header).

| Column | Description |
|--------|-------------|
| `req_name` | Unique requirement identifier (e.g., `FLEX9`, `EVS1`) |
| `source_length` | Word count of the original source requirement |
| `split_factor` | Number of MASTeR requirements the source was split into by human rephrasers |
| `survey_ratings` | List of survey category ratings assigned by evaluators (C1–C4) |
| `source_requirements` | Original natural language requirement text |
| `target_requirements` | List of human-rephrased MASTeR requirement(s) |

### `rephrased_reqs_data.csv`

Contains the LLM-rephrased requirements for all four configurations, together with all computed metric scores and survey ratings (997 rows including header; 249 requirements × 4 configurations).

| Column | Description |
|--------|-------------|
| `run_name` | Configuration identifier (see mapping below) |
| `req_name` | Unique requirement identifier |
| `source_length` | Word count of the original source requirement |
| `split_factor` | Ground-truth split factor from Großer et al. |
| `survey_ratings` | List of survey category ratings (C1–C4), if rated |
| `requirements` | List of LLM-generated MASTeR requirement(s) |
| `QEmbeddingDistance` | Embedding distance using a query-optimized model |
| `BLEUScore` | BLEU score (sentence-level, averaged) |
| `SentBERTRecall` | Sentence-level BERT Recall |
| `SentBERTF1` | Sentence-level BERT F₁ |
| `SentBERTPrecision` | Sentence-level BERT Precision |
| `CorpBERTRecall` | Source-level (corpus) BERT Recall |
| `CorpBERTF1` | Source-level (corpus) BERT F₁ |
| `CorpBERTPrecision` | Source-level (corpus) BERT Precision |
| `SentenceCosineSimilarity` | Sentence-level cosine similarity (embedding-based) |
| `SentDotProduct` | Sentence-level dot product (embedding-based) |
| `SentEmbedDistance` | Sentence-level embedding distance |
| `CorpCosineSimilarity` | Source-level cosine similarity (embedding-based) |
| `CorpDotProduct` | Source-level dot product (embedding-based) |
| `EmbeddingDistance` | Source-level embedding distance |
| `EmbeddingMoversDistance` | Embedding Mover's Distance (EMD) |
| `SophistTemplateF1` | MASTeR main template F₁ score |
| `SophistConditionF1` | MASTeR condition template F₁ score |
| `SophistConditionWithTemplateF1` | MASTeR combined (main + condition) F₁ score |
| `best` | Best compound metric score (weighted combination) |

All metric scores are normalized to the range [0, 1], where 1 indicates a perfect match with the human-rephrased ground truth.

### `limesurvey_survey_export.lss`

Contains the complete LimeSurvey survey structure used to evaluate the quality of LLM-generated MASTeR formalizations. The survey presented participants with the original natural language requirement and multiple LLM-generated formalization options from different configurations. Participants ranked these options into four quality categories (C1–C4) based on how useful they would be as a starting point for creating the final MASTeR formalization.


### `metrics.py`

Contains the implementation of two evaluation metrics used to compare LLM-generated MASTeR formalizations against human-rephrased ground truth:

- **`MasterTemplateMetric`**: Classifies requirements by MASTeR template type (main template and condition) using string matching with LLM fallback, then computes Precision, Recall, and F₁ scores. Corresponds to the `SophistTemplateF1`, `SophistConditionF1`, and `SophistConditionWithTemplateF1` columns in the CSV.
- **`EmbeddingMoversDistanceMetric`**: Computes the Earth Mover's Distance between embeddings of base and candidate requirements, normalized to [0, 1]. Corresponds to the `EmbeddingMoversDistance` column in the CSV.

### `prompts/`

Contains the Jinja2 prompt templates and a reference implementation of the approach:

| File | Description |
|------|-------------|
| `master_format_prompt.md` | System prompt defining the MASTeR/SOPHIST template syntax and quality criteria |
| `formalization_task_prompt.jinja` | Task prompt instructing the LLM to convert requirements into MASTeR syntax |
| `formalization_input_prompt.jinja` | Input prompt template for individual requirements |
| `master_validator_prompt.md.jinja` | Prompt for the LLM-as-a-judge template validator |
| `IsoValidatorPrompt.md.jinja` | Prompt for the INCOSE/ISO quality criteria validator |
| `validator_result_filter_prompt.md.jinja` | Prompt for filtering validator feedback by actionability |
| `simple_loop_implementation.py` | Reference Python implementation of the reflective formalization loop (Section 3) |

## Configuration Mapping

| `run_name` in CSV | Paper Name | Description |
|-------------------|------------|-------------|
| `full-qwen235` | JudgeQwen | Qwen3-235B-A22B-Instruct-2507-AWQ with GPT-OSS-120B LLM-as-a-judge validators |
| `minimal-qwen235` | Qwen | Qwen3-235B-A22B-Instruct-2507-AWQ without LLM-as-a-judge validators |
| `minimal-qwen3-next` | QwenThinking | Qwen3-Next-80B-A3B-Thinking without LLM-as-a-judge validators |
| `full-gpt-5` | JudgeGPT | GPT-5 with GPT-5 LLM-as-a-judge validators |

## Survey Categories

Survey participants categorized each rephrased requirement by usefulness:

| Category | Meaning |
|----------|---------|
| **C1** | No update needed — acceptable as the MASTeR version |
| **C2** | Only small changes necessary — fixable within seconds |
| **C3** | Useful draft — better starting point than the original requirement |
| **C4** | No help — better to rephrase from scratch |

## References

[1] Großer, K., et al. "A Comparative Study of Requirements Rephrasing into Semi-formal Syntax Templates." 2023. Dataset available at: https://doi.org/10.5281/zenodo.13343495

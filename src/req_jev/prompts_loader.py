"""Carregamento dos templates Jinja2 oficiais e tabela de precificação por milhão de tokens."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPT_DIR = REPO_ROOT / "prompts"

# Tabela de preços de referência em USD por 1.000.000 de tokens (input_usd, output_usd).
# Pode ser ajustada conforme a tabela vigente do provedor no momento da execução.
MODEL_PRICING_PER_1M: dict[str, tuple[float, float]] = {
    # Google Gemini API (Paid tier padrão de referência para comparação justa de custo)
    "gemini-3.5-flash": (0.30, 2.50),
    "gemini-3.8-flash": (0.30, 2.50),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.5-pro": (1.25, 10.00),
    # Modelos do artigo via OpenRouter / provedores compatíveis
    "qwen/qwen3-235b-a22b-2507": (0.20, 0.60),
    "qwen/qwen3-next-80b-a3b-thinking": (0.15, 0.60),
    "openai/gpt-oss-120b": (0.15, 0.60),
    "openai/gpt-5": (1.25, 10.00),
    # TypeSafe AI JEV System One (inferência discriminativa sem geração de tokens de prosa)
    "jev-latest": (0.10, 0.00),
    "jev-mock": (0.00, 0.00),
}


def estimate_call_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estima o custo em USD de uma chamada com base no modelo e contagem de tokens."""
    normalized = model.strip()
    prices = MODEL_PRICING_PER_1M.get(normalized)
    if prices is None:
        # Busca parcial caso o modelo contenha sufixo de versão/provedor
        for key, val in MODEL_PRICING_PER_1M.items():
            if key in normalized or normalized in key:
                prices = val
                break
    if prices is None:
        prices = (0.25, 1.00)
    in_price, out_price = prices
    return (input_tokens * in_price + output_tokens * out_price) / 1_000_000.0


def load_prompt(template_name: str, prompt_dir: Path = PROMPT_DIR, **kwargs: Any) -> str:
    """Carrega e renderiza um prompt Jinja2 do diretório oficial `prompts/`."""
    env = Environment(loader=FileSystemLoader(str(prompt_dir)), keep_trailing_newline=True)
    template = env.get_template(template_name)
    return template.render(**kwargs)

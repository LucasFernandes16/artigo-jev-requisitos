"""Backends assíncronos unificados para geração e julgamento estruturado (Gemini e OpenRouter/OpenAI)."""

from __future__ import annotations

import json
import os
import time
from typing import Literal, Protocol, TypeVar

from google import genai
from google.genai import types as genai_types
from openai import AsyncOpenAI
from pydantic import BaseModel

from req_jev.prompts_loader import estimate_call_cost_usd
from req_jev.schemas import CallTelemetry

SchemaT = TypeVar("SchemaT", bound=BaseModel)

StageName = Literal[
    "generator",
    "llm_template_validator",
    "llm_incose_validator",
    "llm_feedback_filter",
]


class StructuredLLMBackend(Protocol):
    """Protocolo agnóstico de provedor para geração com JSON Schema estrito + telemetria."""

    provider_name: Literal["gemini", "openrouter"]

    async def generate_structured(
        self,
        *,
        stage: StageName,
        model: str,
        system_prompt: str | None,
        messages: list[dict[str, str]],
        response_schema: type[SchemaT],
        temperature: float = 0.0,
    ) -> tuple[SchemaT, CallTelemetry]:
        ...


class GeminiStructuredBackend:
    """Adaptador assíncrono oficial usando `google-genai` com Structured Outputs."""

    provider_name: Literal["gemini", "openrouter"] = "gemini"

    def __init__(self, api_key: str | None = None) -> None:
        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise ValueError("Defina GEMINI_API_KEY (ou GOOGLE_API_KEY) no arquivo .env.")
        self._client = genai.Client(api_key=key)

    async def generate_structured(
        self,
        *,
        stage: StageName,
        model: str,
        system_prompt: str | None,
        messages: list[dict[str, str]],
        response_schema: type[SchemaT],
        temperature: float = 0.0,
    ) -> tuple[SchemaT, CallTelemetry]:
        contents: list[genai_types.Content] = []
        for msg in messages:
            role = "model" if msg["role"] == "assistant" else "user"
            contents.append(
                genai_types.Content(
                    role=role,
                    parts=[genai_types.Part.from_text(text=msg["content"])],
                )
            )

        config = genai_types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="application/json",
            response_schema=response_schema,
            temperature=temperature,
        )

        t0 = time.perf_counter()
        resp = await self._client.aio.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        usage = resp.usage_metadata
        in_tok = int(getattr(usage, "prompt_token_count", 0) or 0)
        out_tok = int(getattr(usage, "candidates_token_count", 0) or 0)

        if resp.parsed is not None and isinstance(resp.parsed, response_schema):
            parsed_obj = resp.parsed
        else:
            parsed_obj = response_schema.model_validate_json(resp.text or "{}")

        telemetry = CallTelemetry(
            stage=stage,
            model=model,
            input_tokens=in_tok,
            output_tokens=out_tok,
            latency_ms=latency_ms,
            estimated_cost_usd=estimate_call_cost_usd(model, in_tok, out_tok),
        )
        return parsed_obj, telemetry


class OpenRouterStructuredBackend:
    """Adaptador assíncrono compatível com OpenRouter / OpenAI (suporta Qwen3-235B e GPT-OSS-120B)."""

    provider_name: Literal["gemini", "openrouter"] = "openrouter"

    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        key = api_key or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("Defina OPENROUTER_API_KEY (ou OPENAI_API_KEY) no arquivo .env.")
        resolved_url = base_url or os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        self._client = AsyncOpenAI(api_key=key, base_url=resolved_url)

    async def generate_structured(
        self,
        *,
        stage: StageName,
        model: str,
        system_prompt: str | None,
        messages: list[dict[str, str]],
        response_schema: type[SchemaT],
        temperature: float = 0.0,
    ) -> tuple[SchemaT, CallTelemetry]:
        chat_messages: list[dict[str, str]] = []
        if system_prompt:
            chat_messages.append({"role": "system", "content": system_prompt})
        chat_messages.extend(messages)

        t0 = time.perf_counter()
        try:
            resp = await self._client.beta.chat.completions.parse(
                model=model,
                messages=chat_messages,  # type: ignore[arg-type]
                response_format=response_schema,
                temperature=temperature,
            )
            latency_ms = (time.perf_counter() - t0) * 1000.0
            parsed_obj = resp.choices[0].message.parsed
            if parsed_obj is None:
                raw_text = resp.choices[0].message.content or "{}"
                parsed_obj = response_schema.model_validate_json(_strip_code_fences(raw_text))
        except Exception:
            # Fallback para provedores OpenRouter que exigem json_object explícito no prompt
            schema_json = json.dumps(response_schema.model_json_schema(), ensure_ascii=False)
            augmented = list(chat_messages)
            augmented[-1] = {
                "role": augmented[-1]["role"],
                "content": (
                    f"{augmented[-1]['content']}\n\n"
                    f"Respond strictly with valid JSON matching this JSON Schema:\n{schema_json}"
                ),
            }
            t0 = time.perf_counter()
            resp = await self._client.chat.completions.create(
                model=model,
                messages=augmented,  # type: ignore[arg-type]
                response_format={"type": "json_object"},
                temperature=temperature,
            )
            latency_ms = (time.perf_counter() - t0) * 1000.0
            raw_text = resp.choices[0].message.content or "{}"
            parsed_obj = response_schema.model_validate_json(_strip_code_fences(raw_text))

        usage = resp.usage
        in_tok = int(getattr(usage, "prompt_tokens", 0) or 0)
        out_tok = int(getattr(usage, "completion_tokens", 0) or 0)
        telemetry = CallTelemetry(
            stage=stage,
            model=model,
            input_tokens=in_tok,
            output_tokens=out_tok,
            latency_ms=latency_ms,
            estimated_cost_usd=estimate_call_cost_usd(model, in_tok, out_tok),
        )
        return parsed_obj, telemetry


def _strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return cleaned


def build_llm_backend(provider: Literal["gemini", "openrouter"]) -> StructuredLLMBackend:
    """Instancia o backend correspondente ao provedor selecionado."""
    if provider == "gemini":
        return GeminiStructuredBackend()
    if provider == "openrouter":
        return OpenRouterStructuredBackend()
    raise ValueError(f"Provedor não suportado: {provider}")

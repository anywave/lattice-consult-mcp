"""OpenAI provider (GPT-5, GPT-4, o1, o1-mini)."""
import time
from typing import Optional
from openai import AsyncOpenAI
from .base import LatticeProvider, ProviderConfig, SingleResponse, env_config


class OpenAIProvider(LatticeProvider):
    name = "openai"
    default_model = "gpt-4o"  # fast + cheap default; o1/gpt-5 for harder calls

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        if config.available:
            self._client = AsyncOpenAI(api_key=config.api_key)
        else:
            self._client = None

    async def consult(self, system, prompt, model=None, max_tokens=2048, timeout_s=60.0) -> SingleResponse:
        if not self._client:
            return SingleResponse(
                provider="openai",
                model=model or self.default_model,
                response="",
                error=self.config.skip_reason or "openai not configured",
            )
        m = model or self.default_model
        t0 = time.monotonic()
        try:
            # o1-series models do NOT support system messages or max_tokens (use max_completion_tokens)
            is_reasoning = m.startswith("o1") or m.startswith("o3") or m.startswith("o5")
            if is_reasoning:
                # o-series: roll system into the user message; use max_completion_tokens
                merged = f"{system}\n\n{prompt}" if system else prompt
                resp = await self._client.chat.completions.create(
                    model=m,
                    messages=[{"role": "user", "content": merged}],
                    max_completion_tokens=max_tokens,
                    timeout=timeout_s,
                )
            else:
                messages = []
                if system:
                    messages.append({"role": "system", "content": system})
                messages.append({"role": "user", "content": prompt})
                resp = await self._client.chat.completions.create(
                    model=m,
                    messages=messages,
                    max_tokens=max_tokens,
                    timeout=timeout_s,
                )
            text = resp.choices[0].message.content or ""
            usage = resp.usage
            tin = usage.prompt_tokens if usage else 0
            tout = usage.completion_tokens if usage else 0
            cost = self.estimate_cost(tin, tout)
            return SingleResponse(
                provider="openai",
                model=m,
                response=text,
                tokens_in=tin,
                tokens_out=tout,
                cost_usd=cost,
                latency_ms=int((time.monotonic() - t0) * 1000),
            )
        except Exception as e:
            return SingleResponse(
                provider="openai",
                model=m,
                response="",
                error=f"{type(e).__name__}: {e}",
                latency_ms=int((time.monotonic() - t0) * 1000),
            )


def build_config() -> ProviderConfig:
    # GPT-4o pricing as of 2026: $2.50/M input, $10/M output
    return env_config(
        "openai",
        "LATTICE_OPENAI_API_KEY",
        "gpt-4o",
        cost_in_per_mtok=2.50,
        cost_out_per_mtok=10.0,
    )

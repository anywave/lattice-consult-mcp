"""xAI Grok provider (grok-4, grok-4-fast, grok-2). OpenAI-compatible API at api.x.ai."""
import time
from typing import Optional
from openai import AsyncOpenAI  # reuse openai SDK with custom base_url
from .base import LatticeProvider, ProviderConfig, SingleResponse, env_config


class XaiProvider(LatticeProvider):
    name = "xai"
    default_model = "grok-4-fast"  # fast + cheap default; grok-4 for harder calls

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        if config.available:
            self._client = AsyncOpenAI(
                api_key=config.api_key,
                base_url=config.base_url or "https://api.x.ai/v1",
            )
        else:
            self._client = None

    async def consult(self, system, prompt, model=None, max_tokens=2048, timeout_s=60.0) -> SingleResponse:
        if not self._client:
            return SingleResponse(provider="xai", model=model or self.default_model, response="", error=self.config.skip_reason or "xai not configured")
        m = model or self.default_model
        t0 = time.monotonic()
        try:
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
                provider="xai", model=m, response=text,
                tokens_in=tin, tokens_out=tout, cost_usd=cost,
                latency_ms=int((time.monotonic() - t0) * 1000),
            )
        except Exception as e:
            return SingleResponse(provider="xai", model=m, response="", error=f"{type(e).__name__}: {e}", latency_ms=int((time.monotonic() - t0) * 1000))


def build_config() -> ProviderConfig:
    # Grok-4 pricing as of 2026: ~$3/M input, ~$15/M output (grok-4-fast cheaper)
    cfg = env_config(
        "xai", "LATTICE_XAI_API_KEY", "grok-4-fast",
        cost_in_per_mtok=3.0, cost_out_per_mtok=15.0,
        base_url_env="LATTICE_XAI_BASE_URL", base_url_default="https://api.x.ai/v1",
    )
    return cfg

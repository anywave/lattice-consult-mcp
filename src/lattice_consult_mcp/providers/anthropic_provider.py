"""Anthropic provider (Claude Opus 4.7, Sonnet 4.6, Haiku 4.5)."""
import time
from typing import Optional
from anthropic import AsyncAnthropic
from .base import LatticeProvider, ProviderConfig, SingleResponse, env_config


class AnthropicProvider(LatticeProvider):
    name = "anthropic"
    default_model = "claude-sonnet-4-6"  # fast + cheap default; opus-4-7 for harder calls

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        if config.available:
            self._client = AsyncAnthropic(api_key=config.api_key)
        else:
            self._client = None

    async def consult(self, system, prompt, model=None, max_tokens=2048, timeout_s=60.0) -> SingleResponse:
        if not self._client:
            return SingleResponse(
                provider="anthropic",
                model=model or self.default_model,
                response="",
                error=self.config.skip_reason or "anthropic not configured",
            )
        m = model or self.default_model
        t0 = time.monotonic()
        try:
            kwargs = dict(
                model=m,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
                timeout=timeout_s,
            )
            if system:
                kwargs["system"] = system
            resp = await self._client.messages.create(**kwargs)
            text_parts = [block.text for block in resp.content if hasattr(block, "text")]
            text = "".join(text_parts)
            usage = resp.usage
            tin = usage.input_tokens if usage else 0
            tout = usage.output_tokens if usage else 0
            cost = self.estimate_cost(tin, tout)
            return SingleResponse(
                provider="anthropic",
                model=m,
                response=text,
                tokens_in=tin,
                tokens_out=tout,
                cost_usd=cost,
                latency_ms=int((time.monotonic() - t0) * 1000),
            )
        except Exception as e:
            return SingleResponse(
                provider="anthropic",
                model=m,
                response="",
                error=f"{type(e).__name__}: {e}",
                latency_ms=int((time.monotonic() - t0) * 1000),
            )


def build_config() -> ProviderConfig:
    # Sonnet 4.6 pricing as of 2026: $3/M input, $15/M output
    return env_config(
        "anthropic",
        "LATTICE_ANTHROPIC_API_KEY",
        "claude-sonnet-4-6",
        cost_in_per_mtok=3.0,
        cost_out_per_mtok=15.0,
    )

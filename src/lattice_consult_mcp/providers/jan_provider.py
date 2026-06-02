"""Jan local provider. OpenAI-compatible API at localhost:1337 default.

Zero-cost (user GPU), no rate limits, supports multi-model angularity via the
`model` param -- a single Jan instance hosts multiple angles by model swap
(e.g., qwen2.5:7b vs llama3.1:70b count as two distinct angles in ensemble).
"""
import time
import os
import httpx
from typing import Optional
from openai import AsyncOpenAI  # reuse openai SDK with custom base_url
from .base import LatticeProvider, ProviderConfig, SingleResponse, env_config


class JanProvider(LatticeProvider):
    name = "jan"
    default_model = "qwen2.5:7b-instruct"  # operator overrides via env

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        if config.base_url:
            # No api key needed for local; openai SDK requires SOME string here
            self._client = AsyncOpenAI(
                api_key="not-needed-for-jan",
                base_url=config.base_url,
            )
        else:
            self._client = None

    async def consult(self, system, prompt, model=None, max_tokens=2048, timeout_s=120.0) -> SingleResponse:
        m = model or self.default_model
        if not self._client:
            return SingleResponse(provider="jan", model=m, response="", error="jan base_url not configured")
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
            # Jan is local -> $0
            return SingleResponse(
                provider="jan", model=m, response=text,
                tokens_in=tin, tokens_out=tout, cost_usd=0.0,
                latency_ms=int((time.monotonic() - t0) * 1000),
            )
        except Exception as e:
            return SingleResponse(provider="jan", model=m, response="", error=f"{type(e).__name__}: {e}", latency_ms=int((time.monotonic() - t0) * 1000))


def build_config() -> ProviderConfig:
    """Build Jan config; health-check the endpoint, mark available accordingly."""
    base_url = os.environ.get("LATTICE_JAN_BASE_URL", "http://localhost:1337/v1")
    default_model = os.environ.get("LATTICE_JAN_DEFAULT_MODEL", "qwen2.5:7b-instruct")
    config = ProviderConfig(
        name="jan",
        api_key=None,  # not required
        base_url=base_url,
        default_model=default_model,
        cost_in_per_mtok=0.0,
        cost_out_per_mtok=0.0,
    )
    # Sync health-check at config-build time so list_providers() reports honestly
    try:
        r = httpx.get(f"{base_url.rstrip('/')}/models", timeout=2.0)
        if r.status_code == 200:
            config.available = True
        else:
            config.available = False
            config.skip_reason = f"jan /models returned {r.status_code}"
    except Exception as e:
        config.available = False
        config.skip_reason = f"jan endpoint unreachable: {type(e).__name__}"
    return config

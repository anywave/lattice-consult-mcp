"""Google Gemini provider (Gemini 2.5 Pro, Gemini 2.5 Flash). 1M context."""
import time
import asyncio
from typing import Optional
import google.generativeai as genai
from .base import LatticeProvider, ProviderConfig, SingleResponse, env_config


class GeminiProvider(LatticeProvider):
    name = "gemini"
    default_model = "gemini-2.5-pro"  # 1M context, primary angle

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        if config.available:
            genai.configure(api_key=config.api_key)
        self._configured = config.available

    async def consult(self, system, prompt, model=None, max_tokens=2048, timeout_s=60.0) -> SingleResponse:
        m = model or self.default_model
        if not self._configured:
            return SingleResponse(provider="gemini", model=m, response="", error=self.config.skip_reason or "gemini not configured")
        t0 = time.monotonic()
        try:
            # google-generativeai is sync; run in executor to keep async surface
            gm = genai.GenerativeModel(
                model_name=m,
                system_instruction=system if system else None,
                generation_config={"max_output_tokens": max_tokens},
            )
            # blocking call -> executor
            loop = asyncio.get_running_loop()
            resp = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: gm.generate_content(prompt)),
                timeout=timeout_s,
            )
            # Extract text + usage
            text = resp.text if hasattr(resp, "text") else ""
            tin = 0
            tout = 0
            if hasattr(resp, "usage_metadata") and resp.usage_metadata:
                tin = getattr(resp.usage_metadata, "prompt_token_count", 0) or 0
                tout = getattr(resp.usage_metadata, "candidates_token_count", 0) or 0
            cost = self.estimate_cost(tin, tout)
            return SingleResponse(
                provider="gemini", model=m, response=text,
                tokens_in=tin, tokens_out=tout, cost_usd=cost,
                latency_ms=int((time.monotonic() - t0) * 1000),
            )
        except Exception as e:
            return SingleResponse(provider="gemini", model=m, response="", error=f"{type(e).__name__}: {e}", latency_ms=int((time.monotonic() - t0) * 1000))


def build_config() -> ProviderConfig:
    # Gemini 2.5 Pro pricing as of 2026: ~$1.25/M input, ~$5/M output
    return env_config("gemini", "LATTICE_GEMINI_API_KEY", "gemini-2.5-pro", cost_in_per_mtok=1.25, cost_out_per_mtok=5.0)

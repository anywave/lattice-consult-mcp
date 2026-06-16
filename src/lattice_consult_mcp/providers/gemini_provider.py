"""Google Gemini provider (Gemini 2.5 Pro, Gemini Flash). 1M context.

Uses the REST API directly via httpx rather than the google-generativeai SDK
because:

1. google-generativeai is officially deprecated as of 2026 (FutureWarning urges
   migration to google.genai).
2. Both SDKs use gRPC by default, which does NOT respect the Windows system
   cert store the same way httpx does. On machines with corporate AV (e.g.,
   Avast root CA injection) gRPC fails with CERTIFICATE_VERIFY_FAILED while
   httpx + REST works because Python's `ssl` module loads CAs from the OS
   store.
3. The REST endpoint is what Google documents publicly and is the most stable
   surface. Drops a dependency (google-generativeai + its grpcio chain).
"""
import time
import os
from typing import Optional

import httpx

from .base import LatticeProvider, ProviderConfig, SingleResponse


GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


class GeminiProvider(LatticeProvider):
    name = "gemini"
    default_model = "gemini-3.1-pro-preview"  # RADIX: use Gemini 3.1 Pro, not Flash; env override LATTICE_GEMINI_DEFAULT_MODEL

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        # base.__init__ only stores config; it does NOT set self.default_model,
        # so consult() (which reads self.default_model) would otherwise ignore the
        # env-configured default. Propagate it so LATTICE_GEMINI_DEFAULT_MODEL wins.
        self.default_model = config.default_model or type(self).default_model
        self._base_url = config.base_url or GEMINI_BASE_URL

    async def consult(self, system, prompt, model=None, max_tokens=2048, timeout_s=60.0) -> SingleResponse:
        m = model or self.default_model
        if not self.config.available or not self.config.api_key:
            return SingleResponse(
                provider="gemini",
                model=m,
                response="",
                error=self.config.skip_reason or "gemini not configured",
            )

        t0 = time.monotonic()
        try:
            # Gemini 2.5-class models (Flash/Pro "latest") bill internal "thinking"
            # tokens against maxOutputTokens. With a conservative cap (e.g. 300)
            # thinking can consume the whole budget and leave near-zero visible
            # output, producing a MAX_TOKENS finishReason and a truncated reply.
            # Disable thinking by default so max_tokens behaves like every other
            # provider ("visible output cap"). Callers who want reasoning can
            # opt back in via a future explicit kwarg.
            generation_config = {
                "maxOutputTokens": max_tokens,
            }
            is_thinking_model = (
                "2.5" in m or "flash-latest" in m or "pro-latest" in m
            )
            if is_thinking_model:
                generation_config["thinkingConfig"] = {"thinkingBudget": 0}
            elif "gemini-3" in m:
                # Gemini 3.x (incl. 3.1-pro-preview) thinks dynamically and bills
                # thinking against maxOutputTokens. Forcing thinkingBudget=0 may be
                # rejected by 3-Pro, so instead raise the visible-output ceiling so
                # thinking can't starve the reply (guards the silent-truncation bug).
                generation_config["maxOutputTokens"] = max(max_tokens, max_tokens + 4096)

            payload = {
                "contents": [
                    {"parts": [{"text": prompt}]},
                ],
                "generationConfig": generation_config,
            }
            if system:
                payload["systemInstruction"] = {
                    "parts": [{"text": system}],
                }

            url = f"{self._base_url}/models/{m}:generateContent"
            headers = {
                "Content-Type": "application/json",
                "X-goog-api-key": self.config.api_key,
            }

            async with httpx.AsyncClient(timeout=timeout_s) as client:
                resp = await client.post(url, headers=headers, json=payload)

            if resp.status_code != 200:
                return SingleResponse(
                    provider="gemini",
                    model=m,
                    response="",
                    error=f"HTTP {resp.status_code}: {resp.text[:300]}",
                    latency_ms=int((time.monotonic() - t0) * 1000),
                )

            data = resp.json()

            # Extract text from candidates[0].content.parts[*].text
            text_parts = []
            candidates = data.get("candidates", [])
            if candidates:
                content = candidates[0].get("content", {})
                for part in content.get("parts", []):
                    if "text" in part:
                        text_parts.append(part["text"])
            text = "".join(text_parts)

            # Extract token usage. Include thoughtsTokenCount so that thinking
            # tokens are visible in cost/usage even when present (defends
            # against a regression of the silent-truncation bug if a future
            # caller re-enables thinking).
            usage = data.get("usageMetadata", {})
            tin = usage.get("promptTokenCount", 0)
            tout = usage.get("candidatesTokenCount", 0) + usage.get("thoughtsTokenCount", 0)
            cost = self.estimate_cost(tin, tout)

            return SingleResponse(
                provider="gemini",
                model=m,
                response=text,
                tokens_in=tin,
                tokens_out=tout,
                cost_usd=cost,
                latency_ms=int((time.monotonic() - t0) * 1000),
            )
        except Exception as e:
            return SingleResponse(
                provider="gemini",
                model=m,
                response="",
                error=f"{type(e).__name__}: {e}",
                latency_ms=int((time.monotonic() - t0) * 1000),
            )


def build_config() -> ProviderConfig:
    # Gemini 2.5 Pro pricing as of 2026: ~$1.25/M input, ~$5/M output
    api_key = os.environ.get("LATTICE_GEMINI_API_KEY")
    base_url = os.environ.get("LATTICE_GEMINI_BASE_URL", GEMINI_BASE_URL)
    config = ProviderConfig(
        name="gemini",
        api_key=api_key,
        base_url=base_url,
        default_model=os.environ.get("LATTICE_GEMINI_DEFAULT_MODEL", "gemini-3.1-pro-preview"),
        cost_in_per_mtok=1.25,
        cost_out_per_mtok=5.0,
    )
    if not api_key:
        config.available = False
        config.skip_reason = "LATTICE_GEMINI_API_KEY not set"
    else:
        config.available = True
    return config

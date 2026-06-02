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
    default_model = "gemini-flash-latest"  # 1M context, primary angle

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
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
            payload = {
                "contents": [
                    {"parts": [{"text": prompt}]},
                ],
                "generationConfig": {
                    "maxOutputTokens": max_tokens,
                },
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

            # Extract token usage
            usage = data.get("usageMetadata", {})
            tin = usage.get("promptTokenCount", 0)
            tout = usage.get("candidatesTokenCount", 0)
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
        default_model="gemini-flash-latest",
        cost_in_per_mtok=1.25,
        cost_out_per_mtok=5.0,
    )
    if not api_key:
        config.available = False
        config.skip_reason = "LATTICE_GEMINI_API_KEY not set"
    else:
        config.available = True
    return config

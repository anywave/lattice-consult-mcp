"""Provider registry. Each module exposes `build_config()` + a Provider class."""
from __future__ import annotations

from typing import Dict

from .base import LatticeProvider, ProviderConfig, SingleResponse


def load_all() -> Dict[str, LatticeProvider]:
    """Build every configured provider; skip unavailable ones in-place.

    Returns: { provider_name: instance } for ALL providers (available or not).
    Callers consult `instance.config.available` to know which to dispatch to.
    """
    from . import openai_provider, anthropic_provider, gemini_provider, xai_provider, jan_provider

    builders = [
        ("openai", openai_provider.build_config, openai_provider.OpenAIProvider),
        ("anthropic", anthropic_provider.build_config, anthropic_provider.AnthropicProvider),
        ("gemini", gemini_provider.build_config, gemini_provider.GeminiProvider),
        ("xai", xai_provider.build_config, xai_provider.XaiProvider),
        ("jan", jan_provider.build_config, jan_provider.JanProvider),
    ]
    out: Dict[str, LatticeProvider] = {}
    for name, build, cls in builders:
        try:
            cfg = build()
            out[name] = cls(cfg)
        except Exception as e:
            # Provider failed to even construct -- record an unavailable shell so
            # list_providers can report it honestly.
            cfg = ProviderConfig(
                name=name, available=False, skip_reason=f"{type(e).__name__}: {e}"
            )
            out[name] = _UnavailableProvider(cfg, name)
    return out


class _UnavailableProvider(LatticeProvider):
    """Stand-in for a provider that failed to construct. consult() always errs."""

    def __init__(self, config: ProviderConfig, name: str):
        super().__init__(config)
        self.name = name

    async def consult(self, system, prompt, model=None, max_tokens=2048, timeout_s=60.0) -> SingleResponse:
        return SingleResponse(
            provider=self.name,
            model=model or "(unknown)",
            response="",
            error=self.config.skip_reason or "provider construction failed",
        )

"""Provider base trait + shared data shapes.

Each cloud/local LLM provider implements LatticeProvider. The trait shape mirrors
DSM-26 AngularTechnique's Capability { angle, ... } pattern -- the provider IS
the angle; the model selection is sub-angle within the provider.
"""
from __future__ import annotations

import abc
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SingleResponse:
    """One provider's response to a single prompt."""

    provider: str
    model: str
    response: str
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass
class ProviderConfig:
    """Per-provider config read from env vars at server startup."""

    name: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    default_model: Optional[str] = None
    available: bool = False
    skip_reason: Optional[str] = None
    # cost-per-million-tokens, used by ensemble cost tracking. 0 for local.
    cost_in_per_mtok: float = 0.0
    cost_out_per_mtok: float = 0.0


class LatticeProvider(abc.ABC):
    """One LLM provider, exposing a uniform request shape."""

    name: str = ""
    default_model: str = ""

    def __init__(self, config: ProviderConfig):
        self.config = config

    @abc.abstractmethod
    async def consult(
        self,
        system: str,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: int = 2048,
        timeout_s: float = 60.0,
    ) -> SingleResponse:
        """Send prompt to this provider; return structured response.

        Implementations MUST catch all errors and return them in SingleResponse.error,
        not raise. The ensemble code degrades gracefully on per-provider failure.
        """
        ...

    def estimate_cost(self, tokens_in: int, tokens_out: int) -> float:
        """Default cost calc from config rates. Override if provider has tiered pricing."""
        return (
            (tokens_in / 1_000_000) * self.config.cost_in_per_mtok
            + (tokens_out / 1_000_000) * self.config.cost_out_per_mtok
        )


def env_config(
    name: str,
    api_key_env: str,
    default_model: str,
    cost_in_per_mtok: float = 0.0,
    cost_out_per_mtok: float = 0.0,
    base_url_env: Optional[str] = None,
    base_url_default: Optional[str] = None,
) -> ProviderConfig:
    """Build ProviderConfig from env vars; mark unavailable if key missing."""
    api_key = os.environ.get(api_key_env)
    base_url = (
        os.environ.get(base_url_env, base_url_default) if base_url_env else base_url_default
    )
    config = ProviderConfig(
        name=name,
        api_key=api_key,
        base_url=base_url,
        default_model=default_model,
        cost_in_per_mtok=cost_in_per_mtok,
        cost_out_per_mtok=cost_out_per_mtok,
    )
    if not api_key and name != "jan":  # jan has no api key requirement
        config.available = False
        config.skip_reason = f"{api_key_env} not set"
    else:
        config.available = True
    return config

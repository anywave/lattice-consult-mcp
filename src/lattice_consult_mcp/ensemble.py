"""Ensemble dispatcher -- runs N providers in parallel, applies synthesis."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from .providers.base import LatticeProvider, SingleResponse
from .synthesis import (
    SynthesisResult,
    synthesize_convergence,
    synthesize_majority,
    synthesize_raw_join,
)


@dataclass
class ProviderSpec:
    """Per-provider customization for an ensemble call."""

    provider: str
    model: Optional[str] = None  # None -> use provider default


@dataclass
class EnsembleResponse:
    synthesized_response: str
    convergence_score: float
    divergence_findings: List[str]
    confidence_signal: str
    providers_consulted: List[str]
    providers_failed: List[str]
    total_cost_usd: float
    total_latency_ms: int
    raw_outputs: Dict[str, dict] = field(default_factory=dict)


# Privacy tier definitions -- which providers are eligible per tier
_PRIVACY_TIERS = {
    "any": {"openai", "anthropic", "gemini", "xai", "jan"},
    "cloud_ok": {"openai", "anthropic", "gemini", "xai", "jan"},
    "local_only": {"jan"},
    # opt_out_only: providers with documented "no training on inputs" terms
    # (Anthropic API has this by default; OpenAI API requires opt-out config; Jan is local)
    "opt_out_only": {"anthropic", "openai", "jan"},
}


async def consult_ensemble(
    providers: Dict[str, LatticeProvider],
    prompt: str,
    system: str = "",
    selected: Optional[List[Union[str, ProviderSpec]]] = None,
    synthesis_mode: str = "convergence",
    privacy_tier: str = "any",
    return_raw: bool = False,
    max_tokens: int = 2048,
    timeout_s: float = 60.0,
) -> EnsembleResponse:
    """Dispatch prompt to N providers in parallel; synthesize results.

    `selected` is a list of provider names OR ProviderSpec for explicit model
    selection. None = use all available providers eligible under `privacy_tier`.
    """
    # Determine which provider+model pairs to consult
    eligible = _PRIVACY_TIERS.get(privacy_tier, _PRIVACY_TIERS["any"])
    if selected is None:
        # all available + eligible providers, default model each
        targets: List[ProviderSpec] = [
            ProviderSpec(provider=name, model=None)
            for name, p in providers.items()
            if p.config.available and name in eligible
        ]
    else:
        targets = []
        for s in selected:
            if isinstance(s, str):
                spec = ProviderSpec(provider=s, model=None)
            else:
                spec = s
            if spec.provider in eligible and spec.provider in providers and providers[spec.provider].config.available:
                targets.append(spec)

    if not targets:
        return EnsembleResponse(
            synthesized_response="(no eligible providers; check API keys + privacy_tier)",
            convergence_score=0.0,
            divergence_findings=[f"no providers eligible under tier={privacy_tier}"],
            confidence_signal="low",
            providers_consulted=[],
            providers_failed=[],
            total_cost_usd=0.0,
            total_latency_ms=0,
            raw_outputs={},
        )

    # Dispatch in parallel
    async def call_one(spec: ProviderSpec) -> SingleResponse:
        return await providers[spec.provider].consult(
            system=system,
            prompt=prompt,
            model=spec.model,
            max_tokens=max_tokens,
            timeout_s=timeout_s,
        )

    import time as _t
    t0 = _t.monotonic()
    responses = await asyncio.gather(*(call_one(spec) for spec in targets), return_exceptions=False)
    total_latency_ms = int((_t.monotonic() - t0) * 1000)

    # Split successes / failures
    successes = [r for r in responses if r.ok]
    failures = [r for r in responses if not r.ok]

    # Synthesize
    if synthesis_mode == "convergence":
        syn = synthesize_convergence(successes)
    elif synthesis_mode == "majority":
        syn = synthesize_majority(successes)
    elif synthesis_mode == "raw_join":
        syn = synthesize_raw_join(successes)
    elif synthesis_mode == "weighted":
        # v0.1 -- weighted mode degrades to convergence; v0.2 will add weights
        syn = synthesize_convergence(successes)
    else:
        syn = synthesize_convergence(successes)

    total_cost = sum(r.cost_usd for r in responses)

    raw = {}
    if return_raw:
        for r in responses:
            raw[f"{r.provider}/{r.model}"] = {
                "response": r.response,
                "error": r.error,
                "tokens_in": r.tokens_in,
                "tokens_out": r.tokens_out,
                "cost_usd": r.cost_usd,
                "latency_ms": r.latency_ms,
            }

    return EnsembleResponse(
        synthesized_response=syn.synthesized_response,
        convergence_score=syn.convergence_score,
        divergence_findings=syn.divergence_findings,
        confidence_signal=syn.confidence_signal,
        providers_consulted=[r.provider for r in successes],
        providers_failed=[f"{r.provider}: {r.error}" for r in failures],
        total_cost_usd=round(total_cost, 6),
        total_latency_ms=total_latency_ms,
        raw_outputs=raw,
    )

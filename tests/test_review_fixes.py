"""Tests for the 2026-06-14 review fixes.

FIX-1: single-provider synthesis reports convergence_score=None (undefined),
       not 0.0 (which would falsely read as total disagreement), + single_provider=True.
FIX-2: providers excluded by privacy-tier / availability are SURFACED in
       providers_skipped, never silently dropped.
"""
import asyncio

from lattice_consult_mcp.synthesis import synthesize_convergence
from lattice_consult_mcp.ensemble import consult_ensemble
from lattice_consult_mcp.providers.base import SingleResponse


def test_single_provider_convergence_is_none():
    r = SingleResponse(
        provider="jan", model="qwen2.5",
        response="This is one provider's answer, long enough to be a sentence.",
    )
    out = synthesize_convergence([r])
    assert out.convergence_score is None      # undefined for n=1, NOT 0.0
    assert out.single_provider is True


def test_ineligible_provider_is_surfaced_not_dropped():
    # gemini is not eligible under local_only -> must appear in providers_skipped
    resp = asyncio.run(
        consult_ensemble({}, "q", selected=["gemini"], privacy_tier="local_only")
    )
    assert any("gemini" in s for s in resp.providers_skipped)
    assert resp.providers_consulted == []

"""Live smoke test -- not committed to ensemble until v0.2.

Run with env vars set. Tests:
  1. list_providers shape
  2. consult_single for each configured provider
  3. consult_ensemble across all available providers + synthesis
"""
from __future__ import annotations

import asyncio
import os
import sys
import json
from pathlib import Path

# Allow direct exec without install
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lattice_consult_mcp.providers import load_all
from lattice_consult_mcp.ensemble import consult_ensemble


async def main():
    print("=" * 70)
    print("lattice-consult-mcp -- live smoke test")
    print("=" * 70)
    providers = load_all()

    # Inventory
    print("\n[1/3] Provider inventory:\n")
    for name, p in providers.items():
        status = "AVAILABLE" if p.config.available else f"unavailable ({p.config.skip_reason})"
        print(f"  {name:12s} {status}")
        if p.config.available:
            print(f"               default_model={p.config.default_model}  base_url={p.config.base_url or '(provider default)'}")

    available = [n for n, p in providers.items() if p.config.available]
    if not available:
        print("\nNo providers available -- nothing to test. Set env vars.")
        return

    # Single-provider test
    print(f"\n[2/3] Single-provider consultation tests:\n")
    test_prompt = "In 2-3 sentences, what is a Model Context Protocol (MCP) server and why is it useful?"
    test_system = "You are a concise technical explainer. No preamble."
    for name in available:
        print(f"\n  --- {name} ---")
        resp = await providers[name].consult(
            system=test_system,
            prompt=test_prompt,
            max_tokens=300,
            timeout_s=45.0,
        )
        if resp.ok:
            print(f"  model: {resp.model}")
            print(f"  latency: {resp.latency_ms}ms  cost: ${resp.cost_usd:.5f}  tokens: {resp.tokens_in} in / {resp.tokens_out} out")
            print(f"  response: {resp.response.strip()[:400]}")
            if len(resp.response) > 400:
                print(f"  ... ({len(resp.response)} total chars)")
        else:
            print(f"  ERROR: {resp.error}")

    if len(available) < 2:
        print("\n[3/3] Skipping ensemble test (need ≥2 providers).")
        return

    # Ensemble test
    print(f"\n[3/3] Ensemble test (synthesis_mode=convergence) across {available}:\n")
    arch_prompt = (
        "Should a cross-provider LLM consultation tool use convergence-based synthesis "
        "(detecting cross-provider agreement) or LLM-judge synthesis (one provider summarizes the others)? "
        "Give 3 sentences max -- recommendation and rationale."
    )
    ens = await consult_ensemble(
        providers=providers,
        prompt=arch_prompt,
        system="You are a software architect. Be concise.",
        synthesis_mode="convergence",
        return_raw=True,
        max_tokens=300,
        timeout_s=45.0,
    )
    print(f"  convergence_score: {ens.convergence_score}")
    print(f"  confidence: {ens.confidence_signal}")
    print(f"  total_cost: ${ens.total_cost_usd:.5f}  total_latency: {ens.total_latency_ms}ms")
    print(f"  consulted: {ens.providers_consulted}")
    if ens.providers_failed:
        print(f"  failed: {ens.providers_failed}")
    print()
    print("  --- SYNTHESIZED RESPONSE ---")
    for line in ens.synthesized_response.split("\n"):
        print(f"  {line}")
    print()
    if ens.divergence_findings:
        print("  --- DIVERGENCE FINDINGS ---")
        for f in ens.divergence_findings:
            print(f"  - {f}")
    print()
    print("  --- PER-PROVIDER RAW ---")
    for k, v in ens.raw_outputs.items():
        print(f"\n  [{k}]")
        if v.get("error"):
            print(f"    ERROR: {v['error']}")
        else:
            text = v.get("response", "").strip()
            print(f"    latency={v.get('latency_ms')}ms  cost=${v.get('cost_usd', 0):.5f}")
            for line in text.split("\n"):
                print(f"    {line}")


if __name__ == "__main__":
    asyncio.run(main())

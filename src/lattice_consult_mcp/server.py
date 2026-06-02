"""MCP server entry point. Exposes 3 tools: consult_single, consult_ensemble, list_providers."""
from __future__ import annotations

import asyncio
import json
import logging
import sys
from typing import Any, Dict, List, Optional

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from .ensemble import EnsembleResponse, ProviderSpec, consult_ensemble
from .providers import load_all
from .providers.base import LatticeProvider

logger = logging.getLogger("lattice-consult-mcp")
logging.basicConfig(level=logging.INFO, stream=sys.stderr)


def _build_server() -> tuple[Server, Dict[str, LatticeProvider]]:
    providers = load_all()
    available = [n for n, p in providers.items() if p.config.available]
    unavailable = [(n, p.config.skip_reason) for n, p in providers.items() if not p.config.available]
    logger.info(f"lattice-consult-mcp: available providers: {available}")
    if unavailable:
        for n, reason in unavailable:
            logger.warning(f"  unavailable {n}: {reason}")

    server = Server("lattice-consult-mcp")

    @server.list_tools()
    async def list_tools() -> List[Tool]:
        return [
            Tool(
                name="consult_single",
                description=(
                    "Send a prompt to ONE LLM provider; return its response. Use "
                    "for ad-hoc cross-provider checks. For default operation, "
                    "prefer consult_ensemble (synthesizes across providers)."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "provider": {"type": "string", "description": "One of: openai, anthropic, gemini, xai, jan"},
                        "model": {"type": "string", "description": "Provider-specific model id (omit for default)"},
                        "system": {"type": "string", "description": "System prompt (optional)"},
                        "prompt": {"type": "string", "description": "User prompt"},
                        "max_tokens": {"type": "integer", "default": 2048},
                        "timeout_s": {"type": "number", "default": 60.0},
                    },
                    "required": ["provider", "prompt"],
                },
            ),
            Tool(
                name="consult_ensemble",
                description=(
                    "Dispatch prompt to N providers in parallel; return ONE synthesized "
                    "response + convergence/divergence signals. This is the default tool "
                    "for cross-provider architectural review and second-opinion calls. "
                    "Implements Trellis DSM-26 AngularTechnique substrate."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string", "description": "User prompt"},
                        "system": {"type": "string", "description": "System prompt (optional)"},
                        "providers": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Provider names to consult. Omit = all available + eligible.",
                        },
                        "synthesis_mode": {
                            "type": "string",
                            "enum": ["convergence", "majority", "weighted", "raw_join"],
                            "default": "convergence",
                        },
                        "privacy_tier": {
                            "type": "string",
                            "enum": ["any", "cloud_ok", "local_only", "opt_out_only"],
                            "default": "any",
                            "description": (
                                "Provider eligibility filter. local_only = Jan only "
                                "(zero-egress for sensitive content). opt_out_only = "
                                "providers documented as not training on inputs."
                            ),
                        },
                        "return_raw": {"type": "boolean", "default": False},
                        "max_tokens": {"type": "integer", "default": 2048},
                        "timeout_s": {"type": "number", "default": 60.0},
                    },
                    "required": ["prompt"],
                },
            ),
            Tool(
                name="list_providers",
                description=(
                    "Return inventory of configured providers, their availability, "
                    "default models, and cost rates. Use to discover what's wired up."
                ),
                inputSchema={"type": "object", "properties": {}},
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
        try:
            if name == "consult_single":
                provider_name = arguments["provider"]
                if provider_name not in providers:
                    return [TextContent(type="text", text=json.dumps({"error": f"unknown provider: {provider_name}"}))]
                resp = await providers[provider_name].consult(
                    system=arguments.get("system", ""),
                    prompt=arguments["prompt"],
                    model=arguments.get("model"),
                    max_tokens=arguments.get("max_tokens", 2048),
                    timeout_s=arguments.get("timeout_s", 60.0),
                )
                payload = {
                    "provider": resp.provider,
                    "model": resp.model,
                    "response": resp.response,
                    "tokens_in": resp.tokens_in,
                    "tokens_out": resp.tokens_out,
                    "cost_usd": resp.cost_usd,
                    "latency_ms": resp.latency_ms,
                    "error": resp.error,
                }
                return [TextContent(type="text", text=json.dumps(payload, indent=2))]

            elif name == "consult_ensemble":
                ens = await consult_ensemble(
                    providers=providers,
                    prompt=arguments["prompt"],
                    system=arguments.get("system", ""),
                    selected=arguments.get("providers"),
                    synthesis_mode=arguments.get("synthesis_mode", "convergence"),
                    privacy_tier=arguments.get("privacy_tier", "any"),
                    return_raw=arguments.get("return_raw", False),
                    max_tokens=arguments.get("max_tokens", 2048),
                    timeout_s=arguments.get("timeout_s", 60.0),
                )
                payload = {
                    "synthesized_response": ens.synthesized_response,
                    "convergence_score": ens.convergence_score,
                    "divergence_findings": ens.divergence_findings,
                    "confidence_signal": ens.confidence_signal,
                    "providers_consulted": ens.providers_consulted,
                    "providers_failed": ens.providers_failed,
                    "total_cost_usd": ens.total_cost_usd,
                    "total_latency_ms": ens.total_latency_ms,
                }
                if ens.raw_outputs:
                    payload["raw_outputs"] = ens.raw_outputs
                return [TextContent(type="text", text=json.dumps(payload, indent=2))]

            elif name == "list_providers":
                inventory = {}
                for n, p in providers.items():
                    inventory[n] = {
                        "available": p.config.available,
                        "default_model": p.config.default_model,
                        "cost_in_per_mtok": p.config.cost_in_per_mtok,
                        "cost_out_per_mtok": p.config.cost_out_per_mtok,
                        "skip_reason": p.config.skip_reason,
                        "base_url": p.config.base_url,
                    }
                return [TextContent(type="text", text=json.dumps(inventory, indent=2))]

            else:
                return [TextContent(type="text", text=json.dumps({"error": f"unknown tool: {name}"}))]
        except Exception as e:
            logger.exception(f"tool {name} failed")
            return [TextContent(type="text", text=json.dumps({"error": f"{type(e).__name__}: {e}"}))]

    return server, providers


async def _run() -> None:
    server, _providers = _build_server()
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()

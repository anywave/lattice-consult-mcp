# lattice-consult-mcp -- design

**Status:** v0.1 spec, 2026-06-02. First concrete impl of Trellis DSM-26 `AngularTechnique` substrate (per FP-MSG-310 §4 / TR-023 §4).

## Goal

A single MCP server that exposes cross-provider LLM consultation as tools, callable from any Claude Code lattice node (or Codex CLI, or Gemini CLI). The server runs 1-N providers in parallel for a single prompt, synthesizes the responses, and returns ONE answer plus confidence/divergence signals.

**The crucial design move:** the orchestrator (Claude) consumes the synthesized output, not the raw per-provider outputs. The operator's attention budget (~2.7k tokens per RADIX self-report) stays intact.

## Providers (v0.1)

| Provider | API | Cost shape | Angular value |
|---|---|---|---|
| OpenAI (GPT-5 / o1) | api.openai.com | high $/token, high quality, slow on o1 | OpenAI training corpus + RLHF |
| Anthropic (Claude Opus/Sonnet) | api.anthropic.com | medium $/token, high quality | Anthropic training + Constitutional AI |
| Google (Gemini 2.5 Pro) | generativelanguage.googleapis.com | low $/token, 1M context | Google training corpus, large context |
| xAI (Grok 2/4) | api.x.ai/v1 | medium $/token, OpenAI-compatible SDK | X/Twitter real-time corpus, less RLHF flattening |
| Jan (local) | localhost:1337/v1 (configurable) | $0/call, GPU-bound | privacy, multi-model intra-angularity, no rate limits |

Jan provider supports calling with explicit `model` parameter so a single Jan instance hosts multiple "angles" via model swap (e.g., qwen2.5:7b and llama3.1:70b count as two distinct angles in ensemble).

## Tools exposed (MCP)

```
consult_single(provider: str, model: str, system: str, prompt: str, max_tokens: int) -> SingleResponse
consult_ensemble(prompt: str, system: str, providers: list[str|ProviderSpec], synthesis_mode: str, return_raw: bool, privacy_tier: str) -> EnsembleResponse
list_providers() -> ProviderInventory
```

### SingleResponse

```
{
  provider: str,
  model: str,
  response: str,
  tokens_in: int,
  tokens_out: int,
  cost_usd: float,    # 0 for Jan
  latency_ms: int,
  error: Optional[str]
}
```

### EnsembleResponse

```
{
  synthesized_response: str,        # the thing the orchestrator reads
  convergence_score: float,         # 0.0 (wild disagreement) -- 1.0 (unanimous)
  divergence_findings: list[str],   # brief: where providers disagreed
  confidence_signal: "high"|"medium"|"low",
  providers_consulted: list[str],
  providers_failed: list[str],
  total_cost_usd: float,
  total_latency_ms: int,
  raw_outputs: dict[str, SingleResponse]  # only if return_raw=true
}
```

### synthesis_mode

- `"convergence"` (default): synthesizes by extracting agreed-upon claims + flagging disagreements. Best for architectural decisions.
- `"majority"`: returns the response most providers agree with. Best for fact-check / yes-no questions.
- `"weighted"`: trust-weighted blend (Jan-7B weighted lower than GPT-5). Best for ranking / preference.
- `"raw_join"`: concatenates per-provider responses with headers. Escape hatch when synthesis is wrong-shaped.

### privacy_tier

- `"any"` (default): all configured providers eligible
- `"cloud_ok"`: any cloud + Jan
- `"local_only"`: Jan only — for legally-sensitive or contractually-restricted content
- `"opt_out_only"`: only providers with documented "do not train on inputs" terms (currently: Anthropic enterprise, OpenAI API with opt-out, Jan)

## Provider configuration

Environment variables:

```
LATTICE_OPENAI_API_KEY=sk-...
LATTICE_ANTHROPIC_API_KEY=sk-ant-...
LATTICE_GEMINI_API_KEY=...
LATTICE_XAI_API_KEY=xai-...
LATTICE_JAN_BASE_URL=http://localhost:1337/v1
LATTICE_JAN_DEFAULT_MODEL=qwen2.5:7b-instruct
LATTICE_DEFAULT_MAX_TOKENS=2048
```

Missing keys cause that provider to be skipped silently (logged warn); ensemble still runs across available providers. `list_providers()` reports `available: bool` per provider.

## Synthesis algorithm (convergence mode v0.1)

1. Call all providers in parallel via `asyncio.gather` with per-provider timeout (default 60s).
2. Collect successful responses; record failures.
3. For each response, extract claim sentences via simple sentence-segmentation.
4. Cluster claims by string similarity (rapidfuzz partial_ratio ≥ 85).
5. For each cluster:
   - if ≥75% of providers contributed a claim to it -> "consensus claim"
   - if 25-75% -> "partial consensus, note providers"
   - if <25% -> "outlier claim, attribute to provider"
6. Synthesize: render consensus claims as flat prose, partial claims with "[N of M providers]" markers, outliers as "noted: [provider] also argued X."
7. Compute convergence_score = (consensus_claim_count) / (total_unique_claim_count).
8. confidence_signal: "high" if score ≥0.7, "medium" if ≥0.4, "low" otherwise.

This is intentionally simple v0.1. v0.2 can add LLM-based synthesis (use one provider to synthesize the others' outputs — the canonical "judge" pattern), but adds cost + a meta-bias.

## Substrate mapping (DSM)

| Component | DSM substrate |
|---|---|
| Provider trait + per-provider impls | DSM-26 AngularTechnique (first concrete impl) |
| `VerifiedAnchor` registry passed through to each provider | DSM-16 SubstrateTrust |
| Convergence scoring + divergence findings | DSM-17 AdversarialReview (lightweight) |
| Coordinator-synthesized output | DSM-25-adjacent canonicalization |
| Future: signed convergence report | DSM-27 BijectiveGrammar candidate |

## Integration with Claude Code

Add to user's `~/.claude/claude_desktop_config.json` (or `.mcp.json` in repo):

```json
{
  "mcpServers": {
    "lattice-consult": {
      "command": "python",
      "args": ["-m", "lattice_consult_mcp"],
      "env": {
        "LATTICE_OPENAI_API_KEY": "...",
        "LATTICE_ANTHROPIC_API_KEY": "...",
        "LATTICE_GEMINI_API_KEY": "...",
        "LATTICE_XAI_API_KEY": "...",
        "LATTICE_JAN_BASE_URL": "http://localhost:1337/v1"
      }
    }
  }
}
```

Then in Claude Code: `consult_ensemble({prompt: "...", system: "..."})` becomes a tool call.

## Non-goals (explicit)

- Provider-specific feature surfaces (function calling, vision, etc.). Text-in / text-out only at v0.1.
- Streaming responses. v0.1 is request/response. Streaming adds complexity for marginal benefit in the consultation use case.
- Persistent conversation history across calls. Each `consult_ensemble` call is stateless. Multi-turn dialogue is the orchestrator's job.
- Cost optimization beyond skip-on-missing-key. v0.1 doesn't try to "pick the cheapest provider for this prompt."

## Open follow-ups (post-v0.1)

- Voice integration via Mobius STT+TTS (separate work stream)
- LLM-judge synthesis mode
- Trust-weighted synthesis using DSM-16 anchor registry
- Per-provider streaming proxied through MCP
- Caching layer for identical prompt+provider+model triples

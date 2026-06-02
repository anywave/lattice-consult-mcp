# lattice-consult-mcp

A single MCP server that gives any MCP-capable client (Claude Code, Codex CLI, Gemini CLI, custom agents) a uniform way to consult multiple LLM providers in parallel and receive **one synthesized answer** — not five raw outputs to merge by hand.

**Providers:** OpenAI · Anthropic · Google Gemini · xAI Grok · Jan (local)

## Why

When your primary LLM provider is rate-limited or throttled, the work stops. When you want a second/third opinion on an architectural decision, you have to manually paste the same prompt into multiple web UIs and synthesize the responses yourself. This server runs all configured providers in parallel through one tool call and applies a synthesis algorithm so the consumer reads one answer with explicit convergence/divergence signals.

The synthesis step is the load-bearing design choice. Raw multi-provider outputs cost more attention to consume than they save in coverage; the synthesized output preserves the cross-provider coverage benefit without the merge tax.

## What you get

| Tool | Purpose |
|---|---|
| `consult_ensemble` | **default** — parallel dispatch across N providers; one synthesized response back, plus convergence score + divergence findings + cost + latency |
| `consult_single` | escape hatch — one provider, raw response |
| `list_providers` | inventory — what's wired, what's missing keys |

## Install

Requires Python 3.10+.

```powershell
git clone https://github.com/anywave/lattice-consult-mcp.git
cd lattice-consult-mcp
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

(On macOS/Linux: `source .venv/bin/activate`)

## Configure (API keys)

Set env vars before launching. Missing keys cause that provider to be skipped silently — `list_providers` reports availability honestly so you always know which providers are live.

```powershell
# Cloud providers (set what you have; skip what you don't)
$env:LATTICE_OPENAI_API_KEY    = "sk-..."          # https://platform.openai.com
$env:LATTICE_ANTHROPIC_API_KEY = "sk-ant-..."      # https://console.anthropic.com
$env:LATTICE_GEMINI_API_KEY    = "..."             # https://aistudio.google.com
$env:LATTICE_XAI_API_KEY       = "xai-..."         # https://console.x.ai

# Local provider (Jan -- https://jan.ai)
$env:LATTICE_JAN_BASE_URL      = "http://localhost:1337/v1"
$env:LATTICE_JAN_DEFAULT_MODEL = "qwen2.5:7b-instruct"
```

Copy `.env.example` to `.env` for convenience; `.env` is gitignored.

## Wire into Claude Code

Add to `~/.claude/claude_desktop_config.json` (or `.mcp.json` in your project):

```json
{
  "mcpServers": {
    "lattice-consult": {
      "command": "python",
      "args": ["-m", "lattice_consult_mcp"],
      "env": {
        "LATTICE_OPENAI_API_KEY": "sk-...",
        "LATTICE_ANTHROPIC_API_KEY": "sk-ant-...",
        "LATTICE_GEMINI_API_KEY": "...",
        "LATTICE_XAI_API_KEY": "xai-...",
        "LATTICE_JAN_BASE_URL": "http://localhost:1337/v1"
      }
    }
  }
}
```

Restart Claude Code. The three tools appear.

For Codex CLI or Gemini CLI: same `mcpServers` shape in their respective config files (both follow the MCP spec).

## Usage examples

**Default ensemble call:**

```
consult_ensemble({
  prompt: "Should the EntityEdge canonical type include a metadata field? Why or why not?",
  system: "You are an architectural reviewer. Be concise."
})
```

Returns:

```json
{
  "synthesized_response": "**Consensus** (all/most providers):\n- Yes, metadata should be included for cross-edge audit trails ...\n\n**Partial consensus**:\n- ...",
  "convergence_score": 0.83,
  "divergence_findings": ["gemini, xai said: ..."],
  "confidence_signal": "high",
  "providers_consulted": ["openai", "anthropic", "gemini", "xai", "jan"],
  "providers_failed": [],
  "total_cost_usd": 0.041,
  "total_latency_ms": 4823
}
```

**Local-only (no cloud egress for sensitive content):**

```
consult_ensemble({
  prompt: "Review this confidential strategy memo: ...",
  privacy_tier: "local_only"
})
```

Only Jan answers. Cost $0. Zero data leaves your network.

**Single-provider escape hatch:**

```
consult_single({
  provider: "gemini",
  model: "gemini-2.5-pro",
  prompt: "Summarize this 800-page PDF: ..."
})
```

(Gemini for big-context tasks; pick the provider whose strength matches the query.)

## Synthesis modes

| Mode | Best for | How it works |
|---|---|---|
| `convergence` (default) | Architectural decisions, design review | Sentence-segment + rapidfuzz cluster across providers → consensus rendered flat with partial-consensus and outlier markers |
| `majority` | Fact-check, yes/no | Picks mid-length response (heuristic: shortest tends to be refusal, longest tends to be hedged, mid is usually the substantive answer) |
| `weighted` | Ranking with trust weights | Currently degrades to `convergence`; trust-weighted blend planned for v0.2 |
| `raw_join` | Escape hatch when synthesis is wrong-shaped | Concatenates per-provider responses with headers — caller does their own merge |

## Privacy tiers

| Tier | Eligible providers |
|---|---|
| `any` (default) | all 5 |
| `cloud_ok` | all 5 |
| `local_only` | Jan only — zero-egress for sensitive content |
| `opt_out_only` | Anthropic, OpenAI (API with opt-out), Jan — providers documented as not training on inputs |

## Cost shape

Per `consult_ensemble` call with all 5 providers, typical ~2k input + ~500 output tokens (approximate 2026 pricing):

| Provider | Cost per call |
|---|---|
| OpenAI gpt-4o | ~$0.010 |
| Anthropic Claude Sonnet | ~$0.013 |
| Gemini 2.5 Pro | ~$0.005 |
| xAI grok-4-fast | ~$0.013 |
| Jan (local) | $0 |
| **Total** | **~$0.04 / ensemble call** |

At 100 ensemble calls/day ≈ $4/day ≈ $120/month. Provider pricing changes; this is a rough order-of-magnitude estimate, not a quote.

## Architecture

```
Client (Claude Code / Codex / Gemini CLI / custom)
    │
    │ MCP stdio
    ▼
lattice-consult-mcp server
    │
    ├─ OpenAIProvider  ──► api.openai.com
    ├─ AnthropicProvider ► api.anthropic.com
    ├─ GeminiProvider ───► generativelanguage.googleapis.com
    ├─ XaiProvider ──────► api.x.ai/v1
    ├─ JanProvider ──────► localhost:1337/v1 (or configured)
    │
    ├─ ensemble.consult_ensemble()
    │     asyncio.gather parallel dispatch with per-provider timeout
    │     graceful per-provider failure (errors flow through, not raised)
    │
    └─ synthesis.synthesize_*()
          convergence: rapidfuzz claim-clustering + render
          majority: mid-length heuristic
          raw_join: concat with headers
```

Each provider implements a uniform `LatticeProvider` async trait. Adding a new provider is a single file matching the existing pattern (`providers/<name>_provider.py` + register in `providers/__init__.py`).

## Background

The synthesis approach in this server originates from a substrate-design pattern called *AngularTechnique* in the Trellis DSM (Document Substrate Manager) project. The idea: orthogonal-perspective workers each produce a partial view, and a coordinator extracts cross-worker convergence as a verification signal. Cross-provider LLM consultation is a natural instance — each provider's training data and RLHF gives it a distinct "angle," and the convergence across angles is what makes the synthesized output more trustworthy than any single provider's response.

The substrate primitives this server informally implements:

- **AngularTechnique** — each provider = one angle; the trait shape ensures angle distinctness at the type level
- **AdversarialReview** — convergence detection + divergence findings are the lightweight review signal
- **SubstrateTrust** — verified anchors (system prompts, context documents) pass through to every angle uniformly

These are documented more formally in the design doc; for most users they're invisible — you just call the tool.

## Design doc

See [`docs/design.md`](docs/design.md) for the full architectural spec including:
- Tool schemas and response shapes
- Synthesis algorithm details
- Provider configuration semantics
- Non-goals and roadmap

## Roadmap (post-v0.1)

- LLM-judge synthesis mode (use one provider to synthesize the others — the canonical "judge" pattern; higher quality, higher cost)
- Trust-weighted synthesis (weighted by provider model size / reliability)
- Streaming responses proxied through MCP
- Caching layer for identical prompt + provider + model triples
- Voice integration via local STT/TTS (separate work stream)
- Additional providers (Mistral, Cohere, Together AI, OpenRouter aggregation)

## Contributing

Issues and pull requests welcome. For new provider adapters, follow the pattern in `providers/openai_provider.py` (cloud) or `providers/jan_provider.py` (local). The `LatticeProvider` trait surface is small (one async method) and provider adapters should never raise — return errors in `SingleResponse.error` so ensemble degrades gracefully.

## License

MIT. See [`LICENSE`](LICENSE).

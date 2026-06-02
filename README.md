# lattice-consult-mcp

Cross-provider LLM consultation via a single MCP tool. First concrete impl of Trellis DSM-26 `AngularTechnique` substrate.

**Providers:** OpenAI · Anthropic · Google Gemini · xAI Grok · Jan (local)

**What it solves:** when Claude (or any single provider) is rate-limited or throttled, dispatch architectural review and consultation queries across N providers in parallel; consume ONE synthesized response with convergence/divergence signals. Operator attention budget stays intact.

## Install

Requires Python 3.10+.

```powershell
cd C:\ANYWAVEREPO\lattice-consult-mcp
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

## Configure (API keys)

Set env vars before launching. Missing keys cause that provider to be skipped silently — `list_providers` reports availability honestly.

```powershell
# Cloud providers
$env:LATTICE_OPENAI_API_KEY    = "sk-..."          # https://platform.openai.com
$env:LATTICE_ANTHROPIC_API_KEY = "sk-ant-..."      # https://console.anthropic.com
$env:LATTICE_GEMINI_API_KEY    = "..."             # https://aistudio.google.com
$env:LATTICE_XAI_API_KEY       = "xai-..."         # https://console.x.ai

# Local provider (Jan)
$env:LATTICE_JAN_BASE_URL      = "http://localhost:1337/v1"
$env:LATTICE_JAN_DEFAULT_MODEL = "qwen2.5:7b-instruct"
```

You can configure as few or many as you want. The ensemble degrades gracefully — if only Anthropic + Jan are configured, ensemble runs across those two.

## Wire into Claude Code

Add to `~/.claude/claude_desktop_config.json` (or `.mcp.json` in your repo):

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

Restart Claude Code. The 3 tools (`consult_single`, `consult_ensemble`, `list_providers`) appear.

## Usage examples (from Claude Code)

**Quick second-opinion call:**

```
consult_ensemble({
  prompt: "Should the EntityEdge canonical type include a metadata field? Why or why not?",
  system: "You are an architectural reviewer. Be concise."
})
```

Returns:

```json
{
  "synthesized_response": "**Consensus** (all/most providers):\n- Yes, metadata should be included...\n\n**Partial consensus**:\n- ...\n",
  "convergence_score": 0.83,
  "divergence_findings": ["gemini, xai said: ..."],
  "confidence_signal": "high",
  "providers_consulted": ["openai", "anthropic", "gemini", "xai", "jan"],
  "providers_failed": [],
  "total_cost_usd": 0.041,
  "total_latency_ms": 4823
}
```

**Local-only (no cloud egress, for sensitive content):**

```
consult_ensemble({
  prompt: "Review this legal strategy memo: ...",
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

## Substrate mapping (Trellis DSM)

| Component | DSM substrate |
|---|---|
| Provider trait + per-provider impls | DSM-26 `AngularTechnique` |
| Verified anchors through to provider context | DSM-16 `SubstrateTrust` |
| Convergence + divergence findings | DSM-17 `AdversarialReview` (lightweight) |
| Coordinator-synthesized output | DSM-25-adjacent canonicalization |

This is the **first concrete impl of DSM-26** per FP-MSG-310 §4. Dogfood.

## Synthesis modes

| Mode | Best for | How it works |
|---|---|---|
| `convergence` (default) | Architectural decisions, design review | Sentence-segment + rapidfuzz-cluster; render consensus flat with partial+outlier markers |
| `majority` | Fact-check, yes/no | Picks mid-length response; assumes shortest is refusal, longest is hedged |
| `weighted` | Ranking (v0.1: degrades to convergence; v0.2 will add weights) | TODO |
| `raw_join` | Escape hatch when synthesis is wrong-shaped | Concatenates per-provider responses with headers |

## Privacy tiers

| Tier | Eligible providers |
|---|---|
| `any` (default) | all 5 |
| `cloud_ok` | all 5 |
| `local_only` | jan only |
| `opt_out_only` | anthropic, openai, jan (documented no-train-on-inputs) |

## Costs (approximate, 2026)

Per `consult_ensemble` call with all 5 providers, typical ~2k input + ~500 output tokens:

| Provider | Cost per call |
|---|---|
| OpenAI gpt-4o | $0.010 |
| Anthropic Sonnet 4.6 | $0.013 |
| Gemini 2.5 Pro | $0.005 |
| xAI grok-4-fast | $0.013 |
| Jan | $0 |
| **Total** | **~$0.04 per ensemble call** |

At 100 ensemble calls/day = ~$4/day = ~$120/mo. For reference: that's the cost of bypassing the throttling problem.

## Architecture

```
Claude Code (orchestrator)
    |
    | MCP stdio
    |
    v
lattice-consult-mcp server
    |
    +-- OpenAIProvider ---> api.openai.com
    +-- AnthropicProvider -> api.anthropic.com
    +-- GeminiProvider ----> generativelanguage.googleapis.com
    +-- XaiProvider -------> api.x.ai
    +-- JanProvider -------> localhost:1337 (or configured)
    |
    +-- ensemble.consult_ensemble()  (parallel dispatch via asyncio.gather)
    +-- synthesis.synthesize_convergence()  (rapidfuzz cluster + render)
```

## Design doc

See `docs/design.md` for the full architectural spec.

## Roadmap (post-v0.1)

- LLM-judge synthesis mode (use one provider to synthesize the others — canonical "judge" pattern)
- Trust-weighted synthesis using DSM-16 anchor registry
- Streaming responses (proxied through MCP)
- Caching layer for identical prompt+provider+model triples
- Voice integration via Mobius STT+TTS (separate work stream)

## License

TBD (suggest MIT or Apache-2.0; pick at first push to public).

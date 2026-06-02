"""Synthesis algorithms for ensemble responses.

Per design.md §"Synthesis algorithm (convergence mode v0.1)", the v0.1 synthesis
is intentionally simple: sentence-segment, rapidfuzz-cluster, render consensus
flat with partial+outlier markers. v0.2 will add LLM-judge synthesis.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

try:
    from rapidfuzz import fuzz
except ImportError:  # pragma: no cover -- soft-degrade if dep missing
    fuzz = None

from .providers.base import SingleResponse


_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"\'\(\[])")


def _split_sentences(text: str) -> List[str]:
    """Naive sentence segmentation. Robust enough for synthesis-cluster purposes."""
    if not text:
        return []
    # collapse markdown bullets/numbering -- the FACT in the bullet is what we cluster
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)
    raw = _SENTENCE_RE.split(text)
    return [s.strip() for s in raw if s.strip() and len(s.strip()) > 12]


def _cluster_claims(
    per_provider_sentences: Dict[str, List[str]],
    similarity_threshold: int = 85,
) -> List[Dict]:
    """Cluster claims across providers by sentence-similarity.

    Returns a list of clusters: each is { 'rep': str, 'providers': [str], 'sentences': {provider: str} }.
    """
    if fuzz is None:
        # fallback: treat each provider's response as one giant claim block
        return [
            {"rep": " ".join(sentences)[:400], "providers": [p], "sentences": {p: " ".join(sentences)}}
            for p, sentences in per_provider_sentences.items()
            if sentences
        ]

    clusters: List[Dict] = []
    for provider, sentences in per_provider_sentences.items():
        for sent in sentences:
            placed = False
            for cluster in clusters:
                if fuzz.partial_ratio(sent, cluster["rep"]) >= similarity_threshold:
                    cluster["providers"].append(provider)
                    cluster["sentences"][provider] = sent
                    # update rep to shortest sentence in cluster (most canonical)
                    if len(sent) < len(cluster["rep"]):
                        cluster["rep"] = sent
                    placed = True
                    break
            if not placed:
                clusters.append({"rep": sent, "providers": [provider], "sentences": {provider: sent}})
    return clusters


@dataclass
class SynthesisResult:
    synthesized_response: str
    convergence_score: float
    divergence_findings: List[str]
    confidence_signal: str  # "high" | "medium" | "low"


def synthesize_convergence(
    successful_responses: List[SingleResponse],
    similarity_threshold: int = 85,
) -> SynthesisResult:
    """Default convergence synthesis."""
    if not successful_responses:
        return SynthesisResult(
            synthesized_response="(no provider responses available)",
            convergence_score=0.0,
            divergence_findings=["all providers failed or unavailable"],
            confidence_signal="low",
        )

    if len(successful_responses) == 1:
        # Only one provider -- pass through; report low convergence (n=1)
        r = successful_responses[0]
        return SynthesisResult(
            synthesized_response=r.response,
            convergence_score=0.0,
            divergence_findings=[f"only one provider consulted: {r.provider}"],
            confidence_signal="low",
        )

    per_provider_sentences = {
        r.provider: _split_sentences(r.response) for r in successful_responses
    }
    n_providers = len(successful_responses)
    clusters = _cluster_claims(per_provider_sentences, similarity_threshold)

    consensus_threshold = 0.75
    partial_threshold = 0.25

    consensus_parts: List[str] = []
    partial_parts: List[Tuple[str, List[str]]] = []
    outlier_parts: List[Tuple[str, str]] = []

    for cluster in clusters:
        coverage = len(set(cluster["providers"])) / n_providers
        if coverage >= consensus_threshold:
            consensus_parts.append(cluster["rep"])
        elif coverage >= partial_threshold:
            partial_parts.append((cluster["rep"], list(set(cluster["providers"]))))
        else:
            # outlier -- only one provider contributed
            outlier_parts.append((cluster["rep"], cluster["providers"][0]))

    # Render synthesized response
    lines: List[str] = []
    if consensus_parts:
        lines.append("**Consensus** (all/most providers):")
        for c in consensus_parts:
            lines.append(f"- {c}")
    if partial_parts:
        lines.append("")
        lines.append("**Partial consensus**:")
        for c, provs in partial_parts:
            lines.append(f"- {c} _[{', '.join(provs)}]_")
    if outlier_parts:
        lines.append("")
        lines.append("**Outlier observations** (noted, not converged):")
        for c, prov in outlier_parts:
            lines.append(f"- {c} _[{prov} only]_")

    if not lines:
        # fallback -- providers all said different things; concat first sentences
        first_sents = [
            (r.provider, _split_sentences(r.response)[:1])
            for r in successful_responses
        ]
        lines = ["**No convergence found.** Per-provider first sentences:"]
        for prov, sents in first_sents:
            lines.append(f"- _[{prov}]_ {sents[0] if sents else '(empty)'}")

    synthesized = "\n".join(lines)

    # Convergence score
    total_unique = len(clusters)
    consensus_count = sum(
        1
        for cl in clusters
        if len(set(cl["providers"])) / n_providers >= consensus_threshold
    )
    score = consensus_count / max(total_unique, 1)

    # Confidence signal
    if score >= 0.7:
        confidence = "high"
    elif score >= 0.4:
        confidence = "medium"
    else:
        confidence = "low"

    # Divergence findings -- brief
    divergence_findings = []
    for c, provs in partial_parts[:5]:  # cap at 5 to stay terse
        divergence_findings.append(
            f"{', '.join(provs)} said: {c[:120]}{'...' if len(c) > 120 else ''}"
        )
    for c, prov in outlier_parts[:3]:
        divergence_findings.append(
            f"{prov} alone: {c[:120]}{'...' if len(c) > 120 else ''}"
        )

    return SynthesisResult(
        synthesized_response=synthesized,
        convergence_score=round(score, 3),
        divergence_findings=divergence_findings,
        confidence_signal=confidence,
    )


def synthesize_raw_join(successful_responses: List[SingleResponse]) -> SynthesisResult:
    """Escape hatch: concatenate per-provider responses with headers."""
    if not successful_responses:
        return SynthesisResult(
            synthesized_response="(no provider responses available)",
            convergence_score=0.0,
            divergence_findings=["all providers failed"],
            confidence_signal="low",
        )
    parts = []
    for r in successful_responses:
        parts.append(f"### [{r.provider} / {r.model}]\n\n{r.response}")
    return SynthesisResult(
        synthesized_response="\n\n---\n\n".join(parts),
        convergence_score=0.0,
        divergence_findings=["raw join mode -- synthesis not applied"],
        confidence_signal="medium",  # raw mode -- caller does their own synthesis
    )


def synthesize_majority(successful_responses: List[SingleResponse]) -> SynthesisResult:
    """Majority mode: return the response with most claim-cluster representation.

    Best for yes/no / fact-check queries where the answer is short.
    """
    if not successful_responses:
        return SynthesisResult(
            synthesized_response="(no provider responses available)",
            convergence_score=0.0,
            divergence_findings=["all providers failed"],
            confidence_signal="low",
        )
    # heuristic: rank by mid-length response (shortest tends to be "no I refuse",
    # longest tends to be hedged). Mid is usually the substantive yes/no.
    sorted_by_len = sorted(successful_responses, key=lambda r: len(r.response))
    pick = sorted_by_len[len(sorted_by_len) // 2]
    return SynthesisResult(
        synthesized_response=pick.response,
        convergence_score=0.5,  # majority mode is by definition mid-confidence
        divergence_findings=[
            f"majority-mode pick: {pick.provider}"
        ] + [f"others: {r.provider} ({len(r.response)} chars)" for r in successful_responses if r.provider != pick.provider][:3],
        confidence_signal="medium",
    )

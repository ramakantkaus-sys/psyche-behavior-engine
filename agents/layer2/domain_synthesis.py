"""
PSYCHE OS — Layer 2: Domain Synthesis Agents
Each L2 agent receives all L1 MicroSignals from its domain and produces
a unified domain-level synthesis using an LLM call.
"""

from __future__ import annotations
import json
import re
from typing import Any, Dict, List, Optional

from anthropic import Anthropic
from psyche_schemas import MicroSignal


class DomainSynthesizer:
    """
    Layer 2 agent that synthesizes multiple L1 MicroSignals
    from the same psychological domain into a coherent domain report.
    """

    MODEL = "claude-haiku-4-5-20251001"
    MAX_TOKENS = 3000

    def __init__(self, client: Optional[Anthropic] = None):
        self.client = client or Anthropic()

    def synthesize(
        self,
        domain_name: str,
        signals: List[MicroSignal],
        conflicts: List[Dict],
        target_speaker: str,
    ) -> Dict[str, Any]:
        """
        Synthesize all L1 signals in a domain into a unified analysis.

        Returns a structured dict with:
        - domain_summary: narrative synthesis
        - key_findings: top 3-5 insights
        - confidence: overall domain confidence
        - internal_consistency: how well the agents agree
        - recommendations: domain-specific insights
        """
        if not signals:
            return {
                "domain": domain_name,
                "domain_summary": "No signals available for this domain.",
                "key_findings": [],
                "confidence": 0.0,
                "internal_consistency": "N/A",
            }

        # Build context from all L1 signals
        signal_summaries = []
        for sig in signals:
            signal_summaries.append(
                f"AGENT: {sig.agent_name} ({sig.agent_id})\n"
                f"  Dimension: {sig.dimension}\n"
                f"  Rating: {sig.rating:.2f} | Confidence: {sig.confidence:.2f}\n"
                f"  Label: {sig.label}\n"
                f"  Summary: {sig.summary}\n"
                f"  Patterns: {', '.join(sig.patterns_found)}\n"
                f"  Evidence quotes: {json.dumps(sig.evidence_quotes[:3])}\n"
                f"  Counter-evidence: {json.dumps(sig.counter_evidence)}\n"
                f"  Contradictions: {json.dumps(sig.contradictions_internal)}\n"
                f"  Structured data: {json.dumps(sig.structured_data, default=str)}\n"
            )

        # Build conflict context
        conflict_text = ""
        domain_conflicts = [c for c in conflicts
                           if c.get("domain") == domain_name or c.get("domain") == "cross_domain"]
        if domain_conflicts:
            conflict_text = "\n\nFLAGGED CONFLICTS:\n" + json.dumps(domain_conflicts, indent=2)

        context = "\n---\n".join(signal_summaries) + conflict_text

        system_prompt = (
            f"You are a senior psychological analyst synthesizing multiple "
            f"sub-analyses of the '{domain_name}' domain for the person \"{target_speaker}\". "
            f"Your job is to unify the findings, resolve contradictions, and produce "
            f"a coherent domain-level assessment. Return only valid JSON."
        )

        user_prompt = f"""Here are the Layer 1 micro-agent analyses for the {domain_name} domain:

{context}

Synthesize these into a single unified domain assessment. Address any conflicts between agents.

Return this exact JSON structure:
{{
  "domain": "{domain_name}",
  "domain_summary": "3-5 sentence narrative synthesis of all findings in this domain",
  "key_findings": [
    "Finding 1: most important insight",
    "Finding 2: second insight",
    "Finding 3: third insight"
  ],
  "confidence": 0.0-1.0,
  "internal_consistency": "HIGH | MODERATE | LOW",
  "conflict_resolution": "How conflicts between agents were resolved (if any)",
  "risk_flags": ["any concerning patterns worth highlighting"],
  "strength_flags": ["any positive patterns worth highlighting"],
  "agent_agreement_map": {{
    "agent_id": {{"agrees_with": ["ids"], "rating": 0.0}}
  }}
}}"""

        try:
            response = self.client.messages.create(
                model=self.MODEL,
                max_tokens=self.MAX_TOKENS,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            raw = response.content[0].text
            clean = re.sub(r"```(?:json)?|```", "", raw).strip()
            return json.loads(clean)
        except Exception as e:
            return {
                "domain": domain_name,
                "domain_summary": f"Synthesis failed: {str(e)}",
                "key_findings": [],
                "confidence": 0.0,
                "internal_consistency": "ERROR",
            }


class Layer2Runner:
    """
    Runs domain synthesis across all 6 domain groups.
    """

    DOMAIN_LABELS = {
        "biology": "Biological Foundation (Stress & Energy)",
        "perception_cognition": "Perception & Cognitive Processing",
        "emotional": "Emotional Landscape",
        "cognitive_beliefs": "Cognition, Beliefs & Worldview",
        "motivation_behavior": "Motivation & Behavioral Patterns",
        "social_identity": "Social Dynamics & Identity",
    }

    def __init__(self, client: Optional[Anthropic] = None):
        self.synthesizer = DomainSynthesizer(client)

    def run(
        self,
        grouped_signals: Dict[str, List[MicroSignal]],
        conflicts: List[Dict],
        target_speaker: str,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run synthesis for all domains.
        Returns: {domain_name → synthesis_dict}
        """
        results = {}

        for domain, signals in grouped_signals.items():
            label = self.DOMAIN_LABELS.get(domain, domain)
            print(f"[PSYCHE L2] Synthesizing domain: {label} ({len(signals)} signals)")
            results[domain] = self.synthesizer.synthesize(
                domain_name=label,
                signals=signals,
                conflicts=conflicts,
                target_speaker=target_speaker,
            )

        print(f"[PSYCHE L2] All {len(results)} domains synthesized.")
        return results

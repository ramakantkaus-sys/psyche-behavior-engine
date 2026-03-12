"""
PSYCHE OS — Debate Node
Resolves conflicts between agents when they disagree.
Uses an LLM to mediate between conflicting MicroSignals
and produce a resolved assessment.
"""

from __future__ import annotations
import json
import re
from typing import Any, Dict, List, Optional

from anthropic import Anthropic
from psyche_schemas import MicroSignal


class DebateNode:
    """
    Mediates conflicts between L1 agents.
    When two agents in the same domain produce significantly different
    ratings or contradictory labels, this node:
    1. Examines both agents' evidence
    2. Weighs the quality and quantity of evidence
    3. Produces a resolution with reasoning
    """

    MODEL = "claude-haiku-4-5-20251001"
    MAX_TOKENS = 2000

    def __init__(self, client: Optional[Anthropic] = None):
        self.client = client or Anthropic()

    def resolve_conflict(
        self,
        conflict: Dict,
        signals: Dict[str, MicroSignal],
        target_speaker: str,
    ) -> Dict[str, Any]:
        """
        Resolve a single conflict between two agents.

        Args:
            conflict: Conflict dict from Layer1Runner.detect_conflicts()
            signals: Full signals dict {agent_id → MicroSignal}
            target_speaker: Name of the person being analyzed

        Returns:
            Resolution dict with mediated rating, reasoning, and winner
        """
        agent_a_id = conflict["agent_a"]
        agent_b_id = conflict["agent_b"]

        if agent_a_id not in signals or agent_b_id not in signals:
            return {
                "conflict_type": conflict["type"],
                "resolution": "SKIPPED",
                "reasoning": "One or both agents not found in signals.",
            }

        sig_a = signals[agent_a_id]
        sig_b = signals[agent_b_id]

        # Build debate context
        context = f"""CONFLICT IN DOMAIN: {conflict.get('domain', 'unknown')}

AGENT A: {sig_a.agent_name} ({sig_a.agent_id})
  Rating: {sig_a.rating:.2f} | Confidence: {sig_a.confidence:.2f}
  Label: {sig_a.label}
  Summary: {sig_a.summary}
  Evidence ({len(sig_a.evidence_quotes)} quotes): {json.dumps(sig_a.evidence_quotes[:5])}
  Counter-evidence: {json.dumps(sig_a.counter_evidence)}
  Patterns: {', '.join(sig_a.patterns_found)}
  Low evidence warning: {sig_a.low_evidence_warning}

AGENT B: {sig_b.agent_name} ({sig_b.agent_id})
  Rating: {sig_b.rating:.2f} | Confidence: {sig_b.confidence:.2f}
  Label: {sig_b.label}
  Summary: {sig_b.summary}
  Evidence ({len(sig_b.evidence_quotes)} quotes): {json.dumps(sig_b.evidence_quotes[:5])}
  Counter-evidence: {json.dumps(sig_b.counter_evidence)}
  Patterns: {', '.join(sig_b.patterns_found)}
  Low evidence warning: {sig_b.low_evidence_warning}

RATING DIFFERENCE: {abs(sig_a.rating - sig_b.rating):.2f}
"""

        if conflict.get("note"):
            context += f"\nNOTE: {conflict['note']}\n"

        system_prompt = (
            "You are a senior psychometrician mediating a disagreement between "
            "two specialized psychological analysts. Examine the evidence quality "
            "from both sides and produce a fair resolution. Return only valid JSON."
        )

        user_prompt = f"""{context}

Resolve this conflict. Consider:
1. Which agent has STRONGER evidence (more quotes, more specific)?
2. Which agent has HIGHER confidence?
3. Are they measuring slightly different things that could both be right?
4. Could both be partially correct (e.g., ambivalent patterns)?

Return this exact JSON:
{{
  "conflict_type": "{conflict['type']}",
  "domain": "{conflict.get('domain', 'unknown')}",
  "resolution": "AGENT_A_WINS | AGENT_B_WINS | COMPROMISE | BOTH_VALID",
  "mediated_rating": 0.0-1.0,
  "mediated_confidence": 0.0-1.0,
  "reasoning": "2-3 sentences explaining the resolution",
  "winning_agent": "agent_id of the more credible agent (or null for compromise)",
  "evidence_quality_a": "WEAK | MODERATE | STRONG",
  "evidence_quality_b": "WEAK | MODERATE | STRONG",
  "both_valid_explanation": "Explanation if both can coexist (null otherwise)"
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
                "conflict_type": conflict["type"],
                "resolution": "ERROR",
                "reasoning": f"Debate resolution failed: {str(e)}",
            }

    def resolve_all_conflicts(
        self,
        conflicts: List[Dict],
        signals: Dict[str, MicroSignal],
        target_speaker: str,
    ) -> List[Dict[str, Any]]:
        """
        Resolve all flagged conflicts that need debate.
        Only processes conflicts where 'needs_debate' is True.
        """
        debate_conflicts = [c for c in conflicts if c.get("needs_debate", False)]

        if not debate_conflicts:
            print("[DEBATE] No conflicts requiring debate.")
            return []

        print(f"[DEBATE] Resolving {len(debate_conflicts)} conflicts...")
        resolutions = []

        for conflict in debate_conflicts:
            print(f"  → Debating: {conflict['agent_a']} vs {conflict['agent_b']}")
            resolution = self.resolve_conflict(conflict, signals, target_speaker)
            resolutions.append(resolution)
            print(f"    Result: {resolution.get('resolution', 'UNKNOWN')}")

        return resolutions

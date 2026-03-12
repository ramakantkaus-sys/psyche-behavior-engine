"""
PSYCHE OS — Layer 3: Final Report Generator
Produces a comprehensive psychological profile by synthesizing
all Layer 2 domain outputs into a single coherent report.
"""

from __future__ import annotations
import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from anthropic import Anthropic
from psyche_schemas import MicroSignal, StatProfile


class FinalReportGenerator:
    """
    The final synthesis layer. Takes all domain syntheses from Layer 2,
    debate resolutions, and the statistical profile to produce a
    comprehensive psychological profile report.
    """

    MODEL = "claude-sonnet-4-20250514"  # Sonnet for the final synthesis — quality matters
    MAX_TOKENS = 4000

    def __init__(self, client: Optional[Anthropic] = None):
        self.client = client or Anthropic()

    def generate(
        self,
        domain_syntheses: Dict[str, Dict[str, Any]],
        debate_resolutions: List[Dict[str, Any]],
        stat_profile: StatProfile,
        all_signals: Dict[str, MicroSignal],
        target_speaker: str,
    ) -> Dict[str, Any]:
        """
        Generate the complete psychological profile.

        Returns a structured dict with:
        - executive_summary: Quick overview
        - psychological_profile: Deep analysis
        - risk_areas: Flagged concerns
        - strengths: Identified positive patterns
        - growth_areas: Potential development opportunities
        - behavioral_predictions: Predicted behaviors based on patterns
        """

        # Build comprehensive context
        domain_block = ""
        for domain, synthesis in domain_syntheses.items():
            domain_block += f"\n{'='*50}\nDOMAIN: {domain}\n{'='*50}\n"
            domain_block += json.dumps(synthesis, indent=2, default=str)
            domain_block += "\n"

        # Build debate resolutions block
        debate_block = ""
        if debate_resolutions:
            debate_block = "\nDEBATE RESOLUTIONS:\n" + json.dumps(debate_resolutions, indent=2, default=str)

        # Build stats block
        stats_block = f"""
STATISTICAL PROFILE:
  Total messages analyzed: {stat_profile.total_messages}
  Total words: {stat_profile.total_words}
  Average message length: {stat_profile.avg_message_length:.1f} words
  Question ratio: {stat_profile.question_ratio:.1%}
  First-person ratio: {stat_profile.first_person_ratio:.1%}
  Positive word ratio: {stat_profile.positive_word_ratio:.1%}
  Negative word ratio: {stat_profile.negative_word_ratio:.1%}
  Vocabulary size: {stat_profile.vocabulary_size} unique words
  Data spans: {stat_profile.time_span_days} days
  Most discussed topics: {', '.join(stat_profile.most_common_topics[:10])}
"""

        # Build top-level signal summary
        signal_summary = "\nL1 SIGNAL OVERVIEW:\n"
        for sig_id, sig in sorted(all_signals.items()):
            signal_summary += (
                f"  {sig.agent_name}: {sig.label} "
                f"(rating={sig.rating:.2f}, confidence={sig.confidence:.2f})\n"
            )

        system_prompt = (
            f"You are the chief psychologist of the PSYCHE OS system, producing "
            f"the final comprehensive psychological profile for \"{target_speaker}\". "
            f"You have received analyses from 26 specialized agents across 6 domains, "
            f"plus conflict resolutions and statistical data. Synthesize everything "
            f"into a clear, actionable psychological profile. Return only valid JSON."
        )

        user_prompt = f"""FULL ANALYSIS DATA:
{stats_block}
{signal_summary}
{domain_block}
{debate_block}

Produce the FINAL comprehensive psychological profile. This is the capstone synthesis.
Connect insights across domains — show how stress affects emotions, how beliefs drive behavior, etc.

Return this exact JSON structure:
{{
  "executive_summary": "5-7 sentence high-level overview of the person's psychological profile",

  "psychological_profile": {{
    "emotional_foundation": "3-4 sentences on emotional patterns, regulation, and baseline",
    "cognitive_architecture": "3-4 sentences on thinking style, biases, and information processing",
    "belief_system": "3-4 sentences on core beliefs, values, and worldview",
    "motivational_engine": "3-4 sentences on what drives them, needs, persistence",
    "social_wiring": "3-4 sentences on attachment, communication, and social influence",
    "identity_narrative": "3-4 sentences on self-concept and life story framing",
    "biological_baseline": "2-3 sentences on stress and energy patterns"
  }},

  "cross_domain_insights": [
    "Insight showing how patterns in one domain explain patterns in another"
  ],

  "risk_areas": [
    {{
      "area": "name of risk",
      "severity": "LOW | MODERATE | HIGH | CRITICAL",
      "description": "what the risk is and why it matters",
      "supporting_agents": ["agent_ids that flagged this"]
    }}
  ],

  "strengths": [
    {{
      "strength": "name of strength",
      "description": "how this manifests",
      "supporting_agents": ["agent_ids"]
    }}
  ],

  "growth_areas": [
    {{
      "area": "development opportunity",
      "current_state": "where they are now",
      "potential": "where they could be",
      "recommended_approach": "how to develop this"
    }}
  ],

  "behavioral_predictions": [
    "Prediction about how this person would behave in specific situations"
  ],

  "confidence_assessment": {{
    "overall_confidence": 0.0-1.0,
    "data_quality": "INSUFFICIENT | LOW | MODERATE | GOOD | EXCELLENT",
    "domains_with_strongest_evidence": ["domain1"],
    "domains_needing_more_data": ["domain2"],
    "caveats": ["important limitations to note"]
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
            report = json.loads(clean)
            report["generated_at"] = datetime.now().isoformat()
            report["target_speaker"] = target_speaker
            return report
        except Exception as e:
            return {
                "executive_summary": f"Report generation failed: {str(e)}",
                "generated_at": datetime.now().isoformat(),
                "target_speaker": target_speaker,
                "error": str(e),
            }

    def save_report(
        self,
        report: Dict[str, Any],
        output_dir: str = "reports",
        filename: Optional[str] = None,
    ) -> str:
        """Save the report as a JSON file."""
        os.makedirs(output_dir, exist_ok=True)

        if not filename:
            speaker = report.get("target_speaker", "unknown").replace(" ", "_")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"psyche_report_{speaker}_{timestamp}.json"

        path = os.path.join(output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        print(f"[REPORT] Saved to {path}")
        return path

    def render_markdown(self, report: Dict[str, Any]) -> str:
        """Render the report as a human-readable markdown string."""

        md = []
        md.append(f"# PSYCHE OS — Psychological Profile: {report.get('target_speaker', 'Unknown')}")
        md.append(f"*Generated: {report.get('generated_at', 'N/A')}*\n")

        # Executive summary
        md.append("## Executive Summary")
        md.append(report.get("executive_summary", "N/A"))

        # Psychological profile
        profile = report.get("psychological_profile", {})
        if profile:
            md.append("\n## Psychological Profile")
            sections = {
                "emotional_foundation": "🧠 Emotional Foundation",
                "cognitive_architecture": "⚙️ Cognitive Architecture",
                "belief_system": "🔮 Belief System",
                "motivational_engine": "🔥 Motivational Engine",
                "social_wiring": "🤝 Social Wiring",
                "identity_narrative": "🪞 Identity & Narrative",
                "biological_baseline": "🫀 Biological Baseline",
            }
            for key, label in sections.items():
                if key in profile:
                    md.append(f"\n### {label}")
                    md.append(profile[key])

        # Cross-domain insights
        insights = report.get("cross_domain_insights", [])
        if insights:
            md.append("\n## Cross-Domain Insights")
            for insight in insights:
                md.append(f"- {insight}")

        # Risk areas
        risks = report.get("risk_areas", [])
        if risks:
            md.append("\n## ⚠️ Risk Areas")
            for risk in risks:
                severity = risk.get("severity", "UNKNOWN")
                md.append(f"\n### [{severity}] {risk.get('area', 'Unknown')}")
                md.append(risk.get("description", ""))

        # Strengths
        strengths = report.get("strengths", [])
        if strengths:
            md.append("\n## 💪 Strengths")
            for s in strengths:
                md.append(f"\n### {s.get('strength', 'Unknown')}")
                md.append(s.get("description", ""))

        # Growth areas
        growth = report.get("growth_areas", [])
        if growth:
            md.append("\n## 🌱 Growth Areas")
            for g in growth:
                md.append(f"\n### {g.get('area', 'Unknown')}")
                md.append(f"**Current:** {g.get('current_state', 'N/A')}")
                md.append(f"**Potential:** {g.get('potential', 'N/A')}")
                md.append(f"**Approach:** {g.get('recommended_approach', 'N/A')}")

        # Predictions
        predictions = report.get("behavioral_predictions", [])
        if predictions:
            md.append("\n## 🔮 Behavioral Predictions")
            for p in predictions:
                md.append(f"- {p}")

        # Confidence
        conf = report.get("confidence_assessment", {})
        if conf:
            md.append("\n## 📊 Confidence Assessment")
            md.append(f"- **Overall confidence:** {conf.get('overall_confidence', 'N/A')}")
            md.append(f"- **Data quality:** {conf.get('data_quality', 'N/A')}")
            caveats = conf.get("caveats", [])
            if caveats:
                md.append("\n**Caveats:**")
                for c in caveats:
                    md.append(f"- {c}")

        return "\n".join(md)

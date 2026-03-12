"""
PSYCHE OS — Layer 1 Parallel Runner
Executes all 26 micro agents concurrently using asyncio + ThreadPoolExecutor.
Handles timeouts, retries, and collects results into a structured payload.
"""

from __future__ import annotations
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

from schemas.models import Chunk, MicroSignal, StatProfile
from agents.layer1.all_agents import build_all_agents, DOMAIN_GROUPS


# ─────────────────────────────────────────────
# PARALLEL EXECUTION ENGINE
# ─────────────────────────────────────────────

class Layer1Runner:
    """
    Runs all 26 micro agents in parallel.
    Uses ThreadPoolExecutor because Anthropic SDK calls are blocking I/O.

    Config:
      max_workers: number of concurrent threads (default=26, one per agent)
      timeout_seconds: per-agent timeout before marking as failed
      retry_count: how many times to retry on API error
    """

    def __init__(
        self,
        vector_store,
        client=None,
        max_workers: int = 26,
        timeout_seconds: int = 60,
        retry_count: int = 1
    ):
        self.agents = build_all_agents(vector_store, client)
        self.max_workers = max_workers
        self.timeout_seconds = timeout_seconds
        self.retry_count = retry_count

    def _run_single_agent(
        self,
        agent_id: str,
        all_chunks: List[Chunk],
        stat_profile: StatProfile,
        target_speaker: str
    ) -> Tuple[str, MicroSignal]:
        """
        Run a single agent with retry logic.
        Returns (agent_id, signal) tuple.
        """
        agent = self.agents[agent_id]
        last_error = None

        for attempt in range(self.retry_count + 1):
            try:
                start = time.time()
                signal = agent.run(all_chunks, stat_profile, target_speaker)
                elapsed = time.time() - start
                signal.processing_notes = f"Completed in {elapsed:.1f}s"
                return (agent_id, signal)
            except Exception as e:
                last_error = e
                if attempt < self.retry_count:
                    time.sleep(2 ** attempt)  # exponential backoff
                continue

        # All retries failed — return a failure signal
        from schemas.models import MicroSignal
        return (agent_id, MicroSignal(
            agent_id=agent_id,
            agent_name=self.agents[agent_id].AGENT_NAME,
            dimension=self.agents[agent_id].DIMENSION,
            rating=0.0,
            confidence=0.0,
            label="AGENT_FAILED",
            summary=f"Agent failed after {self.retry_count + 1} attempts: {last_error}",
            low_evidence_warning=True,
            flags_for_debate=True,
            processing_notes=f"Error: {str(last_error)}"
        ))

    def run_all(
        self,
        all_chunks: List[Chunk],
        stat_profile: StatProfile,
        target_speaker: str
    ) -> Dict[str, MicroSignal]:
        """
        Execute all 26 agents in parallel.
        Returns dict: {agent_id → MicroSignal}
        """
        agent_ids = list(self.agents.keys())
        results: Dict[str, MicroSignal] = {}
        failed: List[str] = []

        print(f"[PSYCHE Layer1] Launching {len(agent_ids)} agents in parallel...")
        start_total = time.time()

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_id = {
                executor.submit(
                    self._run_single_agent,
                    agent_id,
                    all_chunks,
                    stat_profile,
                    target_speaker
                ): agent_id
                for agent_id in agent_ids
            }

            for future in as_completed(future_to_id, timeout=self.timeout_seconds):
                agent_id = future_to_id[future]
                try:
                    aid, signal = future.result()
                    results[aid] = signal
                    status = "✓" if signal.label != "AGENT_FAILED" else "✗"
                    print(f"  {status} {aid} → {signal.label} (conf: {signal.confidence:.2f})")
                except Exception as e:
                    failed.append(agent_id)
                    print(f"  ✗ {agent_id} → TIMEOUT/ERROR: {e}")

        elapsed = time.time() - start_total
        print(f"\n[PSYCHE Layer1] Complete: {len(results)} agents done in {elapsed:.1f}s")
        if failed:
            print(f"[PSYCHE Layer1] Failed agents: {failed}")

        return results

    def group_by_domain(
        self, signals: Dict[str, MicroSignal]
    ) -> Dict[str, List[MicroSignal]]:
        """
        Group the MicroSignal results by domain for Layer 2 consumption.
        Returns: {domain_name → [MicroSignal, ...]}
        """
        grouped: Dict[str, List[MicroSignal]] = {}

        for domain, agent_ids in DOMAIN_GROUPS.items():
            grouped[domain] = [
                signals[aid] for aid in agent_ids
                if aid in signals
            ]

        return grouped

    def detect_conflicts(
        self, signals: Dict[str, MicroSignal]
    ) -> List[Dict]:
        """
        Scan for conflicts WITHIN each domain group.
        A conflict is when two agents in the same domain have:
        - Contradictory labels (e.g., one says HIGH STRESS, another LOW)
        - Rating difference > 0.30

        Returns list of conflict dicts for the debate node.
        """
        conflicts = []

        # Domain-level conflict check
        grouped = self.group_by_domain(signals)

        for domain, domain_signals in grouped.items():
            if len(domain_signals) < 2:
                continue

            ratings = [s.rating for s in domain_signals]
            max_diff = max(ratings) - min(ratings)

            if max_diff > 0.30:
                high_agent = max(domain_signals, key=lambda s: s.rating)
                low_agent = min(domain_signals, key=lambda s: s.rating)
                conflicts.append({
                    "domain": domain,
                    "type": "rating_conflict",
                    "agent_a": high_agent.agent_id,
                    "agent_b": low_agent.agent_id,
                    "rating_a": high_agent.rating,
                    "rating_b": low_agent.rating,
                    "diff": max_diff,
                    "label_a": high_agent.label,
                    "label_b": low_agent.label,
                    "needs_debate": True
                })

        # Cross-domain logical conflicts
        # Example: Stress agent says HIGH STRESS but Energy agent says HIGH VITALITY
        # These are logically unusual combinations worth flagging
        cross_domain_checks = [
            ("L1-01-stress", "L1-02-energy",
             "High stress + high energy: unusual combination, check for burnout cycle"),
            ("L1-09-regulation", "L1-06-system1",
             "Check regulation vs impulsivity alignment"),
            ("L1-14-core-beliefs", "L1-25-self-concept",
             "Core beliefs should align with self-concept"),
        ]

        for id_a, id_b, note in cross_domain_checks:
            if id_a in signals and id_b in signals:
                a, b = signals[id_a], signals[id_b]
                if abs(a.rating - b.rating) > 0.40:
                    conflicts.append({
                        "domain": "cross_domain",
                        "type": "logical_inconsistency",
                        "agent_a": id_a,
                        "agent_b": id_b,
                        "rating_a": a.rating,
                        "rating_b": b.rating,
                        "note": note,
                        "needs_debate": False  # flag, not debate
                    })

        print(f"[PSYCHE Layer1] Detected {len(conflicts)} conflicts")
        return conflicts


# ─────────────────────────────────────────────
# LAYER 1 STATE SUMMARY
# ─────────────────────────────────────────────

class Layer1Output:
    """
    Structured container for all Layer 1 results.
    Passed to the debate node and then to Layer 2.
    """

    def __init__(
        self,
        signals: Dict[str, MicroSignal],
        grouped: Dict[str, List[MicroSignal]],
        conflicts: List[Dict],
        stat_profile: StatProfile,
        target_speaker: str
    ):
        self.signals = signals
        self.grouped = grouped
        self.conflicts = conflicts
        self.stat_profile = stat_profile
        self.target_speaker = target_speaker

    def summary(self) -> str:
        total = len(self.signals)
        failed = sum(1 for s in self.signals.values() if s.label == "AGENT_FAILED")
        low_conf = sum(1 for s in self.signals.values() if s.confidence < 0.4)
        n_conflicts = len([c for c in self.conflicts if c.get("needs_debate")])

        return (
            f"Layer 1 Summary:\n"
            f"  Agents run: {total}\n"
            f"  Failed: {failed}\n"
            f"  Low confidence: {low_conf}\n"
            f"  Conflicts requiring debate: {n_conflicts}\n"
            f"  Domains ready: {list(self.grouped.keys())}"
        )

    def to_dict(self) -> dict:
        return {
            "signals": {k: v.model_dump() for k, v in self.signals.items()},
            "conflicts": self.conflicts,
            "domain_groups": {
                domain: [s.model_dump() for s in sigs]
                for domain, sigs in self.grouped.items()
            }
        }

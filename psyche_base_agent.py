"""
PSYCHE OS — Base Micro Agent
Every L1 agent inherits from this. Handles RAG retrieval, LLM calls,
output parsing, and quality checks.
"""

from __future__ import annotations
import json
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from anthropic import Anthropic
from schemas.models import Chunk, MicroSignal, StatProfile


# ─────────────────────────────────────────────
# BASE CLASS
# ─────────────────────────────────────────────

class MicroAgent(ABC):
    """
    Abstract base for all 26 Layer-1 agents.

    Each subclass must define:
      - AGENT_ID        : unique identifier e.g. "L1-01-stress"
      - AGENT_NAME      : human readable name
      - DIMENSION       : psychological dimension owned
      - RAG_QUERIES     : list of semantic queries to retrieve relevant chunks
      - TOP_K           : how many chunks to retrieve per query
      - SYSTEM_PROMPT   : the agent's specialized system prompt
      - OUTPUT_SCHEMA   : description of structured_data expected
    """

    AGENT_ID: str = ""
    AGENT_NAME: str = ""
    DIMENSION: str = ""
    RAG_QUERIES: List[str] = []
    TOP_K: int = 8
    MODEL: str = "claude-haiku-4-5-20251001"   # Haiku for all L1 agents — cost efficient
    MAX_TOKENS: int = 2000

    def __init__(self, vector_store, client: Optional[Anthropic] = None):
        self.vector_store = vector_store
        self.client = client or Anthropic()

    # ─── RAG RETRIEVAL ───────────────────────────────────────────────

    def retrieve_chunks(self, all_chunks: List[Chunk]) -> List[Chunk]:
        """
        Fire all RAG_QUERIES against the vector store.
        De-duplicate results. Return top-K most relevant chunks.
        This ensures each agent reads only the parts of the chat
        most relevant to its specific psychological dimension.
        """
        if not self.vector_store:
            # Fallback: return all chunks (for testing without vector store)
            return all_chunks[:self.TOP_K * len(self.RAG_QUERIES)]

        seen_ids = set()
        retrieved: List[Chunk] = []

        for query in self.RAG_QUERIES:
            results = self.vector_store.similarity_search(
                query=query,
                k=self.TOP_K
            )
            for chunk in results:
                if chunk.chunk_id not in seen_ids:
                    seen_ids.add(chunk.chunk_id)
                    retrieved.append(chunk)

        return retrieved

    # ─── CONTEXT BUILDER ─────────────────────────────────────────────

    def build_context(
        self,
        chunks: List[Chunk],
        stat_profile: StatProfile,
        target_speaker: str
    ) -> str:
        """
        Formats the retrieved chunks + statistical profile into
        a clean context string for the LLM prompt.
        """
        stats_block = f"""
=== STATISTICAL PROFILE ===
Total messages: {stat_profile.total_messages}
Average message length: {stat_profile.avg_message_length:.1f} words
Question ratio: {stat_profile.question_ratio:.1%}
First-person ratio: {stat_profile.first_person_ratio:.1%}
Positive word ratio: {stat_profile.positive_word_ratio:.1%}
Negative word ratio: {stat_profile.negative_word_ratio:.1%}
Vocabulary size: {stat_profile.vocabulary_size} unique words
Data spans: {stat_profile.time_span_days} days
"""

        chunks_block = "\n\n".join([
            f"--- CHUNK {i+1} [{c.chunk_id}] | "
            f"Topic: {c.topic} | Tone: {c.emotional_tone} | "
            f"Type: {c.conversation_type} | Period: {c.time_period} ---\n{c.text}"
            for i, c in enumerate(chunks)
        ])

        return f"{stats_block}\n=== RETRIEVED CONVERSATION CHUNKS ===\n\n{chunks_block}"

    # ─── PROMPT BUILDER ──────────────────────────────────────────────

    def build_prompt(
        self,
        context: str,
        target_speaker: str,
        stat_profile: StatProfile
    ) -> str:
        """
        Combines system context with agent-specific analysis instructions.
        Subclasses define ANALYSIS_PROMPT which is appended here.
        """
        return f"""You are analyzing the chat messages of a specific person: "{target_speaker}".

{context}

---

{self.ANALYSIS_PROMPT}

CRITICAL RULES:
1. Base EVERY claim on specific evidence from the chunks above. Quote directly.
2. Do NOT hallucinate — if you don't have enough evidence, say so and set confidence low.
3. Look for PATTERNS across multiple chunks, not single instances.
4. Distinguish between what is said explicitly vs what is implied by language patterns.
5. Note contradictions — the person may behave differently in different contexts.
6. Return ONLY valid JSON matching the schema below. No preamble. No explanation outside JSON.

{self.OUTPUT_INSTRUCTIONS}"""

    # ─── ABSTRACT PROPERTIES ─────────────────────────────────────────

    @property
    @abstractmethod
    def ANALYSIS_PROMPT(self) -> str:
        """Agent-specific analysis instructions injected into main prompt."""
        ...

    @property
    @abstractmethod
    def OUTPUT_INSTRUCTIONS(self) -> str:
        """JSON schema instructions injected at end of prompt."""
        ...

    @abstractmethod
    def parse_structured_data(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """
        Agent-specific parsing of the structured_data field.
        Called after base MicroSignal is parsed.
        """
        ...

    # ─── LLM CALL ────────────────────────────────────────────────────

    def call_llm(self, system: str, user: str) -> str:
        """Single LLM call. Returns raw text response."""
        response = self.client.messages.create(
            model=self.MODEL,
            max_tokens=self.MAX_TOKENS,
            system=system,
            messages=[{"role": "user", "content": user}]
        )
        return response.content[0].text

    # ─── OUTPUT PARSER ───────────────────────────────────────────────

    def parse_output(self, raw_text: str, chunks: List[Chunk]) -> MicroSignal:
        """
        Parses LLM JSON output into a validated MicroSignal.
        Handles malformed JSON gracefully.
        """
        try:
            # Strip markdown fences if present
            clean = re.sub(r"```(?:json)?|```", "", raw_text).strip()
            data = json.loads(clean)

            signal = MicroSignal(
                agent_id=self.AGENT_ID,
                agent_name=self.AGENT_NAME,
                dimension=self.DIMENSION,
                rating=float(data.get("rating", 0.5)),
                confidence=float(data.get("confidence", 0.3)),
                label=data.get("label", "UNKNOWN"),
                summary=data.get("summary", ""),
                evidence_quotes=data.get("evidence_quotes", []),
                evidence_chunk_ids=data.get("evidence_chunk_ids", []),
                patterns_found=data.get("patterns_found", []),
                counter_evidence=data.get("counter_evidence", []),
                contradictions_internal=data.get("contradictions_internal", []),
                structured_data=self.parse_structured_data(data.get("structured_data", {})),
                chunks_analyzed=len(chunks),
                chunks_retrieved=len(chunks),
            )

            # Auto-quality flags
            signal.low_evidence_warning = len(signal.evidence_quotes) < 3
            signal.flags_for_debate = signal.confidence < 0.4
            signal.needs_more_data = signal.chunks_analyzed < 5

            return signal

        except (json.JSONDecodeError, Exception) as e:
            # Return a low-confidence signal rather than crashing
            return MicroSignal(
                agent_id=self.AGENT_ID,
                agent_name=self.AGENT_NAME,
                dimension=self.DIMENSION,
                rating=0.0,
                confidence=0.0,
                label="PARSE_ERROR",
                summary=f"Failed to parse agent output: {str(e)}",
                low_evidence_warning=True,
                flags_for_debate=True,
                processing_notes=f"Raw output: {raw_text[:500]}"
            )

    # ─── MAIN ENTRY POINT ────────────────────────────────────────────

    def run(
        self,
        all_chunks: List[Chunk],
        stat_profile: StatProfile,
        target_speaker: str
    ) -> MicroSignal:
        """
        Full agent execution:
        1. Retrieve relevant chunks via RAG
        2. Build context
        3. Build prompt
        4. Call LLM
        5. Parse and validate output
        6. Return MicroSignal
        """
        # Step 1: RAG retrieval
        relevant_chunks = self.retrieve_chunks(all_chunks)

        # Step 2: Build context string
        context = self.build_context(relevant_chunks, stat_profile, target_speaker)

        # Step 3: Build full prompt
        prompt = self.build_prompt(context, target_speaker, stat_profile)

        # Step 4: LLM call
        system_prompt = (
            f"You are a specialized psychological analyst. "
            f"Your sole focus is: {self.DIMENSION}. "
            f"You produce precise, evidence-based analysis. "
            f"You never fabricate. You always return valid JSON."
        )
        raw_output = self.call_llm(system=system_prompt, user=prompt)

        # Step 5: Parse
        signal = self.parse_output(raw_output, relevant_chunks)

        return signal

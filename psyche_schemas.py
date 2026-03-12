"""
PSYCHE OS — Core Data Models
All Pydantic schemas shared across the entire agent system.
"""

from __future__ import annotations
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
# INPUT LAYER
# ─────────────────────────────────────────────

class Message(BaseModel):
    id: str
    timestamp: Optional[datetime] = None
    speaker: str
    text: str
    reply_to: Optional[str] = None
    source: Literal["whatsapp", "chatgpt", "generic"] = "generic"
    word_count: int = 0
    char_count: int = 0


class Chunk(BaseModel):
    chunk_id: str
    messages: List[Message]
    text: str                          # full concatenated text for LLM
    topic: str = "unknown"
    emotional_tone: Literal["positive", "negative", "neutral", "mixed"] = "neutral"
    conversation_type: Literal["debate", "vent", "plan", "question", "casual", "reflection"] = "casual"
    time_period: Literal["early", "middle", "recent"] = "middle"
    target_speaker_ratio: float = 1.0  # % of words from the person being analyzed
    word_count: int = 0
    embedding: Optional[List[float]] = None


class StatProfile(BaseModel):
    """Basic statistical fingerprint of the entire chat corpus."""
    total_messages: int
    total_words: int
    avg_message_length: float
    question_ratio: float              # % messages ending with ?
    exclamation_ratio: float           # % messages with !
    first_person_ratio: float          # I / me / my frequency
    negative_word_ratio: float
    positive_word_ratio: float
    avg_response_time_minutes: Optional[float] = None
    most_common_topics: List[str] = []
    vocabulary_size: int = 0
    unique_words: int = 0
    time_span_days: int = 0
    message_frequency_by_hour: Dict[str, int] = {}


# ─────────────────────────────────────────────
# MICRO AGENT OUTPUT — Base Signal
# ─────────────────────────────────────────────

class MicroSignal(BaseModel):
    """
    Standardized output contract every L1 micro agent must return.
    Never return free-form text — always this structure.
    """
    agent_id: str                       # e.g. "L1-01-stress"
    agent_name: str
    dimension: str                      # psychological dimension owned
    layer: Literal["L1"] = "L1"

    # Core finding
    rating: float = Field(ge=0.0, le=1.0)       # 0 = absent/low, 1 = very strong
    confidence: float = Field(ge=0.0, le=1.0)   # how much evidence supports this
    label: str                          # e.g. "HIGH CHRONIC STRESS"
    summary: str                        # 2-3 sentence finding summary

    # Evidence
    evidence_quotes: List[str] = []     # exact quotes from chat supporting this
    evidence_chunk_ids: List[str] = []  # which chunks were used
    patterns_found: List[str] = []      # named patterns detected
    counter_evidence: List[str] = []    # quotes that contradict the finding

    # Structured output (agent-specific, varies per agent)
    structured_data: Dict[str, Any] = {}

    # Quality flags
    contradictions_internal: List[str] = []  # self-contradictions within the data
    low_evidence_warning: bool = False        # True if < 3 supporting quotes found
    needs_more_data: bool = False             # True if corpus is too small
    flags_for_debate: bool = False            # True if confidence < 0.4

    # Meta
    chunks_analyzed: int = 0
    chunks_retrieved: int = 0
    processing_notes: str = ""


# ─────────────────────────────────────────────
# SPECIFIC STRUCTURED OUTPUTS PER AGENT
# (nested inside structured_data field)
# ─────────────────────────────────────────────

class StressProfile(BaseModel):
    baseline_stress_level: Literal["LOW", "MODERATE", "HIGH", "SEVERE"]
    is_chronic: bool
    trigger_topics: List[str] = []
    coping_mechanisms_observed: List[str] = []
    stress_language_frequency: float  # per 100 messages
    recovery_signals: List[str] = []


class BiasEntry(BaseModel):
    bias_name: str
    description: str
    severity: Literal["MILD", "MODERATE", "STRONG"]
    evidence: List[str] = []
    frequency: int = 0


class CoreBeliefEntry(BaseModel):
    belief_statement: str              # "I am not good enough"
    belief_type: Literal["self", "others", "world", "future"]
    valence: Literal["limiting", "empowering", "neutral"]
    confidence: float
    evidence: List[str] = []
    frequency: int = 0                 # how many times implied in corpus


class HabitLoop(BaseModel):
    cue: str
    routine: str
    reward: str
    frequency: Literal["rare", "occasional", "frequent", "constant"]
    evidence: List[str] = []


class EmotionalTheme(BaseModel):
    emotion: str
    frequency: float          # 0-1 normalized
    intensity: Literal["LOW", "MEDIUM", "HIGH"]
    triggers: List[str] = []
    evidence: List[str] = []


class ValueEntry(BaseModel):
    value: str
    rank: int
    behavioral_evidence: List[str] = []  # actions, not stated values
    stated_vs_actual_gap: bool = False   # True if they say one thing, do another


class AttachmentProfile(BaseModel):
    primary_style: Literal["SECURE", "ANXIOUS", "AVOIDANT", "DISORGANIZED"]
    confidence: float
    secondary_style: Optional[str] = None
    relationship_patterns: List[str] = []
    intimacy_comfort: float = 0.5     # 0=avoids, 1=seeks
    abandonment_sensitivity: float = 0.5
    trust_baseline: float = 0.5


class NarrativeProfile(BaseModel):
    primary_role: Literal["PROTAGONIST", "VICTIM", "HERO", "OBSERVER", "MIXED"]
    agency_level: float               # 0=external locus, 1=full internal locus
    story_arc: Literal["growth", "decline", "stuck", "cyclical", "unclear"]
    self_compassion_level: float
    blame_attribution: Literal["self", "others", "situation", "mixed"]
    key_story_themes: List[str] = []


class ThinkingStyleProfile(BaseModel):
    dominant_style: Literal["ANALYTICAL", "INTUITIVE", "ABSTRACT", "SYSTEMS", "PRACTICAL"]
    style_distribution: Dict[str, float] = {}  # e.g. {"analytical": 0.6, "intuitive": 0.4}
    complexity_level: float           # 0=simple, 1=highly complex
    abstraction_preference: float     # 0=concrete, 1=abstract
    evidence_requirements: float      # 0=accepts claims easily, 1=demands proof always

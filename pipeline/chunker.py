"""
PSYCHE OS — Conversation Chunker
Splits a list of Messages into topical/temporal Chunks.
Each chunk is a coherent segment of conversation with metadata.
"""

from __future__ import annotations
import re
import uuid
from typing import List, Optional

from psyche_schemas import Message, Chunk


# ─────────────────────────────────────────────
# TONE DETECTION (SIMPLE KEYWORD-BASED)
# ─────────────────────────────────────────────

POSITIVE_MARKERS = {
    "happy", "great", "love", "amazing", "wonderful", "awesome", "excited",
    "proud", "grateful", "thankful", "beautiful", "fantastic", "glad",
    "joyful", "hopeful", "confident", "motivated", "inspired", "fun",
    "enjoy", "brilliant", "perfect", "excellent", "blessed", "relieved",
}

NEGATIVE_MARKERS = {
    "sad", "angry", "frustrated", "depressed", "anxious", "worried",
    "scared", "stressed", "exhausted", "overwhelmed", "hopeless",
    "worthless", "guilty", "ashamed", "miserable", "hate", "terrible",
    "awful", "horrible", "hurt", "painful", "lonely", "confused",
    "disappointed", "failure", "stuck", "trapped", "broken",
}


def _detect_tone(text: str) -> str:
    """Simple keyword-based tone detection."""
    words = set(text.lower().split())
    pos = len(words & POSITIVE_MARKERS)
    neg = len(words & NEGATIVE_MARKERS)

    if pos > 0 and neg > 0:
        return "mixed"
    elif pos > neg:
        return "positive"
    elif neg > pos:
        return "negative"
    return "neutral"


# ─────────────────────────────────────────────
# CONVERSATION TYPE DETECTION
# ─────────────────────────────────────────────

def _detect_conversation_type(text: str) -> str:
    """Simple heuristic to classify conversation type."""
    lower = text.lower()
    question_count = lower.count("?")
    excl_count = lower.count("!")
    word_count = len(lower.split())

    # Question-heavy
    if question_count >= 3 or (question_count / max(word_count, 1)) > 0.05:
        return "question"

    # Planning language
    plan_words = ["plan", "should we", "let's", "going to", "will do", "schedule",
                  "tomorrow", "next week", "meeting", "arrange"]
    if any(w in lower for w in plan_words):
        return "plan"

    # Venting / emotional
    vent_words = ["can't believe", "so frustrated", "i'm so", "ugh", "hate",
                  "fed up", "sick of", "I just need to", "let me rant"]
    if any(w in lower for w in vent_words):
        return "vent"

    # Debate / disagreement
    debate_words = ["disagree", "but actually", "no that's", "wrong", "you're not",
                    "I think you", "on the other hand", "argument"]
    if any(w in lower for w in debate_words):
        return "debate"

    # Reflection
    reflect_words = ["I realized", "looking back", "I've been thinking",
                     "it made me think", "I learned", "in hindsight"]
    if any(w in lower for w in reflect_words):
        return "reflection"

    return "casual"


# ─────────────────────────────────────────────
# TOPIC EXTRACTION (SIMPLE)
# ─────────────────────────────────────────────

STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "and", "but", "or", "not", "so", "this", "that", "it", "its",
    "i", "me", "my", "you", "your", "we", "our", "they", "them", "their",
    "he", "she", "him", "her", "what", "which", "who", "how", "when",
    "where", "why", "if", "than", "then", "just", "also", "about",
    "like", "really", "very", "yeah", "yes", "no", "ok", "okay",
    "don't", "can't", "i'm", "it's", "that's", "i've", "i'll",
}

from collections import Counter

def _extract_topic(text: str) -> str:
    """Extract topic based on most frequent content words."""
    words = re.findall(r"[a-z]+", text.lower())
    content = [w for w in words if w not in STOP_WORDS and len(w) > 3]
    if not content:
        return "general"
    top = Counter(content).most_common(3)
    return ", ".join(w for w, _ in top)


# ─────────────────────────────────────────────
# TIME PERIOD DETECTION
# ─────────────────────────────────────────────

def _assign_time_period(
    chunk_index: int, total_chunks: int
) -> str:
    """Assign early/middle/recent based on chunk position."""
    ratio = chunk_index / max(total_chunks - 1, 1)
    if ratio < 0.33:
        return "early"
    elif ratio < 0.66:
        return "middle"
    return "recent"


# ─────────────────────────────────────────────
# MAIN CHUNKER
# ─────────────────────────────────────────────

def chunk_messages(
    messages: List[Message],
    target_speaker: Optional[str] = None,
    chunk_size: int = 30,
    overlap: int = 5,
) -> List[Chunk]:
    """
    Split messages into overlapping chunks of conversation.

    Args:
        messages: List of Message objects (should be sorted by time)
        target_speaker: If set, compute target_speaker_ratio per chunk
        chunk_size: Number of messages per chunk
        overlap: Number of messages overlap between consecutive chunks
    """
    if not messages:
        return []

    chunks: List[Chunk] = []
    step = max(chunk_size - overlap, 1)
    total_expected = max(1, (len(messages) - overlap) // step)

    for i in range(0, len(messages), step):
        chunk_msgs = messages[i : i + chunk_size]
        if len(chunk_msgs) < 5:  # skip tiny trailing chunks
            break

        # Build concatenated text
        text = "\n".join(
            f"[{m.speaker}]: {m.text}" for m in chunk_msgs
        )

        # Compute target speaker ratio
        if target_speaker:
            target_words = sum(
                m.word_count for m in chunk_msgs
                if m.speaker.lower() == target_speaker.lower()
            )
            total_words = sum(m.word_count for m in chunk_msgs)
            ratio = target_words / max(total_words, 1)
        else:
            ratio = 1.0

        chunk_index = len(chunks)
        word_count = sum(m.word_count for m in chunk_msgs)

        chunk = Chunk(
            chunk_id=f"chunk-{str(uuid.uuid4())[:8]}",
            messages=chunk_msgs,
            text=text,
            topic=_extract_topic(text),
            emotional_tone=_detect_tone(text),
            conversation_type=_detect_conversation_type(text),
            time_period=_assign_time_period(chunk_index, total_expected),
            target_speaker_ratio=round(ratio, 3),
            word_count=word_count,
        )
        chunks.append(chunk)

    print(f"[CHUNKER] Created {len(chunks)} chunks from {len(messages)} messages "
          f"(size={chunk_size}, overlap={overlap})")
    return chunks

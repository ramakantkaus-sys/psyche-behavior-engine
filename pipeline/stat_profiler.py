"""
PSYCHE OS — Statistical Profiler
Computes a StatProfile from raw Message objects.
Calculates word counts, ratios, vocabulary size, frequencies, etc.
"""

from __future__ import annotations
import re
from collections import Counter
from typing import Dict, List, Optional

from psyche_schemas import Message, StatProfile


# ─────────────────────────────────────────────
# WORD LISTS FOR RATIO COMPUTATION
# ─────────────────────────────────────────────

POSITIVE_WORDS = {
    "happy", "great", "good", "love", "amazing", "wonderful", "awesome",
    "fantastic", "excellent", "beautiful", "brilliant", "grateful", "thankful",
    "excited", "proud", "confident", "optimistic", "hopeful", "joyful",
    "peaceful", "satisfied", "content", "inspired", "motivated", "blessed",
    "lucky", "glad", "delighted", "thrilled", "relieved", "cheerful",
    "kind", "generous", "pleasant", "positive", "nice", "perfect",
    "incredible", "superb", "outstanding", "magnificent", "lovely",
    "enjoy", "enjoyed", "enjoying", "appreciate", "appreciated", "fun",
}

NEGATIVE_WORDS = {
    "sad", "bad", "terrible", "awful", "horrible", "angry", "frustrated",
    "depressed", "anxious", "worried", "scared", "afraid", "lonely",
    "stressed", "exhausted", "tired", "overwhelmed", "hopeless", "helpless",
    "worthless", "guilty", "ashamed", "regret", "miserable", "pathetic",
    "useless", "stupid", "ugly", "hate", "hated", "hating", "disgusting",
    "disappointed", "failure", "failed", "failing", "lost", "confused",
    "uncertain", "doubt", "doubt", "insecure", "jealous", "envious",
    "resentful", "bitter", "hurt", "painful", "suffering", "struggling",
    "stuck", "trapped", "broken", "damaged", "ruined", "destroyed",
    "annoyed", "irritated", "furious", "devastated", "heartbroken",
}

FIRST_PERSON_WORDS = {"i", "me", "my", "mine", "myself", "i'm", "i've", "i'll", "i'd"}


# ─────────────────────────────────────────────
# PROFILER
# ─────────────────────────────────────────────

def compute_stat_profile(
    messages: List[Message],
    target_speaker: Optional[str] = None
) -> StatProfile:
    """
    Compute statistical profile from a list of messages.
    If target_speaker is provided, compute stats only for that speaker's messages
    but use the full corpus for context.
    """
    if target_speaker:
        target_msgs = [m for m in messages if m.speaker.lower() == target_speaker.lower()]
    else:
        target_msgs = messages

    if not target_msgs:
        target_msgs = messages  # fallback to all

    # ── Basic counts ──
    total_messages = len(target_msgs)
    all_words: List[str] = []
    all_words_lower: List[str] = []
    message_lengths: List[int] = []
    question_count = 0
    exclamation_count = 0

    for msg in target_msgs:
        words = msg.text.split()
        word_count = len(words)
        all_words.extend(words)
        all_words_lower.extend(w.lower() for w in words)
        message_lengths.append(word_count)

        if msg.text.strip().endswith("?"):
            question_count += 1
        if "!" in msg.text:
            exclamation_count += 1

    total_words = len(all_words)
    avg_message_length = sum(message_lengths) / max(len(message_lengths), 1)

    # ── Vocabulary ──
    word_freq = Counter(all_words_lower)
    unique_words = len(word_freq)

    # ── Ratios ──
    question_ratio = question_count / max(total_messages, 1)
    exclamation_ratio = exclamation_count / max(total_messages, 1)

    first_person_count = sum(1 for w in all_words_lower if w in FIRST_PERSON_WORDS)
    first_person_ratio = first_person_count / max(total_words, 1)

    positive_count = sum(1 for w in all_words_lower if w in POSITIVE_WORDS)
    negative_count = sum(1 for w in all_words_lower if w in NEGATIVE_WORDS)
    positive_word_ratio = positive_count / max(total_words, 1)
    negative_word_ratio = negative_count / max(total_words, 1)

    # ── Time analysis ──
    timestamps = [m.timestamp for m in target_msgs if m.timestamp]
    time_span_days = 0
    avg_response_time = None
    message_frequency_by_hour: Dict[str, int] = {}

    if timestamps:
        time_span_days = max(1, (max(timestamps) - min(timestamps)).days)

        for ts in timestamps:
            hour = str(ts.hour).zfill(2)
            message_frequency_by_hour[hour] = message_frequency_by_hour.get(hour, 0) + 1

    # ── Response time (average minutes between consecutive target speaker messages) ──
    if len(timestamps) >= 2:
        sorted_ts = sorted(timestamps)
        gaps = []
        for i in range(1, len(sorted_ts)):
            gap = (sorted_ts[i] - sorted_ts[i - 1]).total_seconds() / 60.0
            if gap < 1440:  # ignore gaps > 24h (different sessions)
                gaps.append(gap)
        if gaps:
            avg_response_time = sum(gaps) / len(gaps)

    # ── Most common topics (approximation via most common nouns/phrases) ──
    # Simple: use most common non-stop words as topic proxies
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "to", "of", "in", "for",
        "on", "with", "at", "by", "from", "as", "into", "through", "during",
        "before", "after", "above", "below", "between", "out", "off", "up",
        "down", "and", "but", "or", "nor", "not", "so", "yet", "both",
        "either", "neither", "each", "every", "all", "any", "few", "more",
        "most", "other", "some", "such", "no", "only", "own", "same", "than",
        "too", "very", "just", "about", "also", "then", "this", "that",
        "these", "those", "it", "its", "he", "she", "they", "them", "his",
        "her", "their", "we", "us", "our", "you", "your", "what", "which",
        "who", "whom", "how", "when", "where", "why", "if", "because",
        "while", "although", "though", "even", "like", "really", "yeah",
        "yes", "no", "ok", "okay", "lol", "haha", "hmm", "oh", "hi",
        "hello", "hey", "thanks", "thank", "please", "sorry", "don't",
        "doesn't", "didn't", "won't", "can't", "couldn't", "wouldn't",
        "shouldn't", "haven't", "hasn't", "hadn't", "isn't", "aren't",
        "wasn't", "weren't", "i'm", "i've", "i'll", "i'd", "it's",
        "that's", "there's", "here's", "what's", "who's", "let's",
        "i", "me", "my", "mine", "myself", "got", "get", "thing",
        "things", "going", "know", "think", "want", "need", "say",
        "said", "go", "went", "come", "came", "make", "made", "see",
        "saw", "take", "took", "give", "gave", "tell", "told", "one",
        "two", "much", "many", "still", "back", "now", "well", "way",
    }

    content_words = [w for w in all_words_lower if w not in stop_words and len(w) > 3]
    topic_counter = Counter(content_words)
    most_common_topics = [word for word, _ in topic_counter.most_common(15)]

    return StatProfile(
        total_messages=total_messages,
        total_words=total_words,
        avg_message_length=avg_message_length,
        question_ratio=question_ratio,
        exclamation_ratio=exclamation_ratio,
        first_person_ratio=first_person_ratio,
        negative_word_ratio=negative_word_ratio,
        positive_word_ratio=positive_word_ratio,
        avg_response_time_minutes=avg_response_time,
        most_common_topics=most_common_topics,
        vocabulary_size=unique_words,
        unique_words=unique_words,
        time_span_days=time_span_days,
        message_frequency_by_hour=message_frequency_by_hour,
    )

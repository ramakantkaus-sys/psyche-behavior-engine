"""
PSYCHE OS — All 26 Layer-1 Micro Agents
========================================
Each agent:
  - Owns exactly ONE psychological sub-dimension
  - Retrieves only its relevant chunks via specialized RAG queries
  - Has a precise system prompt tailored to its dimension
  - Returns a fully typed MicroSignal with evidence quotes

AGENT INDEX
───────────────────────────────────────────────────────
BIOLOGY LAYER (2 agents)
  L1-01  StressSignalAgent         — chronic stress & cortisol patterns
  L1-02  EnergyVitalityAgent       — energy levels, fatigue, physical state

PERCEPTION LAYER (3 agents)
  L1-03  AttentionFocusAgent       — detail vs big-picture, what gets noticed
  L1-04  InterpretationBiasAgent   — optimism/pessimism lens, threat sensitivity
  L1-05  ConfirmationBiasAgent     — how info is filtered and sought

DUAL-PROCESS LAYER (2 agents)
  L1-06  AutomaticReactionAgent    — System 1 triggers, impulsive responses
  L1-07  CognitiveBiasMapAgent     — 30+ bias types, identifies which are active

EMOTION LAYER (3 agents)
  L1-08  EmotionalRangeAgent       — vocabulary, range, expression intensity
  L1-09  EmotionalRegulationAgent  — impulse control, reactivity vs calm
  L1-10  EmotionalThemeAgent       — dominant recurring emotional patterns

COGNITION LAYER (3 agents)
  L1-11  ThinkingStyleAgent        — analytical/intuitive/systems ratio
  L1-12  LanguageComplexityAgent   — vocabulary, sophistication, structure
  L1-13  MemoryPatternAgent        — how past is referenced and reconstructed

BELIEFS LAYER (3 agents)
  L1-14  CoreBeliefAgent           — implicit rules about self/others/world
  L1-15  ValuesHierarchyAgent      — actual behavioral priorities
  L1-16  WorldviewAgent            — abundance/scarcity, trust/threat lens

MOTIVATION LAYER (3 agents)
  L1-17  NeedsSignalAgent          — dominant unmet needs (Maslow + SDT)
  L1-18  DriveTypeAgent            — intrinsic vs extrinsic, fear vs growth
  L1-19  PersistenceAgent          — grit, setback response, resilience

BEHAVIOR LAYER (2 agents)
  L1-20  HabitLoopAgent            — cue-routine-reward patterns
  L1-21  DecisionStyleAgent        — impulsive vs deliberate, risk appetite

SOCIAL LAYER (3 agents)
  L1-22  AttachmentStyleAgent      — secure/anxious/avoidant patterns
  L1-23  CommunicationStyleAgent   — assertive/passive/collaborative
  L1-24  SocialInfluenceAgent      — conformity, authority, social proof reliance

IDENTITY LAYER (2 agents)
  L1-25  SelfConceptAgent          — self-image, self-labels, worth signals
  L1-26  NarrativeIdentityAgent    — life story framing, protagonist/victim role
───────────────────────────────────────────────────────
"""

from __future__ import annotations
from typing import Any, Dict, List
from agents.base_agent import MicroAgent


# ══════════════════════════════════════════════════════════════════════
# BIOLOGY LAYER
# ══════════════════════════════════════════════════════════════════════

class StressSignalAgent(MicroAgent):
    """
    L1-01 | Stress Signal Agent
    ──────────────────────────────────────────────────────────────────
    OWNS: Chronic stress detection, cortisol-pattern language,
          pressure overload signals, and recovery patterns.

    WHY THIS MATTERS: Stress is the master variable. It shrinks the
    prefrontal cortex, narrows perception, degrades decision quality,
    and amplifies every negative emotional pattern. Understanding the
    person's stress baseline explains much of their behavior.

    WHAT IT LOOKS FOR:
    - Urgency / panic language ("I need to", "I can't handle", "deadline")
    - Overwhelm declarations ("too much", "I'm drowning", "I can't cope")
    - Physical stress mentions (headaches, can't sleep, exhaustion)
    - Tone degradation under pressure (shorter sentences, more errors)
    - Chronic vs acute — is stress always present, or situational?
    - Recovery language ("I'm better now", "taking a break", "relieved")
    - Anticipatory stress ("I'm already dreading", "tomorrow will be awful")

    OUTPUT: StressProfile + rated 0 (no stress signals) → 1 (severe chronic stress)
    """

    AGENT_ID = "L1-01-stress"
    AGENT_NAME = "Stress Signal Agent"
    DIMENSION = "Biological stress baseline and cortisol-pattern language"
    RAG_QUERIES = [
        "overwhelmed exhausted can't handle too much",
        "deadline pressure urgent need to finish",
        "can't sleep tired headache physical pain",
        "I'm drowning struggling falling behind",
        "feeling better relieved calmer recovering",
        "anxious worried nervous scared about",
    ]
    TOP_K = 8

    @property
    def ANALYSIS_PROMPT(self) -> str:
        return """
ANALYSIS TASK: Stress Signal Detection

Analyze the person's language for chronic and acute stress patterns.

LOOK FOR THESE SPECIFIC SIGNALS:

1. CHRONIC STRESS MARKERS (present across many different contexts):
   - Persistent urgency language regardless of topic
   - Recurring overwhelm statements
   - Frequent references to being tired, drained, depleted
   - Constant future-threat scanning ("what if X happens")

2. ACUTE STRESS MARKERS (situational, time-bounded):
   - Deadline/event-linked panic language
   - Before/after emotional contrast around specific events
   - Crisis language that resolves after the event

3. PHYSICAL CORRELATES:
   - Mentions of physical symptoms (headaches, sleep issues, nausea)
   - References to eating / not eating under pressure
   - Exercise or movement mentions as stress relief

4. STRESS RESPONSE STYLE:
   - Fight (aggressive language when stressed)
   - Flight (avoidance, withdrawal mentions)
   - Freeze (paralysis, "I don't know what to do")
   - Fawn (excessive people-pleasing under pressure)

5. COPING MECHANISMS (healthy and unhealthy):
   - What do they turn to? (food, venting, exercise, substances, distraction)
   - Do they have healthy regulation strategies?

6. RECOVERY CAPACITY:
   - How quickly do they return to baseline after stress?
   - Do they have rest periods, or is it constant?

Rate 0.0 = no stress signals found
Rate 0.5 = moderate, situational stress patterns
Rate 1.0 = severe, chronic, pervasive stress baseline
"""

    @property
    def OUTPUT_INSTRUCTIONS(self) -> str:
        return """
Return this exact JSON structure:
{
  "rating": 0.0-1.0,
  "confidence": 0.0-1.0,
  "label": "ONE OF: NO_STRESS | LOW_STRESS | MODERATE_STRESS | HIGH_STRESS | SEVERE_CHRONIC_STRESS",
  "summary": "2-3 sentences describing what you found",
  "evidence_quotes": ["exact quote 1", "exact quote 2", "exact quote 3"],
  "evidence_chunk_ids": ["chunk_id_1", "chunk_id_2"],
  "patterns_found": ["pattern name 1", "pattern name 2"],
  "counter_evidence": ["quote showing low stress or recovery"],
  "contradictions_internal": ["any self-contradictions in stress patterns"],
  "structured_data": {
    "baseline_stress_level": "LOW | MODERATE | HIGH | SEVERE",
    "is_chronic": true or false,
    "trigger_topics": ["topic1", "topic2"],
    "coping_mechanisms_observed": ["mechanism1", "mechanism2"],
    "stress_language_frequency": 0.0-1.0,
    "recovery_signals": ["quote showing recovery"],
    "stress_response_style": "fight | flight | freeze | fawn | mixed"
  }
}"""

    def parse_structured_data(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "baseline_stress_level": raw.get("baseline_stress_level", "UNKNOWN"),
            "is_chronic": raw.get("is_chronic", False),
            "trigger_topics": raw.get("trigger_topics", []),
            "coping_mechanisms_observed": raw.get("coping_mechanisms_observed", []),
            "stress_language_frequency": raw.get("stress_language_frequency", 0.0),
            "recovery_signals": raw.get("recovery_signals", []),
            "stress_response_style": raw.get("stress_response_style", "unknown"),
        }


class EnergyVitalityAgent(MicroAgent):
    """
    L1-02 | Energy & Vitality Agent
    ──────────────────────────────────────────────────────────────────
    OWNS: Physical and mental energy levels, fatigue patterns,
          sleep quality signals, and physical state self-reports.

    WHY THIS MATTERS: Energy is the substrate of all mental activity.
    A person running on low energy will have degraded cognition, emotion
    regulation, and willpower — even if nothing else is "wrong". This
    agent distinguishes genuine low-energy people from stressed ones.

    WHAT IT LOOKS FOR:
    - Sleep mentions (quality, quantity, insomnia references)
    - Fatigue vocabulary (tired, drained, exhausted, sluggish)
    - High-energy signals (excited, motivated, feeling great, alert)
    - Energy variance across day/week (morning vs night patterns)
    - Activity level (gym, walks, sedentary mentions)
    - Stimulant mentions (coffee dependence as energy signal)
    - Vitality language (feeling alive, energized, in flow)
    """

    AGENT_ID = "L1-02-energy"
    AGENT_NAME = "Energy & Vitality Agent"
    DIMENSION = "Physical and mental energy levels, fatigue, and vitality patterns"
    RAG_QUERIES = [
        "tired exhausted drained no energy sleepy",
        "can't sleep insomnia awake night",
        "feeling great energetic motivated alive",
        "coffee caffeine energy boost need",
        "gym workout exercise running active",
        "feeling sluggish heavy foggy brain",
    ]
    TOP_K = 6

    @property
    def ANALYSIS_PROMPT(self) -> str:
        return """
ANALYSIS TASK: Energy & Vitality Pattern Detection

Map the person's energy landscape across the chat history.

WHAT TO ASSESS:

1. BASELINE ENERGY LEVEL:
   - What is their default energy state? High, moderate, or consistently low?
   - Is this stable or highly variable?

2. SLEEP QUALITY SIGNALS:
   - Any direct mentions of sleep (good/bad nights, insomnia, oversleeping)
   - Indirect signals: time of messages (very late/early), mental fog references
   - "Tired" as a chronic state vs occasional mention

3. PHYSICAL ACTIVITY INDICATORS:
   - Exercise, sport, walking, movement mentions
   - Sedentary mentions (sitting all day, not moving)
   - Body-related activity or neglect

4. ENERGY-EMOTION CORRELATION:
   - Do high-energy states correlate with positive emotional content?
   - Do low-energy mentions cluster with negative topics/contexts?

5. STIMULANT DEPENDENCE:
   - Coffee/caffeine as a functional dependency (signals baseline low energy)
   - Mentions of needing stimulants to function normally

6. VITALITY VS DEPLETION RATIO:
   - Count approximate ratio of vitality language vs depletion language
   - Trend over time: improving, declining, or stable?

Rate 0.0 = severe, chronic depletion / no vitality signals
Rate 0.5 = moderate, mixed energy with periodic recovery
Rate 1.0 = consistently high vitality, good energy management
"""

    @property
    def OUTPUT_INSTRUCTIONS(self) -> str:
        return """
Return this exact JSON structure:
{
  "rating": 0.0-1.0,
  "confidence": 0.0-1.0,
  "label": "ONE OF: DEPLETED | LOW_ENERGY | MODERATE | HIGH_VITALITY | FLUCTUATING",
  "summary": "2-3 sentences describing energy patterns found",
  "evidence_quotes": ["exact quote 1", "exact quote 2", "exact quote 3"],
  "evidence_chunk_ids": ["chunk_id_1"],
  "patterns_found": ["pattern 1", "pattern 2"],
  "counter_evidence": ["quotes showing good energy"],
  "contradictions_internal": [],
  "structured_data": {
    "baseline_energy": "DEPLETED | LOW | MODERATE | HIGH",
    "sleep_quality_signal": "POOR | MODERATE | GOOD | UNKNOWN",
    "activity_level": "SEDENTARY | LOW | MODERATE | ACTIVE",
    "stimulant_dependence": true or false,
    "energy_trend": "DECLINING | STABLE | IMPROVING | FLUCTUATING",
    "peak_energy_contexts": ["when they seem most energized"],
    "depletion_contexts": ["when they seem most drained"]
  }
}"""

    def parse_structured_data(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "baseline_energy": raw.get("baseline_energy", "UNKNOWN"),
            "sleep_quality_signal": raw.get("sleep_quality_signal", "UNKNOWN"),
            "activity_level": raw.get("activity_level", "UNKNOWN"),
            "stimulant_dependence": raw.get("stimulant_dependence", False),
            "energy_trend": raw.get("energy_trend", "STABLE"),
            "peak_energy_contexts": raw.get("peak_energy_contexts", []),
            "depletion_contexts": raw.get("depletion_contexts", []),
        }


# ══════════════════════════════════════════════════════════════════════
# PERCEPTION LAYER
# ══════════════════════════════════════════════════════════════════════

class AttentionFocusAgent(MicroAgent):
    """
    L1-03 | Attention Focus Agent
    ──────────────────────────────────────────────────────────────────
    OWNS: What the person consistently notices vs ignores.
          Detail orientation vs big-picture thinking. Curiosity signals.

    WHY THIS MATTERS: Attention is reality construction. What you
    habitually attend to becomes your world. A detail-oriented person
    and a big-picture thinker in the same situation will have
    completely different experiences and problems.

    WHAT IT LOOKS FOR:
    - Specificity of language (exact numbers, names, details vs vague generalities)
    - Topic recursion (what they keep coming back to unprompted)
    - Tangent behavior (do they follow every interesting thread or stay focused?)
    - What they notice that others miss (observational depth)
    - What conversational threads they ignore or drop
    - Novelty-seeking language vs depth-seeking language
    - Curiosity signals: questions asked, subjects explored voluntarily
    """

    AGENT_ID = "L1-03-attention"
    AGENT_NAME = "Attention Focus Agent"
    DIMENSION = "Attention patterns, detail vs big-picture orientation, curiosity signals"
    RAG_QUERIES = [
        "specifically exactly precisely the exact number",
        "in general overall big picture broadly",
        "I noticed I realized I observed interesting",
        "why how does what if I wonder",
        "going back to topic I keep thinking about",
        "curious questions asking exploring",
    ]
    TOP_K = 8

    @property
    def ANALYSIS_PROMPT(self) -> str:
        return """
ANALYSIS TASK: Attention Focus Pattern Analysis

Determine WHERE this person's attention habitually lands.

SPECIFIC THINGS TO ASSESS:

1. DETAIL VS BIG-PICTURE RATIO:
   - Does their language use specific numbers, names, exact details?
   - OR do they speak in generalities, concepts, overviews?
   - Score detail-orientation 0 (pure big-picture) to 1 (obsessive detail)

2. TOPIC GRAVITY (what pulls attention repeatedly):
   - What subjects appear across multiple unrelated conversations?
   - These are the person's natural attention magnets
   - List them — they reveal values AND obsessions

3. CURIOSITY SIGNATURE:
   - How many questions do they ask vs statements they make?
   - Do they ask deep "why" questions or surface "how/what" questions?
   - Do they follow up on answers or drop threads?
   - Exploratory tangents: do they chase interesting ideas?

4. ATTENTION GAPS (what they consistently miss/ignore):
   - Topics raised by others that they don't engage with
   - Details they overlook that seem important
   - Emotional signals from others they appear to miss

5. FOCUS STABILITY:
   - Do they stay on topic in conversations or scatter?
   - Signs of hyperfocus (going very deep on narrow topics)?
   - Signs of attention fragmentation (jumping between topics constantly)?

Rate 0.0 = very scattered, surface-level, misses details
Rate 0.5 = balanced, context-dependent focus
Rate 1.0 = exceptional focus, deep noticing, strong curiosity
"""

    @property
    def OUTPUT_INSTRUCTIONS(self) -> str:
        return """
Return this exact JSON structure:
{
  "rating": 0.0-1.0,
  "confidence": 0.0-1.0,
  "label": "ONE OF: SCATTERED | SURFACE | BALANCED | DETAIL_ORIENTED | DEEP_FOCUSER",
  "summary": "2-3 sentences",
  "evidence_quotes": ["quote1", "quote2", "quote3"],
  "evidence_chunk_ids": ["id1"],
  "patterns_found": ["pattern1", "pattern2"],
  "counter_evidence": [],
  "contradictions_internal": [],
  "structured_data": {
    "orientation": "DETAIL | BIG_PICTURE | BALANCED",
    "curiosity_level": 0.0-1.0,
    "dominant_attention_topics": ["topic1", "topic2", "topic3"],
    "question_to_statement_ratio": 0.0-1.0,
    "focus_stability": "SCATTERED | MODERATE | STABLE | HYPERFOCUSED",
    "attention_gaps_observed": ["what they miss"],
    "novelty_vs_depth": "NOVELTY_SEEKER | DEPTH_SEEKER | BALANCED"
  }
}"""

    def parse_structured_data(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "orientation": raw.get("orientation", "BALANCED"),
            "curiosity_level": raw.get("curiosity_level", 0.5),
            "dominant_attention_topics": raw.get("dominant_attention_topics", []),
            "question_to_statement_ratio": raw.get("question_to_statement_ratio", 0.2),
            "focus_stability": raw.get("focus_stability", "MODERATE"),
            "attention_gaps_observed": raw.get("attention_gaps_observed", []),
            "novelty_vs_depth": raw.get("novelty_vs_depth", "BALANCED"),
        }


class InterpretationBiasAgent(MicroAgent):
    """
    L1-04 | Interpretation Bias Agent
    ──────────────────────────────────────────────────────────────────
    OWNS: The lens through which events are interpreted.
          Optimism vs pessimism. Threat sensitivity. Attribution style.

    WHY THIS MATTERS: Two people experience the same event completely
    differently based on their interpretation lens. This lens is
    invisible to the person — it just feels like "seeing reality."
    Understanding it reveals the person's emotional default state.

    WHAT IT LOOKS FOR:
    - Default framing of ambiguous events (charitable vs suspicious)
    - Threat detection sensitivity (seeing danger in neutral situations)
    - Optimism markers (silver lining, "it'll work out", positive reframes)
    - Pessimism markers (catastrophizing, worst-case assumption)
    - Attribution style: internal ("my fault") vs external ("their fault" / "situation")
    - Stable vs unstable attribution ("always" vs "this time")
    - Global vs specific ("I'm a failure" vs "I failed at this")
    """

    AGENT_ID = "L1-04-interpretation"
    AGENT_NAME = "Interpretation Bias Agent"
    DIMENSION = "Optimism/pessimism lens, threat sensitivity, attribution style"
    RAG_QUERIES = [
        "it'll be fine will work out positive hopeful",
        "worst case probably will fail going wrong",
        "they must think they probably meant suspicious",
        "my fault I should have I always mess up",
        "their fault because of them unfair situation",
        "always never this will never change",
    ]
    TOP_K = 8

    @property
    def ANALYSIS_PROMPT(self) -> str:
        return """
ANALYSIS TASK: Interpretation Bias Detection

Map how this person habitually interprets events, other people's intentions,
and their own experiences.

ASSESS THESE DIMENSIONS:

1. OPTIMISM vs PESSIMISM BASELINE:
   - How do they frame uncertain outcomes? Best case or worst case?
   - When something is ambiguous, what do they assume?
   - Look for: silver lining language, hopeful predictions, positive reframes
   - vs: catastrophizing, doom predictions, "knowing" bad outcomes before they happen

2. THREAT SENSITIVITY:
   - Do they frequently interpret neutral events as threats?
   - Do they read hostile intent into ambiguous social situations?
   - Hypervigilance signals: constantly preparing for problems, seeing risks everywhere
   - OR: relaxed, assumes benign intent, low threat scanning

3. ATTRIBUTION STYLE (Seligman's 3 dimensions):
   INTERNAL/EXTERNAL: Do they blame themselves or others/situations?
   STABLE/UNSTABLE: "This always happens to me" vs "This happened this time"
   GLOBAL/SPECIFIC: "I'm a failure" vs "I failed at this task"

   Depressive attributional style = Internal + Stable + Global for bad events
   Optimistic style = External + Unstable + Specific for bad events

4. CHARITABLE vs SUSPICIOUS INTERPRETATION:
   When someone's behavior is ambiguous, do they assume the best or worst?
   Evidence: how they interpret other people's messages, actions, silences

5. FILTERING:
   - Do they systematically filter out positive information?
   - Or negative information?
   - Mental filtering is a cognitive distortion — look for systematic blind spots

Rate 0.0 = severe negative bias (pessimism + threat + internal-stable-global)
Rate 0.5 = balanced, context-sensitive interpretation
Rate 1.0 = strong positive bias (optimism + charitable + external-unstable-specific)
"""

    @property
    def OUTPUT_INSTRUCTIONS(self) -> str:
        return """
Return this exact JSON structure:
{
  "rating": 0.0-1.0,
  "confidence": 0.0-1.0,
  "label": "ONE OF: SEVERE_NEGATIVE_BIAS | NEGATIVE_BIAS | BALANCED | POSITIVE_BIAS | STRONG_POSITIVE_BIAS",
  "summary": "2-3 sentences",
  "evidence_quotes": ["quote1", "quote2", "quote3"],
  "evidence_chunk_ids": ["id1"],
  "patterns_found": ["pattern1"],
  "counter_evidence": [],
  "contradictions_internal": [],
  "structured_data": {
    "optimism_pessimism_score": 0.0-1.0,
    "threat_sensitivity": "LOW | MODERATE | HIGH | HYPERVIGILANT",
    "attribution_style": {
      "internal_external": "INTERNAL | EXTERNAL | MIXED",
      "stable_unstable": "STABLE | UNSTABLE | MIXED",
      "global_specific": "GLOBAL | SPECIFIC | MIXED"
    },
    "charitable_vs_suspicious": "VERY_CHARITABLE | CHARITABLE | NEUTRAL | SUSPICIOUS | PARANOID",
    "catastrophizing_frequency": "RARE | OCCASIONAL | FREQUENT | CONSTANT",
    "silver_lining_frequency": "RARE | OCCASIONAL | FREQUENT | CONSTANT"
  }
}"""

    def parse_structured_data(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "optimism_pessimism_score": raw.get("optimism_pessimism_score", 0.5),
            "threat_sensitivity": raw.get("threat_sensitivity", "MODERATE"),
            "attribution_style": raw.get("attribution_style", {}),
            "charitable_vs_suspicious": raw.get("charitable_vs_suspicious", "NEUTRAL"),
            "catastrophizing_frequency": raw.get("catastrophizing_frequency", "OCCASIONAL"),
            "silver_lining_frequency": raw.get("silver_lining_frequency", "OCCASIONAL"),
        }


class ConfirmationBiasAgent(MicroAgent):
    """
    L1-05 | Confirmation Bias Agent
    ──────────────────────────────────────────────────────────────────
    OWNS: How the person filters information to match existing beliefs.
          What they seek out vs dismiss. Echo chamber behavior.

    WHAT IT LOOKS FOR:
    - Resistance to contradictory information ("but...", "yes but...", dismissal)
    - Seeking validation vs seeking truth (questions that expect agreement)
    - Dismissal patterns ("that's not valid", "they don't understand")
    - Selectively citing evidence that confirms existing views
    - Updating beliefs when faced with evidence (rare if high confirmation bias)
    - Intellectual openness markers (genuine uncertainty, changing views)
    """

    AGENT_ID = "L1-05-confirmation"
    AGENT_NAME = "Confirmation Bias Agent"
    DIMENSION = "Information filtering, confirmation bias, intellectual openness"
    RAG_QUERIES = [
        "but actually no that's wrong they don't understand",
        "see this proves I was right exactly what I thought",
        "I changed my mind I was wrong actually reconsidering",
        "what do you think am I right is this correct validate",
        "dismissing ignoring not relevant not applicable",
        "I've always known this is obvious clearly",
    ]
    TOP_K = 7

    @property
    def ANALYSIS_PROMPT(self) -> str:
        return """
ANALYSIS TASK: Confirmation Bias Detection

Assess how this person handles information that challenges vs confirms their beliefs.

KEY SIGNALS:

1. VALIDATION-SEEKING vs TRUTH-SEEKING:
   - Do they ask questions that assume an answer? ("It's bad, right?")
   - Do they genuinely ask open questions? ("What do you think about X?")
   - Do they engage with counter-arguments or redirect away from them?

2. BELIEF UPDATING EVIDENCE:
   - Have they ever explicitly changed their mind in the conversation?
   - Do they acknowledge being wrong? (This is rare with high confirmation bias)
   - Phrases like "actually I was wrong about that" or "good point, I hadn't considered"

3. DISMISSAL PATTERNS:
   - How do they handle contradictory information?
   - Do they attack the source? ("They're biased", "They don't know what they're talking about")
   - Do they find technicalities to dismiss? ("That's a different situation")
   - Do they ignore it silently?

4. EVIDENCE SELECTIVITY:
   - Do they cite only evidence supporting their view?
   - Do they steelman opposing arguments or strawman them?
   - Do they acknowledge nuance or present black-and-white?

5. INTELLECTUAL HUMILITY MARKERS:
   - "I'm not sure", "I might be wrong", "I need to think about this"
   - Asking for opposing views genuinely
   - Revising positions after discussion

Rate 0.0 = severe confirmation bias (never updates, dismisses all counter-evidence)
Rate 0.5 = moderate, some openness with some dismissal
Rate 1.0 = highly open, actively seeks disconfirming evidence
"""

    @property
    def OUTPUT_INSTRUCTIONS(self) -> str:
        return """
Return this exact JSON structure:
{
  "rating": 0.0-1.0,
  "confidence": 0.0-1.0,
  "label": "ONE OF: SEVERE_ECHO_CHAMBER | HIGH_CONFIRMATION_BIAS | MODERATE | OPEN_MINDED | HIGHLY_OPEN",
  "summary": "2-3 sentences",
  "evidence_quotes": ["quote1", "quote2", "quote3"],
  "evidence_chunk_ids": ["id1"],
  "patterns_found": ["pattern1"],
  "counter_evidence": [],
  "contradictions_internal": [],
  "structured_data": {
    "intellectual_openness": 0.0-1.0,
    "belief_updating_observed": true or false,
    "validation_seeking_frequency": "LOW | MODERATE | HIGH",
    "dismissal_style": "IGNORE | SOURCE_ATTACK | TECHNICALITY | REDIRECT | ENGAGES_HONESTLY",
    "instances_of_mind_change": 0,
    "steelmanning_ability": "NONE | WEAK | MODERATE | STRONG"
  }
}"""

    def parse_structured_data(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "intellectual_openness": raw.get("intellectual_openness", 0.5),
            "belief_updating_observed": raw.get("belief_updating_observed", False),
            "validation_seeking_frequency": raw.get("validation_seeking_frequency", "MODERATE"),
            "dismissal_style": raw.get("dismissal_style", "UNKNOWN"),
            "instances_of_mind_change": raw.get("instances_of_mind_change", 0),
            "steelmanning_ability": raw.get("steelmanning_ability", "WEAK"),
        }


# ══════════════════════════════════════════════════════════════════════
# DUAL-PROCESS LAYER
# ══════════════════════════════════════════════════════════════════════

class AutomaticReactionAgent(MicroAgent):
    """
    L1-06 | Automatic Reaction Agent
    ──────────────────────────────────────────────────────────────────
    OWNS: System 1 triggers — moments where the person reacts before
          thinking, makes snap judgments, and shows automatic responses.

    WHY THIS MATTERS: Kahneman showed that System 1 runs the show 95%
    of the time. Understanding what triggers automatic (non-deliberate)
    responses reveals the person's deepest behavioral drivers — the ones
    they aren't aware of and can't easily control.

    WHAT IT LOOKS FOR:
    - Speed of opinion formation on certain topics (instant certainty)
    - Emotional spikes followed by rational explanation after the fact
    - Patterns of regret ("I shouldn't have said that", "I reacted too fast")
    - Topics that always produce the same automatic response
    - Gut-feeling language ("I just know", "something feels off")
    - Impulsive decisions mentioned ("I just decided", "without thinking")
    """

    AGENT_ID = "L1-06-system1"
    AGENT_NAME = "Automatic Reaction Agent"
    DIMENSION = "System 1 triggers, snap judgments, automatic responses, impulsivity"
    RAG_QUERIES = [
        "instantly immediately suddenly I just knew gut feeling",
        "I shouldn't have said that reacted too fast regret",
        "just decided without thinking impulsive",
        "something feels off wrong doesn't feel right",
        "I can't help it automatic just happens",
        "immediately responded said before thinking",
    ]
    TOP_K = 7

    @property
    def ANALYSIS_PROMPT(self) -> str:
        return """
ANALYSIS TASK: Automatic Reaction (System 1) Pattern Detection

Identify where this person's behavior is driven by fast, automatic,
unconscious processes rather than deliberate reasoning.

WHAT TO LOOK FOR:

1. SNAP JUDGMENT PATTERNS:
   - Topics where they form instant opinions with high certainty
   - No deliberation language before the conclusion
   - "Obviously", "clearly", "of course" on complex topics = automatic processing

2. REACTIVE LANGUAGE:
   - Messages sent in the heat of the moment (later walked back or regretted)
   - Emotional spike followed by rationalization ("I was upset because...")
   - Short, intense responses that break their normal communication pattern

3. GUT-FEELING RELIANCE:
   - "I just know", "my gut says", "something feels off"
   - Making important decisions based on feel rather than analysis
   - Intuition language that bypasses reasoning

4. IMPULSIVITY SIGNALS:
   - "I just bought / did / said / decided..."
   - Acting before thinking, then dealing with consequences
   - Patterns of impulsive behavior mentioned across different domains

5. REGRET PATTERNS (evidence of System 1 override):
   - "I shouldn't have said that"
   - "I reacted too quickly"
   - "I wish I'd thought before..."
   - These are gold — they reveal where System 1 misfired

6. RECURRING AUTOMATIC TRIGGERS:
   - What specific topics/people/situations always produce the same fast reaction?
   - These are their hot buttons — System 1 has learned a pattern

Rate 0.0 = highly deliberate, System 2 dominant
Rate 0.5 = balanced, context-dependent
Rate 1.0 = highly reactive, System 1 dominant, frequent automatic overrides
"""

    @property
    def OUTPUT_INSTRUCTIONS(self) -> str:
        return """
Return this exact JSON structure:
{
  "rating": 0.0-1.0,
  "confidence": 0.0-1.0,
  "label": "ONE OF: HIGHLY_DELIBERATE | DELIBERATE | BALANCED | REACTIVE | HIGHLY_REACTIVE",
  "summary": "2-3 sentences",
  "evidence_quotes": ["quote1", "quote2", "quote3"],
  "evidence_chunk_ids": ["id1"],
  "patterns_found": ["pattern1"],
  "counter_evidence": [],
  "contradictions_internal": [],
  "structured_data": {
    "system1_dominance": 0.0-1.0,
    "hot_button_topics": ["topic that always triggers automatic response"],
    "regret_instances": ["quote showing post-reaction regret"],
    "gut_reliance_frequency": "LOW | MODERATE | HIGH",
    "impulsivity_domains": ["where impulsivity appears: money, relationships, speech"],
    "snap_judgment_strength": "WEAK | MODERATE | STRONG"
  }
}"""

    def parse_structured_data(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "system1_dominance": raw.get("system1_dominance", 0.5),
            "hot_button_topics": raw.get("hot_button_topics", []),
            "regret_instances": raw.get("regret_instances", []),
            "gut_reliance_frequency": raw.get("gut_reliance_frequency", "MODERATE"),
            "impulsivity_domains": raw.get("impulsivity_domains", []),
            "snap_judgment_strength": raw.get("snap_judgment_strength", "MODERATE"),
        }


class CognitiveBiasMapAgent(MicroAgent):
    """
    L1-07 | Cognitive Bias Map Agent
    ──────────────────────────────────────────────────────────────────
    OWNS: Identification of specific active cognitive biases from
          a library of 30+ documented bias types.

    WHY THIS MATTERS: This is the most diagnostic agent. Biases are
    systematic errors — once identified, they predict future mistakes.
    Someone with strong sunk cost bias will make specific predictable
    errors. Someone with strong availability bias will have specific
    predictable fears. Knowing the bias map = knowing the failure modes.

    WHAT IT LOOKS FOR:
    - Sunk cost fallacy ("I've already invested so much")
    - Availability heuristic (overweighting recent/vivid events)
    - Dunning-Kruger signals (overconfidence in areas of low knowledge)
    - In-group/out-group bias (us vs them framing)
    - Anchoring (first number/option dominates thinking)
    - Fundamental attribution error (blaming character not situation)
    - Planning fallacy (consistently underestimating time/difficulty)
    - Negativity bias (negative info weighted more than positive)
    """

    AGENT_ID = "L1-07-biases"
    AGENT_NAME = "Cognitive Bias Map Agent"
    DIMENSION = "Identification of active cognitive biases from 30+ documented types"
    RAG_QUERIES = [
        "already invested can't give up too much time effort",
        "everyone knows obviously people like us they",
        "I'm pretty good at sure it will work confident",
        "last time this happened remember when recently",
        "their fault type of person character",
        "should have been done by now underestimated",
        "bad news terrible outweighs good news problem focus",
        "first option number reference comparison anchor",
    ]
    TOP_K = 10

    @property
    def ANALYSIS_PROMPT(self) -> str:
        return """
ANALYSIS TASK: Cognitive Bias Identification

Scan ALL retrieved chunks and identify which specific cognitive biases
are active and dominant in this person's thinking.

CHECK FOR EACH OF THESE BIASES:

1. SUNK COST FALLACY: "I've already put X in, can't stop now"
   Signal: Continuing something because of past investment, not future value

2. AVAILABILITY HEURISTIC: Recent/vivid events dominate risk assessment
   Signal: Overweighting dramatic recent examples when reasoning about probability

3. DUNNING-KRUGER: Overconfidence in areas of limited knowledge
   Signal: Very confident statements in domains where errors also appear

4. FUNDAMENTAL ATTRIBUTION ERROR: People's behavior = their character, not situation
   Signal: "He's just lazy" rather than "He had a difficult situation"

5. IN-GROUP BIAS: Favoritism toward perceived in-group
   Signal: "People like us", "they" (out-group) described negatively

6. ANCHORING: First piece of info dominates subsequent reasoning
   Signal: Reference to initial number/option even when updating

7. PLANNING FALLACY: Underestimating time/cost/difficulty of own projects
   Signal: "It'll only take me X" followed by overruns

8. NEGATIVITY BIAS: Negative information weighted more heavily than positive
   Signal: One negative outweighs multiple positives systematically

9. SUNK COST / STATUS QUO BIAS: Preference for current state
   Signal: Resistance to change even when change is clearly better

10. SOCIAL PROOF: "Everyone does X, so it must be right"
    Signal: Justifying behavior by pointing to what others do

11. CONFIRMATION BIAS: (coordinate with L1-05)
    Signal: Only noticing/citing evidence that confirms existing beliefs

12. SELF-SERVING BIAS: Success = my talent; failure = external factors
    Signal: Asymmetric attribution of outcomes

ONLY report biases where you have DIRECT EVIDENCE from the chat.
Do not report every possible bias — only those with quote-level proof.
"""

    @property
    def OUTPUT_INSTRUCTIONS(self) -> str:
        return """
Return this exact JSON structure:
{
  "rating": 0.0-1.0,
  "confidence": 0.0-1.0,
  "label": "ONE OF: LOW_BIAS | MODERATE_BIAS | HIGH_BIAS | SEVERE_BIAS",
  "summary": "2-3 sentences listing dominant biases found",
  "evidence_quotes": ["quote1", "quote2", "quote3"],
  "evidence_chunk_ids": ["id1"],
  "patterns_found": ["bias name 1", "bias name 2"],
  "counter_evidence": [],
  "contradictions_internal": [],
  "structured_data": {
    "active_biases": [
      {
        "bias_name": "Sunk Cost Fallacy",
        "severity": "MILD | MODERATE | STRONG",
        "evidence": ["exact quote showing this bias"],
        "frequency": 3
      }
    ],
    "total_biases_identified": 0,
    "dominant_bias": "most prominent bias name",
    "bias_domains": ["relationships", "work", "money"]
  }
}"""

    def parse_structured_data(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "active_biases": raw.get("active_biases", []),
            "total_biases_identified": raw.get("total_biases_identified", 0),
            "dominant_bias": raw.get("dominant_bias", "UNKNOWN"),
            "bias_domains": raw.get("bias_domains", []),
        }


# ══════════════════════════════════════════════════════════════════════
# EMOTION LAYER
# ══════════════════════════════════════════════════════════════════════

class EmotionalRangeAgent(MicroAgent):
    """
    L1-08 | Emotional Range Agent
    ──────────────────────────────────────────────────────────────────
    OWNS: The breadth and expressiveness of emotional vocabulary.
          Emotional intensity patterns. Emotional literacy.

    WHY THIS MATTERS: Emotional vocabulary is emotional intelligence.
    People who can name only 3 emotions (good, bad, fine) have fewer
    tools for self-regulation than people who can distinguish between
    shame, embarrassment, guilt, and regret. Range predicts regulation.

    WHAT IT LOOKS FOR:
    - Total emotional vocabulary used
    - Granularity (fine vs. content vs. peaceful vs. satisfied)
    - Intensity markers (a bit vs. devastated)
    - Emotional suppression signals ("I'm fine", emotion-free descriptions)
    - Positivity range vs negativity range (asymmetry)
    - Body-emotion language (physical description of feelings)
    - Emotion labeling ability
    """

    AGENT_ID = "L1-08-emotional-range"
    AGENT_NAME = "Emotional Range Agent"
    DIMENSION = "Emotional vocabulary breadth, expression intensity, emotional literacy"
    RAG_QUERIES = [
        "feel feeling felt emotion happy sad angry",
        "devastated ecstatic furious terrified elated",
        "I'm fine okay not a big deal whatever",
        "heart racing chest tight physically felt in my body",
        "a bit slightly somewhat kind of sort of",
        "really very extremely totally completely intensely",
    ]
    TOP_K = 8

    @property
    def ANALYSIS_PROMPT(self) -> str:
        return """
ANALYSIS TASK: Emotional Range & Vocabulary Assessment

Map the emotional expressiveness and literacy of this person.

WHAT TO ASSESS:

1. EMOTIONAL VOCABULARY SIZE:
   Count ALL distinct emotion words used across the corpus.
   LOW (< 10 distinct emotions): limited emotional literacy
   MODERATE (10-25): average range
   HIGH (25-50): good emotional vocabulary
   EXCEPTIONAL (50+): rich emotional language

2. GRANULARITY vs COARSENESS:
   Coarse: happy, sad, angry, scared, fine
   Granular: content, melancholy, irritated, apprehensive, satisfied
   Does this person use fine-grained emotional distinctions?

3. INTENSITY PATTERNS:
   Do they understate? ("a bit upset" when clearly devastated)
   Do they overstate? (dramatic language for minor events)
   Are they calibrated? (intensity matches apparent situation)

4. SUPPRESSION SIGNALS:
   "I'm fine" / "it's nothing" / "whatever" when context suggests emotion
   Neutral language in clearly emotional situations
   Changing the subject when emotions arise

5. BODY-EMOTION CONNECTION:
   Do they describe feelings in their body? (chest, stomach, throat)
   This indicates somatic awareness — higher emotional intelligence

6. EMOTIONAL RANGE ASYMMETRY:
   Is their vocabulary richer for positive or negative emotions?
   Some people have 20 words for negative emotions and 3 for positive.

7. EMOTIONAL CONTAGION LANGUAGE:
   Do they match others' emotional language in conversation?
   This reveals empathy and emotional attunement.

Rate 0.0 = emotionally shut down, minimal vocabulary, suppressed
Rate 0.5 = average range, moderate expressiveness
Rate 1.0 = rich vocabulary, high granularity, somatic awareness, well-calibrated
"""

    @property
    def OUTPUT_INSTRUCTIONS(self) -> str:
        return """
Return this exact JSON structure:
{
  "rating": 0.0-1.0,
  "confidence": 0.0-1.0,
  "label": "ONE OF: EMOTIONALLY_CLOSED | RESTRICTED | MODERATE | EXPRESSIVE | HIGHLY_EXPRESSIVE",
  "summary": "2-3 sentences",
  "evidence_quotes": ["quote1", "quote2", "quote3"],
  "evidence_chunk_ids": ["id1"],
  "patterns_found": ["pattern1"],
  "counter_evidence": [],
  "contradictions_internal": [],
  "structured_data": {
    "vocabulary_richness": "LOW | MODERATE | HIGH | EXCEPTIONAL",
    "granularity_level": "COARSE | MODERATE | FINE | VERY_FINE",
    "intensity_calibration": "UNDERSTATES | CALIBRATED | OVERSTATES",
    "suppression_signals_found": true or false,
    "body_emotion_connection": true or false,
    "positive_vocab_count_estimate": 0,
    "negative_vocab_count_estimate": 0,
    "emotional_words_sample": ["word1", "word2", "word3", "word4", "word5"]
  }
}"""

    def parse_structured_data(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "vocabulary_richness": raw.get("vocabulary_richness", "MODERATE"),
            "granularity_level": raw.get("granularity_level", "MODERATE"),
            "intensity_calibration": raw.get("intensity_calibration", "CALIBRATED"),
            "suppression_signals_found": raw.get("suppression_signals_found", False),
            "body_emotion_connection": raw.get("body_emotion_connection", False),
            "positive_vocab_count_estimate": raw.get("positive_vocab_count_estimate", 0),
            "negative_vocab_count_estimate": raw.get("negative_vocab_count_estimate", 0),
            "emotional_words_sample": raw.get("emotional_words_sample", []),
        }


class EmotionalRegulationAgent(MicroAgent):
    """
    L1-09 | Emotional Regulation Agent
    ──────────────────────────────────────────────────────────────────
    OWNS: Impulse control, reactivity vs calm reasoning, the ability
          to feel emotions without being controlled by them.

    WHY THIS MATTERS: Regulation is the difference between emotional
    intelligence and emotional chaos. A person can have high emotional
    range but terrible regulation. This is the most important emotional
    competency for relationships and decision-making.

    WHAT IT LOOKS FOR:
    - Amygdala hijack evidence (things said in anger later regretted)
    - Cooling-down patterns (time between trigger and response)
    - Emotional flooding: when emotion completely takes over language
    - Self-regulation strategies mentioned (breathing, walking away)
    - Recovery time after emotional spikes
    - Ability to reason while emotional vs complete takeover
    - Escalation patterns in conflict
    """

    AGENT_ID = "L1-09-regulation"
    AGENT_NAME = "Emotional Regulation Agent"
    DIMENSION = "Impulse control, emotional reactivity vs calm reasoning, regulation capacity"
    RAG_QUERIES = [
        "I was so angry I couldn't think couldn't stop myself",
        "I calmed down took a breath stepped back paused",
        "I regret saying that I shouldn't have sent that",
        "even though I'm upset I need to think about this",
        "I can't control it just snapped lost it",
        "I'm trying to stay calm despite how I feel",
        "working through it processing dealing with emotions",
    ]
    TOP_K = 8

    @property
    def ANALYSIS_PROMPT(self) -> str:
        return """
ANALYSIS TASK: Emotional Regulation Assessment

Evaluate how well this person manages their emotional states, especially
in stressful or provocative situations.

REGULATION SIGNALS TO LOOK FOR:

1. AMYGDALA HIJACK EVIDENCE:
   - Situations where emotion clearly took over reasoning
   - Statements made in heat-of-moment later regretted
   - Language patterns that show flooding (all caps, rapid fire messages,
     extreme language during conflict)

2. IMPULSE CONTROL:
   - Evidence of pausing before responding
   - "I didn't say what I was really thinking"
   - Choosing NOT to engage when provoked
   - vs. Evidence of blurting without filter

3. RECOVERY TIME:
   - How long after an emotional event before they're back to baseline?
   - Do they keep re-flooding on the same event?
   - Can they move on? Or do they ruminate?

4. ACTIVE REGULATION STRATEGIES:
   - Mentions of calming techniques (breathing, walking, time alone)
   - Seeking rational perspective after emotional reaction
   - Journaling, exercise, or other processing

5. EMOTION-REASON INTEGRATION:
   - Can they reason while emotional? Or does all logic shut off?
   - Do they acknowledge both their feelings AND the facts?
   - "I feel X but I understand Y" = good integration

6. CONFLICT ESCALATION PATTERNS:
   - Do they escalate or de-escalate in conflicts?
   - Do they match aggression or stay regulated?
   - Exit strategies when regulation fails?

Rate 0.0 = severely dysregulated (frequent hijacks, no recovery, explosive)
Rate 0.5 = moderate regulation with identifiable failure points
Rate 1.0 = excellent regulation (can feel strongly without losing function)
"""

    @property
    def OUTPUT_INSTRUCTIONS(self) -> str:
        return """
Return this exact JSON structure:
{
  "rating": 0.0-1.0,
  "confidence": 0.0-1.0,
  "label": "ONE OF: SEVERELY_DYSREGULATED | POOR_REGULATION | MODERATE | GOOD_REGULATION | EXCELLENT_REGULATION",
  "summary": "2-3 sentences",
  "evidence_quotes": ["quote1", "quote2", "quote3"],
  "evidence_chunk_ids": ["id1"],
  "patterns_found": ["pattern1"],
  "counter_evidence": [],
  "contradictions_internal": [],
  "structured_data": {
    "hijack_frequency": "RARE | OCCASIONAL | FREQUENT | CONSTANT",
    "impulse_control_level": 0.0-1.0,
    "recovery_speed": "SLOW | MODERATE | FAST",
    "active_strategies_observed": ["strategy1", "strategy2"],
    "conflict_escalation_tendency": "DE_ESCALATES | NEUTRAL | ESCALATES",
    "emotion_reason_integration": "POOR | MODERATE | GOOD | EXCELLENT",
    "rumination_tendency": true or false
  }
}"""

    def parse_structured_data(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "hijack_frequency": raw.get("hijack_frequency", "OCCASIONAL"),
            "impulse_control_level": raw.get("impulse_control_level", 0.5),
            "recovery_speed": raw.get("recovery_speed", "MODERATE"),
            "active_strategies_observed": raw.get("active_strategies_observed", []),
            "conflict_escalation_tendency": raw.get("conflict_escalation_tendency", "NEUTRAL"),
            "emotion_reason_integration": raw.get("emotion_reason_integration", "MODERATE"),
            "rumination_tendency": raw.get("rumination_tendency", False),
        }


class EmotionalThemeAgent(MicroAgent):
    """
    L1-10 | Emotional Theme Agent
    ──────────────────────────────────────────────────────────────────
    OWNS: The dominant recurring emotional patterns across the corpus.
          Emotional climate. What emotional states characterize this person.

    WHAT IT LOOKS FOR:
    - Which emotions appear most frequently across all contexts
    - Emotional themes that persist regardless of topic
    - Seasonal/temporal emotional patterns
    - The underlying emotional tone of the person's life
    """

    AGENT_ID = "L1-10-emotional-themes"
    AGENT_NAME = "Emotional Theme Agent"
    DIMENSION = "Dominant recurring emotional patterns, emotional climate, chronic emotional states"
    RAG_QUERIES = [
        "feeling anxious worried nervous scared uneasy",
        "frustrated annoyed irritated bothered resentful",
        "happy excited joyful enthusiastic grateful",
        "sad lonely empty disconnected hopeless",
        "proud accomplished confident capable",
        "ashamed guilty embarrassed worthless",
        "curious interested passionate engaged",
        "bored indifferent meh whatever don't care",
    ]
    TOP_K = 10

    @property
    def ANALYSIS_PROMPT(self) -> str:
        return """
ANALYSIS TASK: Emotional Theme Mapping

Identify the dominant emotional themes — the recurring emotional states
that characterize this person's inner life.

WHAT TO DO:

1. IDENTIFY ALL RECURRING EMOTIONS:
   Not just what they feel sometimes, but what they feel REPEATEDLY
   across many different contexts and conversations.

2. MAP FREQUENCY AND INTENSITY:
   For each recurring emotion:
   - How often does it appear? (frequency)
   - How strong when it appears? (intensity)
   - In what contexts does it emerge? (triggers)

3. IDENTIFY THE EMOTIONAL BASELINE:
   Strip away specific events — what is the person's DEFAULT emotional state?
   This is the color of their background experience.

4. DETECT EMOTIONAL CONTRADICTIONS:
   Some people display one emotion on the surface and another underneath.
   (e.g., anger on the surface, hurt underneath; humor on the surface, anxiety underneath)

5. IDENTIFY SUPPRESSED EMOTIONS:
   What emotions seem notably ABSENT given what's happening in their life?
   A person going through something difficult who shows zero sadness? (Suppression)
   A person who seems anxious but never mentions it directly?

6. TEMPORAL PATTERNS:
   Are emotional themes consistent across the whole chat history?
   Or have they shifted? (e.g., started hopeful, became more anxious over time)

Build a ranking of the top 5-7 emotional themes from most to least prominent.
"""

    @property
    def OUTPUT_INSTRUCTIONS(self) -> str:
        return """
Return this exact JSON structure:
{
  "rating": 0.0-1.0,
  "confidence": 0.0-1.0,
  "label": "Brief description of dominant emotional climate",
  "summary": "2-3 sentences describing overall emotional themes",
  "evidence_quotes": ["quote1", "quote2", "quote3"],
  "evidence_chunk_ids": ["id1"],
  "patterns_found": ["pattern1"],
  "counter_evidence": [],
  "contradictions_internal": [],
  "structured_data": {
    "emotional_baseline": "what is their default feeling state",
    "top_emotional_themes": [
      {
        "emotion": "anxiety",
        "frequency": 0.0-1.0,
        "intensity": "LOW | MEDIUM | HIGH",
        "triggers": ["situation1", "situation2"],
        "evidence": ["quote showing this emotion"]
      }
    ],
    "surface_vs_depth_mismatch": "description of any emotion hiding another",
    "suppressed_emotions": ["emotion seemingly absent given context"],
    "emotional_trend": "WORSENING | STABLE | IMPROVING | FLUCTUATING"
  }
}"""

    def parse_structured_data(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "emotional_baseline": raw.get("emotional_baseline", "UNKNOWN"),
            "top_emotional_themes": raw.get("top_emotional_themes", []),
            "surface_vs_depth_mismatch": raw.get("surface_vs_depth_mismatch", ""),
            "suppressed_emotions": raw.get("suppressed_emotions", []),
            "emotional_trend": raw.get("emotional_trend", "STABLE"),
        }


# ══════════════════════════════════════════════════════════════════════
# COGNITION LAYER
# ══════════════════════════════════════════════════════════════════════

class ThinkingStyleAgent(MicroAgent):
    """
    L1-11 | Thinking Style Agent
    ──────────────────────────────────────────────────────────────────
    OWNS: How the person processes information — their cognitive style.
          Analytical, intuitive, abstract, systems, or practical thinking.

    WHY THIS MATTERS: Thinking style determines how problems get solved
    and how decisions get made. It predicts what kinds of arguments
    will land, what frustrates this person cognitively, and where their
    blind spots in reasoning are.
    """

    AGENT_ID = "L1-11-thinking"
    AGENT_NAME = "Thinking Style Agent"
    DIMENSION = "Cognitive style: analytical/intuitive/abstract/systems/practical ratio"
    RAG_QUERIES = [
        "because therefore logically if then evidence data",
        "gut feeling intuition just feels right sense",
        "system pattern interconnected big picture how it all fits",
        "concept theory idea abstract principle model",
        "practically specifically concretely in practice real world",
        "step by step first then next finally process",
        "analogy like metaphor similar to imagine if",
    ]
    TOP_K = 9

    @property
    def ANALYSIS_PROMPT(self) -> str:
        return """
ANALYSIS TASK: Thinking Style Profiling

Map this person's cognitive processing style across five dimensions.

THE FIVE THINKING STYLES:

1. ANALYTICAL THINKING:
   Evidence: logical connectors (because, therefore, if-then), data references,
   evidence requirements, systematic reasoning chains, pro/con analysis
   
2. INTUITIVE THINKING:
   Evidence: gut-feeling language, quick conclusions without reasoning shown,
   "just knowing", feeling-based decisions, trusting first impressions

3. ABSTRACT THINKING:
   Evidence: concept building, theoretical exploration, analogies and metaphors,
   interest in ideas-for-their-own-sake, hypotheticals ("what if...")

4. SYSTEMS THINKING:
   Evidence: interconnection language, cause-and-effect chains,
   second and third-order thinking, "how this affects that",
   looking for root causes rather than symptoms

5. PRACTICAL THINKING:
   Evidence: "what does this mean in practice?", real-world application focus,
   impatience with pure theory, action-orientation, concrete examples preferred

ASSESS THE DISTRIBUTION:
What % of their thinking is each type?
What's the dominant style?
Do they switch styles by context? (Domain-dependent thinking styles)

Also assess COGNITIVE COMPLEXITY:
- Do they handle nuance well or prefer black-and-white?
- Can they hold multiple perspectives simultaneously?
- Do they tolerate ambiguity or need certainty?
"""

    @property
    def OUTPUT_INSTRUCTIONS(self) -> str:
        return """
Return this exact JSON structure:
{
  "rating": 0.0-1.0,
  "confidence": 0.0-1.0,
  "label": "DOMINANT THINKING STYLE (e.g. ANALYTICAL-SYSTEMS)",
  "summary": "2-3 sentences",
  "evidence_quotes": ["quote1", "quote2", "quote3"],
  "evidence_chunk_ids": ["id1"],
  "patterns_found": ["pattern1"],
  "counter_evidence": [],
  "contradictions_internal": [],
  "structured_data": {
    "dominant_style": "ANALYTICAL | INTUITIVE | ABSTRACT | SYSTEMS | PRACTICAL | MIXED",
    "style_distribution": {
      "analytical": 0.0-1.0,
      "intuitive": 0.0-1.0,
      "abstract": 0.0-1.0,
      "systems": 0.0-1.0,
      "practical": 0.0-1.0
    },
    "complexity_level": 0.0-1.0,
    "ambiguity_tolerance": 0.0-1.0,
    "abstraction_preference": 0.0-1.0,
    "evidence_requirements": 0.0-1.0,
    "nuance_handling": "POOR | MODERATE | GOOD | EXCELLENT"
  }
}"""

    def parse_structured_data(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "dominant_style": raw.get("dominant_style", "MIXED"),
            "style_distribution": raw.get("style_distribution", {}),
            "complexity_level": raw.get("complexity_level", 0.5),
            "ambiguity_tolerance": raw.get("ambiguity_tolerance", 0.5),
            "abstraction_preference": raw.get("abstraction_preference", 0.5),
            "evidence_requirements": raw.get("evidence_requirements", 0.5),
            "nuance_handling": raw.get("nuance_handling", "MODERATE"),
        }


class LanguageComplexityAgent(MicroAgent):
    """
    L1-12 | Language Complexity Agent
    ──────────────────────────────────────────────────────────────────
    OWNS: Vocabulary sophistication, sentence structure complexity,
          cognitive load of expression, and communication patterns.

    WHY THIS MATTERS: Language IS thinking. Complexity of expression
    is a proxy for cognitive sophistication, education level, and how
    the person organizes their thoughts. It also reveals context
    switching (formal vs casual, stressed vs relaxed).
    """

    AGENT_ID = "L1-12-language"
    AGENT_NAME = "Language Complexity Agent"
    DIMENSION = "Vocabulary sophistication, sentence structure, cognitive expression complexity"
    RAG_QUERIES = [
        "therefore consequently furthermore notwithstanding",
        "complex sophisticated nuanced multifaceted",
        "yeah lol idk tbh basically literally",
        "I think I mean kind of sort of basically",
        "elaborate explain detail describe carefully",
        "short quick fast response brief",
    ]
    TOP_K = 7

    @property
    def ANALYSIS_PROMPT(self) -> str:
        return """
ANALYSIS TASK: Language Complexity Profiling

Assess the sophistication, structure, and expressiveness of this person's language.

WHAT TO MEASURE:

1. VOCABULARY LEVEL:
   Simple: basic, everyday words only
   Moderate: varied vocabulary, some domain-specific terms
   Advanced: rich vocabulary, low-frequency words, precise terminology
   Expert: highly specialized language in one or more domains

2. SENTENCE STRUCTURE:
   Do they write in simple sentences? Or complex compound sentences?
   Do they use subordinate clauses, parentheticals, qualifications?
   Grammar quality (not judging, but as signal of cognitive style)

3. LOGICAL CONNECTORS:
   Do they use connectors showing logical relationships?
   (because, therefore, however, although, whereas, consequently)
   OR mostly simple juxtaposition without explicit logical links?

4. REGISTER SWITCHING:
   How much does their language change between:
   - Casual vs serious topics
   - Emotional vs analytical contexts
   - Confident vs uncertain situations
   Wide register range = high linguistic flexibility

5. HEDGING PATTERNS:
   Too much hedging ("I kind of maybe think...") = low confidence OR carefulness
   No hedging ("It's definitely...") = high confidence OR overconfidence
   Calibrated hedging = epistemic accuracy

6. PRECISION vs VAGUENESS:
   Do they use precise quantifiers? Or vague ones?
   ("about 80% of the time" vs "a lot")
   Precision signals analytical thinking AND attention to detail
"""

    @property
    def OUTPUT_INSTRUCTIONS(self) -> str:
        return """
Return this exact JSON structure:
{
  "rating": 0.0-1.0,
  "confidence": 0.0-1.0,
  "label": "ONE OF: SIMPLE | BELOW_AVERAGE | AVERAGE | ABOVE_AVERAGE | SOPHISTICATED | EXPERT",
  "summary": "2-3 sentences",
  "evidence_quotes": ["quote1", "quote2", "quote3"],
  "evidence_chunk_ids": ["id1"],
  "patterns_found": ["pattern1"],
  "counter_evidence": [],
  "contradictions_internal": [],
  "structured_data": {
    "vocabulary_level": "SIMPLE | MODERATE | ADVANCED | EXPERT",
    "sentence_complexity": "SIMPLE | MODERATE | COMPLEX",
    "logical_connector_frequency": "LOW | MODERATE | HIGH",
    "register_flexibility": "RIGID | MODERATE | FLEXIBLE | HIGHLY_FLEXIBLE",
    "hedging_level": "NONE | LOW | CALIBRATED | EXCESSIVE",
    "precision_vs_vagueness": "VAGUE | MODERATE | PRECISE",
    "domain_expertise_signals": ["domains where expert language appears"]
  }
}"""

    def parse_structured_data(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "vocabulary_level": raw.get("vocabulary_level", "MODERATE"),
            "sentence_complexity": raw.get("sentence_complexity", "MODERATE"),
            "logical_connector_frequency": raw.get("logical_connector_frequency", "MODERATE"),
            "register_flexibility": raw.get("register_flexibility", "MODERATE"),
            "hedging_level": raw.get("hedging_level", "CALIBRATED"),
            "precision_vs_vagueness": raw.get("precision_vs_vagueness", "MODERATE"),
            "domain_expertise_signals": raw.get("domain_expertise_signals", []),
        }


class MemoryPatternAgent(MicroAgent):
    """
    L1-13 | Memory Pattern Agent
    ──────────────────────────────────────────────────────────────────
    OWNS: How the person references and reconstructs their past.
          Selective memory patterns. Narrative consistency over time.

    WHAT IT LOOKS FOR:
    - How far back they reference (long-term vs short-term memory dominance)
    - Whether past memories are positive or negative (mood-congruent memory)
    - Narrative consistency: do they tell the same story the same way?
    - Reframing: do memories change meaning over time in their telling?
    - What they emphasize vs omit in stories
    """

    AGENT_ID = "L1-13-memory"
    AGENT_NAME = "Memory Pattern Agent"
    DIMENSION = "How past is referenced, memory selectivity, narrative consistency"
    RAG_QUERIES = [
        "I remember when back then years ago childhood",
        "last time this happened before when I was",
        "I always have always been since I was young",
        "I used to I was different before change",
        "story told about past event experience",
        "I keep thinking about still remember can't forget",
    ]
    TOP_K = 7

    @property
    def ANALYSIS_PROMPT(self) -> str:
        return """
ANALYSIS TASK: Memory Pattern Analysis

Examine how this person references, constructs, and uses their past.

WHAT TO ASSESS:

1. MEMORY TIMEFRAME PREFERENCE:
   Do they reference distant past? Recent past? Childhood?
   What time period do they draw on most heavily for self-understanding?

2. MEMORY VALENCE SELECTIVITY:
   Do they preferentially recall positive or negative memories?
   Mood-congruent memory = depressed people remember more negatives
   Look for systematic slant in what gets recalled

3. NARRATIVE CONSISTENCY:
   If the same event or period is mentioned multiple times,
   is it told consistently? Or does the story change/evolve?
   Major narrative shifts can indicate memory reconstruction or emotional processing.

4. WHAT MEMORIES ARE USED FOR:
   - Justifying current behavior ("I've always been like this")
   - Explaining relationships ("because of what happened with X")
   - Building identity ("I'm someone who...")
   - Processing unresolved events (returning to them repeatedly)

5. RUMINATION vs INTEGRATION:
   Rumination: replaying the same memory with same emotional charge
   Integration: referencing the past with decreased emotional intensity, lessons drawn

6. MEMORY GAPS:
   Are there expected memory references that are ABSENT?
   (e.g., no mention of a parent despite relevant context)
   Gaps can indicate avoidance of painful material
"""

    @property
    def OUTPUT_INSTRUCTIONS(self) -> str:
        return """
Return this exact JSON structure:
{
  "rating": 0.0-1.0,
  "confidence": 0.0-1.0,
  "label": "Brief description of memory pattern",
  "summary": "2-3 sentences",
  "evidence_quotes": ["quote1", "quote2", "quote3"],
  "evidence_chunk_ids": ["id1"],
  "patterns_found": ["pattern1"],
  "counter_evidence": [],
  "contradictions_internal": [],
  "structured_data": {
    "memory_timeframe_preference": "DISTANT_PAST | RECENT | MIXED",
    "memory_valence": "NEGATIVE_SELECTIVE | NEUTRAL | POSITIVE_SELECTIVE | BALANCED",
    "rumination_vs_integration": "RUMINATION | PROCESSING | INTEGRATION",
    "key_recurring_memories": ["event/period that keeps coming up"],
    "memory_function": ["justifying behavior", "identity building", "processing"],
    "narrative_consistency": "INCONSISTENT | MODERATE | CONSISTENT",
    "suspected_avoidance_areas": ["topics/periods notably absent"]
  }
}"""

    def parse_structured_data(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "memory_timeframe_preference": raw.get("memory_timeframe_preference", "MIXED"),
            "memory_valence": raw.get("memory_valence", "BALANCED"),
            "rumination_vs_integration": raw.get("rumination_vs_integration", "PROCESSING"),
            "key_recurring_memories": raw.get("key_recurring_memories", []),
            "memory_function": raw.get("memory_function", []),
            "narrative_consistency": raw.get("narrative_consistency", "MODERATE"),
            "suspected_avoidance_areas": raw.get("suspected_avoidance_areas", []),
        }


# ══════════════════════════════════════════════════════════════════════
# BELIEFS LAYER
# ══════════════════════════════════════════════════════════════════════

class CoreBeliefAgent(MicroAgent):
    """
    L1-14 | Core Belief Extractor Agent
    ──────────────────────────────────────────────────────────────────
    OWNS: The deepest implicit rules about self, others, and the world.
          These are the operating system — mostly invisible, always running.

    WHY THIS MATTERS: This is arguably the highest-leverage agent.
    Core beliefs drive perception, emotion, motivation, and behavior.
    They feel like FACTS, not beliefs, which is what makes them powerful.
    "I'm not good enough" filters out evidence of adequacy.
    "People can't be trusted" makes genuine intimacy impossible.

    WHAT IT LOOKS FOR:
    - Absolute language (always, never, everyone, nobody, impossible)
    - Generalization patterns from specific events to universal rules
    - Identity-level statements ("I'm the type of person who...")
    - Self-worth signals ("I don't deserve", "why would anyone want to")
    - Catastrophic predictions stated as fact
    - Core schema categories: worthiness, lovability, competence, safety
    """

    AGENT_ID = "L1-14-core-beliefs"
    AGENT_NAME = "Core Belief Extractor Agent"
    DIMENSION = "Implicit core beliefs about self, others, and world — the operating system"
    RAG_QUERIES = [
        "I always I never I can't I'm not able to",
        "I don't deserve I'm not worth I'm not good enough",
        "people always people never nobody ever everyone",
        "I'm the type of person who I've always been",
        "world is dangerous can't trust unsafe bad",
        "impossible pointless what's the point never works",
        "I'm stupid I'm ugly I'm a failure I'm broken",
        "I'm capable I deserve I can do I'm strong",
    ]
    TOP_K = 10

    @property
    def ANALYSIS_PROMPT(self) -> str:
        return """
ANALYSIS TASK: Core Belief Extraction

This is the most critical and difficult analysis in the system.
Your job is to surface the IMPLICIT rules this person operates by —
not what they say they believe, but what their language REVEALS they believe.

EXTRACTION METHOD:

1. ABSOLUTE LANGUAGE SCAN:
   Every time they use: always, never, everyone, nobody, impossible, can't, won't ever
   These words mark where a belief has been overgeneralized into a rule.
   Extract the underlying rule being expressed.

2. SELF-TALK PATTERNS:
   How do they describe themselves? (I am X, I'm the type who, I always...)
   Extract: beliefs about own worth, competence, lovability, capability

3. PREDICTION PATTERNS:
   What do they predict will happen? These reveal their models of the world.
   Consistent negative predictions = core belief about world being dangerous/hostile
   Consistent positive = core belief in own efficacy / world's benevolence

4. SCHEMA CATEGORIES TO MAP:
   WORTHINESS: "Do I deserve good things?"
   LOVABILITY: "Am I the type of person people want to be with?"
   COMPETENCE: "Can I do the things that matter?"
   SAFETY: "Is the world fundamentally safe or dangerous?"
   TRUST: "Can people be relied on?"
   CONTROL: "Do my actions matter? Can I affect outcomes?"

5. DISTINGUISH SURFACE BELIEFS FROM CORE BELIEFS:
   Surface: "I don't like my job" (situational)
   Core: "I'll never be successful" (identity/universal)

6. MARK CONFIDENCE:
   High confidence beliefs have multiple pieces of evidence
   Low confidence = only 1-2 signals

IMPORTANT: Also look for EMPOWERING core beliefs, not just limiting ones.
Some people have genuinely strong positive core beliefs that predict resilience.
"""

    @property
    def OUTPUT_INSTRUCTIONS(self) -> str:
        return """
Return this exact JSON structure:
{
  "rating": 0.0-1.0,
  "confidence": 0.0-1.0,
  "label": "Overall belief architecture summary",
  "summary": "2-3 sentences describing the core belief structure",
  "evidence_quotes": ["quote1", "quote2", "quote3"],
  "evidence_chunk_ids": ["id1"],
  "patterns_found": ["belief pattern 1"],
  "counter_evidence": [],
  "contradictions_internal": ["belief A contradicts belief B"],
  "structured_data": {
    "beliefs": [
      {
        "belief_statement": "I am not good enough for success",
        "belief_type": "self | others | world | future",
        "valence": "limiting | empowering | neutral",
        "schema_category": "worthiness | lovability | competence | safety | trust | control",
        "confidence": 0.0-1.0,
        "evidence": ["exact quote implying this belief"],
        "frequency": 3
      }
    ],
    "dominant_schema": "worthiness | lovability | competence | safety | trust | control",
    "overall_world_model": "safe_benevolent | neutral | hostile_dangerous",
    "self_worth_baseline": "LOW | MODERATE | HIGH | FLUCTUATING"
  }
}"""

    def parse_structured_data(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "beliefs": raw.get("beliefs", []),
            "dominant_schema": raw.get("dominant_schema", "UNKNOWN"),
            "overall_world_model": raw.get("overall_world_model", "neutral"),
            "self_worth_baseline": raw.get("self_worth_baseline", "MODERATE"),
        }


class ValuesHierarchyAgent(MicroAgent):
    """
    L1-15 | Values Hierarchy Agent
    ──────────────────────────────────────────────────────────────────
    OWNS: Identifying the person's ACTUAL values — not what they say
          they value, but what their behavior reveals they prioritize.

    WHY THIS MATTERS: There is almost always a gap between stated values
    and behavioral values. Someone who says family is #1 but works 80
    hours per week actually prioritizes achievement. This gap is the
    source of enormous internal conflict.

    WHAT IT LOOKS FOR:
    - Where they spend time (discussed topics, activities mentioned)
    - What they sacrifice other things for
    - What generates guilt (indicates violated value)
    - What generates pride (indicates honored value)
    - What angers them in others (projected values)
    - Stated values vs. behavioral evidence of different values
    """

    AGENT_ID = "L1-15-values"
    AGENT_NAME = "Values Hierarchy Agent"
    DIMENSION = "Actual behavioral values, stated vs real values, priorities hierarchy"
    RAG_QUERIES = [
        "important to me matters most I care about value",
        "proud of achievement accomplishment worked hard",
        "guilty feel bad should have neglected failed",
        "angry at unfair wrong violated principle",
        "sacrifice gave up chose instead of priority",
        "spending time working on focused on doing",
        "family friends relationship love connection",
        "success money career ambition status",
    ]
    TOP_K = 9

    @property
    def ANALYSIS_PROMPT(self) -> str:
        return """
ANALYSIS TASK: Values Hierarchy Reconstruction

Identify this person's ACTUAL value hierarchy from behavioral evidence,
NOT from what they say their values are.

HOW TO EXTRACT REAL VALUES:

1. WHAT GENERATES PRIDE:
   The things they're proud of reveal what they value achieving.
   Look for self-congratulatory moments, things mentioned with satisfaction.

2. WHAT GENERATES GUILT/REGRET:
   Guilt only appears when you violate something you value.
   "I feel bad that I didn't call" = relationship/connection is a value
   "I haven't been to the gym" = health/discipline is a value

3. WHAT ANGERS THEM IN OTHERS:
   We get angry when others violate our values.
   "I can't stand dishonesty" = honesty is a core value
   "I hate when people are lazy" = work ethic / achievement is a value

4. TIME AND ATTENTION ALLOCATION:
   What topics come up most? What activities are mentioned?
   These reveal where their actual investment goes.

5. SACRIFICE EVIDENCE:
   What have they given up for something else?
   The thing chosen over other things reveals hierarchy.

6. STATED vs BEHAVIORAL GAP:
   Does stated value X have behavioral backing?
   If someone says "I value health" but the only health mention is guilt about not exercising — there's a gap.

RANK the top 7 values from strongest to weakest behavioral evidence.
Flag any stated-vs-actual value conflicts.
"""

    @property
    def OUTPUT_INSTRUCTIONS(self) -> str:
        return """
Return this exact JSON structure:
{
  "rating": 0.0-1.0,
  "confidence": 0.0-1.0,
  "label": "Brief characterization of value system",
  "summary": "2-3 sentences describing values hierarchy",
  "evidence_quotes": ["quote1", "quote2", "quote3"],
  "evidence_chunk_ids": ["id1"],
  "patterns_found": ["pattern1"],
  "counter_evidence": [],
  "contradictions_internal": [],
  "structured_data": {
    "values_ranked": [
      {
        "value": "Achievement",
        "rank": 1,
        "behavioral_evidence": ["example of behavior showing this value"],
        "stated_vs_actual_gap": false
      }
    ],
    "value_conflicts": ["Value A vs Value B — creates tension around X"],
    "dominant_value_domain": "achievement | relationships | security | pleasure | meaning | status | autonomy",
    "integrity_score": 0.0-1.0
  }
}"""

    def parse_structured_data(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "values_ranked": raw.get("values_ranked", []),
            "value_conflicts": raw.get("value_conflicts", []),
            "dominant_value_domain": raw.get("dominant_value_domain", "UNKNOWN"),
            "integrity_score": raw.get("integrity_score", 0.5),
        }


class WorldviewAgent(MicroAgent):
    """
    L1-16 | Worldview Agent
    ──────────────────────────────────────────────────────────────────
    OWNS: The person's fundamental model of how the world works.
          Abundance vs scarcity. Trust vs threat. Zero-sum vs positive-sum.
          Safety vs danger as default assumption.
    """

    AGENT_ID = "L1-16-worldview"
    AGENT_NAME = "Worldview Agent"
    DIMENSION = "Fundamental world model: abundance/scarcity, trust/threat, safety/danger lens"
    RAG_QUERIES = [
        "world is dangerous threat can't trust unsafe",
        "people are generally good kind helpful trustworthy",
        "competition zero sum win lose scarce limited",
        "opportunity possibilities can grow enough for everyone",
        "unfair rigged system against me disadvantaged",
        "fair chance opportunity if you work hard possible",
        "society broken corrupt wrong getting worse",
    ]
    TOP_K = 7

    @property
    def ANALYSIS_PROMPT(self) -> str:
        return """
ANALYSIS TASK: Worldview Architecture

Map the person's fundamental assumptions about how the world works.
These are the deepest cognitive frames — not what they say they believe,
but the assumptions that appear consistently in how they describe reality.

DIMENSIONS TO MAP:

1. ABUNDANCE vs SCARCITY MINDSET:
   Scarcity: "There's not enough" — limited resources, zero-sum thinking,
   others' success threatens your own, defensive hoarding behavior
   Abundance: "There's enough" — positive sum thinking, collaboration,
   others' success doesn't threaten yours

2. TRUST vs THREAT BASELINE:
   Trust: Default assumption that people are OK until proven otherwise
   Threat: Default assumption that people are self-interested, deceptive, or dangerous
   (This is the most consequential worldview dimension for relationships)

3. LOCUS OF CONTROL:
   Internal: "My actions determine my outcomes"
   External: "Forces outside me control what happens to me"
   Mixed: Context-dependent

4. FAIR WORLD vs UNFAIR WORLD:
   Fair world belief: hard work + merit = success
   Unfair world: rigged systems, luck, privilege determine outcomes
   Both can be accurate OR distorted — look for systematic patterns

5. HUMAN NATURE ASSUMPTION:
   "People are fundamentally good" vs "people are fundamentally selfish"
   This shapes every social interaction.

6. PROGRESS vs DECLINE:
   Is the world / their life getting better or worse?
   Temporal optimism vs temporal pessimism
"""

    @property
    def OUTPUT_INSTRUCTIONS(self) -> str:
        return """
Return this exact JSON structure:
{
  "rating": 0.0-1.0,
  "confidence": 0.0-1.0,
  "label": "Brief worldview characterization",
  "summary": "2-3 sentences",
  "evidence_quotes": ["quote1", "quote2", "quote3"],
  "evidence_chunk_ids": ["id1"],
  "patterns_found": ["pattern1"],
  "counter_evidence": [],
  "contradictions_internal": [],
  "structured_data": {
    "abundance_vs_scarcity": "SCARCITY | MODERATE_SCARCITY | NEUTRAL | MODERATE_ABUNDANCE | ABUNDANCE",
    "trust_vs_threat": "THREAT | SUSPICIOUS | NEUTRAL | TRUSTING | HIGHLY_TRUSTING",
    "locus_of_control": "EXTERNAL | MOSTLY_EXTERNAL | MIXED | MOSTLY_INTERNAL | INTERNAL",
    "fair_world_belief": 0.0-1.0,
    "human_nature_assumption": "SELFISH | MIXED | GOOD",
    "temporal_optimism": "DECLINING | NEUTRAL | IMPROVING",
    "worldview_consistency": "INCONSISTENT | MODERATE | CONSISTENT"
  }
}"""

    def parse_structured_data(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "abundance_vs_scarcity": raw.get("abundance_vs_scarcity", "NEUTRAL"),
            "trust_vs_threat": raw.get("trust_vs_threat", "NEUTRAL"),
            "locus_of_control": raw.get("locus_of_control", "MIXED"),
            "fair_world_belief": raw.get("fair_world_belief", 0.5),
            "human_nature_assumption": raw.get("human_nature_assumption", "MIXED"),
            "temporal_optimism": raw.get("temporal_optimism", "NEUTRAL"),
            "worldview_consistency": raw.get("worldview_consistency", "MODERATE"),
        }


# ══════════════════════════════════════════════════════════════════════
# MOTIVATION LAYER
# ══════════════════════════════════════════════════════════════════════

class NeedsSignalAgent(MicroAgent):
    """
    L1-17 | Needs Signal Agent
    ──────────────────────────────────────────────────────────────────
    OWNS: Which needs are dominant and which are chronically unmet.
          Framework: Maslow's hierarchy + Self-Determination Theory.

    WHY THIS MATTERS: Unmet needs don't disappear — they become
    unconscious behavioral drivers. The person craving belonging
    makes poor relationship decisions. The person starved for safety
    takes no risks. Knowing unmet needs explains otherwise puzzling behavior.

    SDT's 3 Core Needs: Autonomy, Competence, Relatedness
    Maslow: Physiological, Safety, Belonging, Esteem, Self-actualization
    """

    AGENT_ID = "L1-17-needs"
    AGENT_NAME = "Needs Signal Agent"
    DIMENSION = "Dominant and unmet psychological needs — Maslow + SDT framework"
    RAG_QUERIES = [
        "belong connected part of included love",
        "respect recognized valued appreciated matter",
        "safe secure stable certain protect",
        "independent free autonomous own choice control",
        "capable competent good at succeed achieve",
        "lonely isolated excluded rejected left out",
        "meaning purpose why reason deeper",
        "approval validation praise liked",
    ]
    TOP_K = 9

    @property
    def ANALYSIS_PROMPT(self) -> str:
        return """
ANALYSIS TASK: Psychological Needs Assessment

Identify which of the core human needs appear DOMINANT (actively sought)
and which appear UNMET (craved but frustrated) in this person's life.

THE NEEDS FRAMEWORK:

MASLOW LEVELS:
1. SAFETY/SECURITY: Need for stability, predictability, protection from threat
2. BELONGING/LOVE: Need for connection, acceptance, being part of something
3. ESTEEM: Need for respect, recognition, competence, achievement
4. SELF-ACTUALIZATION: Need to grow, create, realize potential

SDT CORE NEEDS:
1. AUTONOMY: Need to feel self-directed, acting from own values/choice
2. COMPETENCE: Need to feel effective, capable, able to produce results
3. RELATEDNESS: Need to feel meaningfully connected to others

FOR EACH NEED, DETERMINE:
- Is it being MET (satisfied language around this area)?
- Is it DOMINANT (constantly sought, much energy directed here)?
- Is it CHRONICALLY UNMET (frustrated, absent, longed for)?
- Is it SUPPRESSED (the person has given up seeking it)?

SIGNALS OF UNMET NEEDS:
- Frequent mention without satisfaction
- Attempts to meet it in indirect ways
- Compensatory behavior (e.g., seeking esteem through status symbols if relatedness is unmet)
- Direct expressions of longing
"""

    @property
    def OUTPUT_INSTRUCTIONS(self) -> str:
        return """
Return this exact JSON structure:
{
  "rating": 0.0-1.0,
  "confidence": 0.0-1.0,
  "label": "Brief characterization of needs profile",
  "summary": "2-3 sentences",
  "evidence_quotes": ["quote1", "quote2", "quote3"],
  "evidence_chunk_ids": ["id1"],
  "patterns_found": ["pattern1"],
  "counter_evidence": [],
  "contradictions_internal": [],
  "structured_data": {
    "needs_map": {
      "safety": "MET | DOMINANT | UNMET | SUPPRESSED",
      "belonging": "MET | DOMINANT | UNMET | SUPPRESSED",
      "esteem": "MET | DOMINANT | UNMET | SUPPRESSED",
      "autonomy": "MET | DOMINANT | UNMET | SUPPRESSED",
      "competence": "MET | DOMINANT | UNMET | SUPPRESSED",
      "relatedness": "MET | DOMINANT | UNMET | SUPPRESSED",
      "meaning": "MET | DOMINANT | UNMET | SUPPRESSED"
    },
    "most_critical_unmet_need": "name of the need most urgently unmet",
    "compensation_patterns": ["how they try to meet needs indirectly"],
    "needs_evidence": {"need_name": ["quote showing this need"]}
  }
}"""

    def parse_structured_data(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "needs_map": raw.get("needs_map", {}),
            "most_critical_unmet_need": raw.get("most_critical_unmet_need", "UNKNOWN"),
            "compensation_patterns": raw.get("compensation_patterns", []),
            "needs_evidence": raw.get("needs_evidence", {}),
        }


class DriveTypeAgent(MicroAgent):
    """
    L1-18 | Drive Type Agent
    ──────────────────────────────────────────────────────────────────
    OWNS: The quality and source of motivation energy.
          Intrinsic vs extrinsic. Fear-based vs growth-based.
          What actually gets this person moving.

    WHY THIS MATTERS: People driven by fear accomplish things but burn out.
    People driven by validation are unstable. People driven by intrinsic
    interest sustain for life. Knowing the drive type predicts which
    goals will be sustained vs abandoned.
    """

    AGENT_ID = "L1-18-drive"
    AGENT_NAME = "Drive Type Agent"
    DIMENSION = "Intrinsic vs extrinsic motivation, fear vs growth drive, motivational fuel type"
    RAG_QUERIES = [
        "I love doing this passionate interest enjoy for its own sake",
        "I should I have to I must I need to obligation",
        "what will people think approval recognition reward",
        "afraid of failing scared of what happens if I don't",
        "I want to grow improve become better learning",
        "avoiding shame embarrassment what others think",
        "because it matters to me important personally meaningful",
        "pushing myself challenge difficult stretch",
    ]
    TOP_K = 8

    @property
    def ANALYSIS_PROMPT(self) -> str:
        return """
ANALYSIS TASK: Motivational Drive Type Analysis

Identify the QUALITY and SOURCE of this person's motivation.

THE MOTIVATION SPECTRUM:

INTRINSIC MOTIVATION (sustainable, self-generating):
- Doing something because it's inherently interesting or meaningful
- Language: "I love", "I enjoy", "it's fascinating", "I'm curious about"
- No external justification needed

EXTRINSIC MOTIVATION (can work, but fragile):
- Doing something for external reward or to avoid punishment
- Language: "so that people will", "to get the grade/promotion/approval"
- Depends on external conditions remaining constant

FEAR-BASED DRIVE (powerful but toxic):
- Moving AWAY from something bad rather than TOWARD something good
- Language: "I can't let X happen", "terrified of", "have to avoid"
- Creates urgency but generates anxiety, not fulfillment

SHAME-BASED DRIVE (the most destructive):
- "If I fail, it means I'm worthless / unlovable"
- Language: "I can't let anyone see me fail", "embarrassed to admit"
- High performance possible but devastating psychological cost

GROWTH-BASED DRIVE (ideal):
- Moving toward becoming a better version of self
- Language: "I want to improve", "I'm working on", "learning to"
- Connected to intrinsic interest + external impact

OBLIGATION/DUTY DRIVE (mixed):
- "I have to because it's what's expected/right"
- Can be healthy (aligned with genuine values) or coercive

ASSESS RATIO: What % of motivation is each type?
What domains use which drive?
"""

    @property
    def OUTPUT_INSTRUCTIONS(self) -> str:
        return """
Return this exact JSON structure:
{
  "rating": 0.0-1.0,
  "confidence": 0.0-1.0,
  "label": "Primary drive type characterization",
  "summary": "2-3 sentences",
  "evidence_quotes": ["quote1", "quote2", "quote3"],
  "evidence_chunk_ids": ["id1"],
  "patterns_found": ["pattern1"],
  "counter_evidence": [],
  "contradictions_internal": [],
  "structured_data": {
    "primary_drive": "INTRINSIC | EXTRINSIC | FEAR | SHAME | GROWTH | OBLIGATION | MIXED",
    "drive_distribution": {
      "intrinsic": 0.0-1.0,
      "extrinsic": 0.0-1.0,
      "fear_based": 0.0-1.0,
      "shame_based": 0.0-1.0,
      "growth_based": 0.0-1.0
    },
    "sustainability_score": 0.0-1.0,
    "domain_drive_mapping": {"work": "fear", "relationships": "intrinsic"},
    "motivational_language_samples": {"intrinsic": ["quote"], "fear": ["quote"]}
  }
}"""

    def parse_structured_data(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "primary_drive": raw.get("primary_drive", "MIXED"),
            "drive_distribution": raw.get("drive_distribution", {}),
            "sustainability_score": raw.get("sustainability_score", 0.5),
            "domain_drive_mapping": raw.get("domain_drive_mapping", {}),
            "motivational_language_samples": raw.get("motivational_language_samples", {}),
        }


class PersistenceAgent(MicroAgent):
    """
    L1-19 | Persistence Agent
    ──────────────────────────────────────────────────────────────────
    OWNS: Grit signals, setback response patterns, resilience indicators.
          How the person responds when things don't go as planned.
    """

    AGENT_ID = "L1-19-persistence"
    AGENT_NAME = "Persistence Agent"
    DIMENSION = "Grit, setback response, resilience, persistence under difficulty"
    RAG_QUERIES = [
        "kept going despite difficult didn't give up continued",
        "gave up stopped quit abandoned failed gave in",
        "bounced back recovered resilient handled it",
        "I'll keep trying not going to stop determined",
        "setback failure obstacle challenge problem",
        "I'm done can't anymore it's too hard not worth it",
    ]
    TOP_K = 7

    @property
    def ANALYSIS_PROMPT(self) -> str:
        return """
ANALYSIS TASK: Persistence & Resilience Assessment

Analyze how this person responds to difficulty, setbacks, and sustained challenge.

WHAT TO LOOK FOR:

1. SETBACK RESPONSE PATTERN:
   When things go wrong — what's the immediate reaction?
   And what's the response AFTER the immediate reaction?
   (First reaction can be emotional flooding even in gritty people)

2. PERSISTENCE EVIDENCE:
   Has this person continued working on something despite obstacles?
   Multi-message arcs where they return to a difficult goal
   "I'm still trying", "even though it's hard", "keeping at it"

3. ABANDONMENT PATTERNS:
   Projects, goals, or relationships mentioned as abandoned
   Language of giving up: "it's not worth it", "I'm done", "pointless"
   Are these healthy exits or avoidance patterns?

4. ATTRIBUTION AFTER FAILURE:
   Do they blame themselves? External factors? Learn and adjust?
   Healthy: "I failed because X, here's what I'll do differently"
   Unhealthy: "I'm just not the type who can do this"

5. RECOVERY SPEED:
   How long do setbacks derail them?
   Do they bounce back or stay down?

6. CONSISTENCY vs INCONSISTENCY:
   Are they gritty in some domains but quitting in others?
   Domain-specific persistence patterns matter a lot.

Rate 0.0 = immediately abandons on first obstacle, no resilience
Rate 0.5 = moderate grit, some resilience with notable avoidance patterns
Rate 1.0 = exceptional grit, adapts after failure, sustains despite hardship
"""

    @property
    def OUTPUT_INSTRUCTIONS(self) -> str:
        return """
Return this exact JSON structure:
{
  "rating": 0.0-1.0,
  "confidence": 0.0-1.0,
  "label": "ONE OF: QUITTER | LOW_GRIT | MODERATE_GRIT | HIGH_GRIT | EXCEPTIONAL_GRIT",
  "summary": "2-3 sentences",
  "evidence_quotes": ["quote1", "quote2", "quote3"],
  "evidence_chunk_ids": ["id1"],
  "patterns_found": ["pattern1"],
  "counter_evidence": [],
  "contradictions_internal": [],
  "structured_data": {
    "grit_level": 0.0-1.0,
    "recovery_speed": "SLOW | MODERATE | FAST",
    "setback_attribution": "SELF_BLAME | BALANCED | EXTERNAL_BLAME",
    "abandonment_instances": ["thing they quit + context"],
    "persistence_instances": ["thing they kept going with despite difficulty"],
    "domain_grit_map": {"work": 0.8, "relationships": 0.3},
    "failure_reframe_ability": "NONE | WEAK | MODERATE | STRONG"
  }
}"""

    def parse_structured_data(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "grit_level": raw.get("grit_level", 0.5),
            "recovery_speed": raw.get("recovery_speed", "MODERATE"),
            "setback_attribution": raw.get("setback_attribution", "BALANCED"),
            "abandonment_instances": raw.get("abandonment_instances", []),
            "persistence_instances": raw.get("persistence_instances", []),
            "domain_grit_map": raw.get("domain_grit_map", {}),
            "failure_reframe_ability": raw.get("failure_reframe_ability", "MODERATE"),
        }


# ══════════════════════════════════════════════════════════════════════
# BEHAVIOR LAYER
# ══════════════════════════════════════════════════════════════════════

class HabitLoopAgent(MicroAgent):
    """
    L1-20 | Habit Loop Agent
    ──────────────────────────────────────────────────────────────────
    OWNS: Identifying recurring cue-routine-reward loops.
          The automated behavioral patterns that bypass decision-making.

    WHY THIS MATTERS: ~40% of daily behavior is habitual — running on
    automatic without conscious choice. Understanding the habit loops
    explains behavior that seems puzzling, random, or self-destructive.
    They persist because the reward is real even when the routine is harmful.
    """

    AGENT_ID = "L1-20-habits"
    AGENT_NAME = "Habit Loop Agent"
    DIMENSION = "Cue-routine-reward loops, automated behavioral patterns"
    RAG_QUERIES = [
        "whenever I feel I usually always then I",
        "habit routine always do same every time",
        "when stressed when bored when anxious I turn to",
        "automatically without thinking just do it reflexively",
        "I've been trying to stop can't break the habit",
        "social media phone scrolling distraction",
        "every morning evening night before after",
        "comfort food drink smoke exercise when feeling",
    ]
    TOP_K = 8

    @property
    def ANALYSIS_PROMPT(self) -> str:
        return """
ANALYSIS TASK: Habit Loop Identification

Map the automated behavioral patterns — cue → routine → reward loops.

THE HABIT LOOP MODEL:
CUE: The trigger that automatically initiates the behavior
ROUTINE: The behavior itself
REWARD: The payoff that reinforces the loop

WHAT TO LOOK FOR:

1. CONDITIONAL BEHAVIOR PATTERNS:
   "When X happens, I always do Y"
   "Whenever I feel Z, I automatically ..."
   These conditional patterns ARE habit loops waiting to be mapped.

2. TEMPORAL ROUTINES:
   Fixed-time behaviors (morning routine, evening wind-down)
   These are habits triggered by time as cue.

3. EMOTIONAL TRIGGERS:
   What behaviors appear consistently after specific emotions?
   (anxiety → scrolling phone; boredom → eating; stress → isolating)

4. IDENTIFY THE HIDDEN REWARD:
   This is the hard part. The reward of a habit isn't always obvious.
   Scrolling phone when bored → reward: stimulation / social connection
   Procrastinating → reward: anxiety relief (not doing the scary thing)
   Picking fights → reward: feeling powerful / getting attention

5. PRODUCTIVE HABITS:
   Don't only look for negative loops.
   Identify healthy automated routines (exercise, learning, gratitude)

6. BREAKING ATTEMPTS:
   Times they tried to break a habit and what happened.
   This reveals the habit's strength and underlying reward.

For each loop: identify CUE clearly, ROUTINE clearly, REWARD explicitly.
Don't list it if you can't name all three.
"""

    @property
    def OUTPUT_INSTRUCTIONS(self) -> str:
        return """
Return this exact JSON structure:
{
  "rating": 0.0-1.0,
  "confidence": 0.0-1.0,
  "label": "Brief characterization of habit landscape",
  "summary": "2-3 sentences",
  "evidence_quotes": ["quote1", "quote2", "quote3"],
  "evidence_chunk_ids": ["id1"],
  "patterns_found": ["pattern1"],
  "counter_evidence": [],
  "contradictions_internal": [],
  "structured_data": {
    "habit_loops": [
      {
        "cue": "feeling anxious",
        "routine": "check social media",
        "reward": "distraction from anxiety",
        "frequency": "frequent | occasional | rare | constant",
        "valence": "healthy | neutral | unhealthy",
        "evidence": ["quote showing this loop"]
      }
    ],
    "dominant_emotional_triggers": ["anxiety", "boredom"],
    "healthy_habit_count": 0,
    "unhealthy_habit_count": 0,
    "habit_awareness_level": "UNAWARE | AWARE | WORKING_ON_IT"
  }
}"""

    def parse_structured_data(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "habit_loops": raw.get("habit_loops", []),
            "dominant_emotional_triggers": raw.get("dominant_emotional_triggers", []),
            "healthy_habit_count": raw.get("healthy_habit_count", 0),
            "unhealthy_habit_count": raw.get("unhealthy_habit_count", 0),
            "habit_awareness_level": raw.get("habit_awareness_level", "AWARE"),
        }


class DecisionStyleAgent(MicroAgent):
    """
    L1-21 | Decision Style Agent
    ──────────────────────────────────────────────────────────────────
    OWNS: How the person makes decisions. Impulsive vs deliberate.
          Risk orientation. Regret patterns. Decision quality signals.

    WHAT IT LOOKS FOR:
    - Deliberation language (pros/cons, weighing, considering)
    - Impulsive decision language ("I just decided", "without thinking")
    - Risk appetite (towards or away from uncertainty)
    - Decision regret patterns (what kinds of decisions they regret)
    - Reversibility preference (do they hedge, keep options open?)
    """

    AGENT_ID = "L1-21-decisions"
    AGENT_NAME = "Decision Style Agent"
    DIMENSION = "Decision-making style: impulsive vs deliberate, risk orientation, regret patterns"
    RAG_QUERIES = [
        "I decided I chose I'm going to made up my mind",
        "weighing pros cons considering thinking through options",
        "risky safe certain uncertain bet gamble chance",
        "I regret I shouldn't have wish I had chosen differently",
        "on impulse spontaneously suddenly just decided",
        "keeping options open hedging waiting to decide",
        "analysis paralysis can't decide too many options",
    ]
    TOP_K = 7

    @property
    def ANALYSIS_PROMPT(self) -> str:
        return """
ANALYSIS TASK: Decision Style Profiling

Map how this person makes decisions across different life domains.

KEY DIMENSIONS:

1. DELIBERATE vs IMPULSIVE:
   Deliberate: shows reasoning, considers alternatives, takes time
   Impulsive: "just decided", no deliberation shown, acts on emotion/impulse

2. RISK ORIENTATION:
   Risk-seeking: chooses uncertainty, bets, takes big leaps
   Risk-averse: chooses the safe option, avoids uncertainty
   Calibrated: context-dependent risk assessment

3. DECISION QUALITY SIGNALS:
   Do their decisions consistently lead to stated positive outcomes?
   Do they frequently make decisions they later regret?
   Type of regrets: action regrets (I did X) vs inaction regrets (I didn't do Y)

4. INFORMATION REQUIREMENTS:
   Maximizer: needs maximum information before deciding (can cause paralysis)
   Satisficer: decides when "good enough" is found

5. REVERSIBILITY PREFERENCE:
   Do they prefer reversible decisions? Or do they commit fully?
   Do they keep "escape hatches"?

6. DECISION DOMAINS:
   People are often impulsive in one domain and deliberate in another.
   Map: career decisions vs relationship decisions vs financial decisions vs social decisions.
"""

    @property
    def OUTPUT_INSTRUCTIONS(self) -> str:
        return """
Return this exact JSON structure:
{
  "rating": 0.0-1.0,
  "confidence": 0.0-1.0,
  "label": "PRIMARY DECISION STYLE",
  "summary": "2-3 sentences",
  "evidence_quotes": ["quote1", "quote2", "quote3"],
  "evidence_chunk_ids": ["id1"],
  "patterns_found": ["pattern1"],
  "counter_evidence": [],
  "contradictions_internal": [],
  "structured_data": {
    "deliberateness_score": 0.0-1.0,
    "risk_orientation": "RISK_AVERSE | CAUTIOUS | MODERATE | RISK_TOLERANT | RISK_SEEKING",
    "maximizer_vs_satisficer": "MAXIMIZER | SATISFICER | MIXED",
    "regret_pattern": "ACTION_REGRETS | INACTION_REGRETS | MIXED | FEW_REGRETS",
    "decision_quality_signal": "POOR | MODERATE | GOOD",
    "domain_decision_styles": {"relationships": "impulsive", "work": "deliberate"},
    "analysis_paralysis_tendency": true or false
  }
}"""

    def parse_structured_data(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "deliberateness_score": raw.get("deliberateness_score", 0.5),
            "risk_orientation": raw.get("risk_orientation", "MODERATE"),
            "maximizer_vs_satisficer": raw.get("maximizer_vs_satisficer", "MIXED"),
            "regret_pattern": raw.get("regret_pattern", "MIXED"),
            "decision_quality_signal": raw.get("decision_quality_signal", "MODERATE"),
            "domain_decision_styles": raw.get("domain_decision_styles", {}),
            "analysis_paralysis_tendency": raw.get("analysis_paralysis_tendency", False),
        }


# ══════════════════════════════════════════════════════════════════════
# SOCIAL LAYER
# ══════════════════════════════════════════════════════════════════════

class AttachmentStyleAgent(MicroAgent):
    """
    L1-22 | Attachment Style Agent
    ──────────────────────────────────────────────────────────────────
    OWNS: The person's fundamental relational wiring.
          Secure, anxious, avoidant, or disorganized attachment patterns.

    WHY THIS MATTERS: Attachment style is the single most predictive
    variable for relationship outcomes. It was formed in infancy/childhood
    from the primary caregiver relationship and operates automatically
    in all close relationships — often invisibly.

    WHAT IT LOOKS FOR:
    - Fear of abandonment / rejection sensitivity (anxious)
    - Deactivation of attachment needs / intimacy avoidance (avoidant)
    - Comfort with closeness and healthy dependency (secure)
    - Simultaneous desire and fear of closeness (disorganized)
    - How they describe relationships going wrong
    - What they do when they feel disconnected
    """

    AGENT_ID = "L1-22-attachment"
    AGENT_NAME = "Attachment Style Agent"
    DIMENSION = "Adult attachment style: secure/anxious/avoidant/disorganized patterns"
    RAG_QUERIES = [
        "afraid they'll leave scared of rejection abandoned",
        "I need space too close suffocating independent",
        "relationship close connection intimacy comfortable",
        "testing pushing away pulling closer mixed signals",
        "I don't need anyone fine alone don't get attached",
        "why don't they respond worried about relationship",
        "trust issues opening up letting people in vulnerable",
        "close relationship partner friend comfortable safe",
    ]
    TOP_K = 9

    @property
    def ANALYSIS_PROMPT(self) -> str:
        return """
ANALYSIS TASK: Attachment Style Assessment

Identify this person's adult attachment patterns across close relationships.

THE FOUR ATTACHMENT STYLES:

1. SECURE (comfortable with closeness AND independence):
   - Comfortable being vulnerable
   - Can rely on others without fear
   - Doesn't panic when connection is temporarily disrupted
   - Language: easy discussion of needs, balanced relationship descriptions

2. ANXIOUS (preoccupied with attachment):
   - Fear of abandonment, rejection sensitivity
   - Hyperactivation of attachment system (clingy, monitoring, needs reassurance)
   - Oscillates between idealization and devaluation
   - Language: "do they still care?", "why didn't they respond?", "I'm worried about us"

3. AVOIDANT (deactivates attachment needs):
   - Discomfort with closeness, intimacy avoidance
   - Values independence above connection
   - Deactivates when relationship gets too close
   - Language: "I need space", "I'm fine alone", "I don't really need people"

4. DISORGANIZED/FEARFUL (wants AND fears closeness):
   - Simultaneous desire and fear of intimacy
   - Unpredictable: sometimes clingy, sometimes withdrawing
   - Often linked to unresolved trauma
   - Language: contradictory statements about relationships ("I want connection but...")

LOOK FOR:
- Relationship conflict descriptions (HOW they describe what went wrong)
- What they do when they feel disconnected
- Their response to vulnerability in others
- Self-sufficiency claims vs actual behavior
"""

    @property
    def OUTPUT_INSTRUCTIONS(self) -> str:
        return """
Return this exact JSON structure:
{
  "rating": 0.0-1.0,
  "confidence": 0.0-1.0,
  "label": "Primary attachment style",
  "summary": "2-3 sentences",
  "evidence_quotes": ["quote1", "quote2", "quote3"],
  "evidence_chunk_ids": ["id1"],
  "patterns_found": ["pattern1"],
  "counter_evidence": [],
  "contradictions_internal": [],
  "structured_data": {
    "primary_style": "SECURE | ANXIOUS | AVOIDANT | DISORGANIZED",
    "confidence": 0.0-1.0,
    "secondary_style": "style name or null",
    "rejection_sensitivity": 0.0-1.0,
    "intimacy_comfort": 0.0-1.0,
    "independence_emphasis": 0.0-1.0,
    "relationship_patterns": ["pattern observed in how relationships are described"],
    "abandonment_fear_evidence": ["quote showing fear of abandonment"],
    "avoidance_evidence": ["quote showing avoidance of intimacy"]
  }
}"""

    def parse_structured_data(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "primary_style": raw.get("primary_style", "UNKNOWN"),
            "secondary_style": raw.get("secondary_style"),
            "rejection_sensitivity": raw.get("rejection_sensitivity", 0.5),
            "intimacy_comfort": raw.get("intimacy_comfort", 0.5),
            "independence_emphasis": raw.get("independence_emphasis", 0.5),
            "relationship_patterns": raw.get("relationship_patterns", []),
            "abandonment_fear_evidence": raw.get("abandonment_fear_evidence", []),
            "avoidance_evidence": raw.get("avoidance_evidence", []),
        }


class CommunicationStyleAgent(MicroAgent):
    """
    L1-23 | Communication Style Agent
    ──────────────────────────────────────────────────────────────────
    OWNS: How the person communicates — assertive, passive, aggressive,
          passive-aggressive, collaborative. Conflict approach. Listening signals.
    """

    AGENT_ID = "L1-23-communication"
    AGENT_NAME = "Communication Style Agent"
    DIMENSION = "Communication patterns: assertive/passive/collaborative, conflict approach"
    RAG_QUERIES = [
        "I need I want I feel directly stated clearly",
        "I don't want to cause problems maybe if possible",
        "you always you never your fault blame",
        "what do you think how about we let's discuss",
        "sarcasm passive aggressive indirect hinting",
        "conflict argument disagree fight difficult conversation",
        "listening understanding perspective seeing your point",
    ]
    TOP_K = 8

    @property
    def ANALYSIS_PROMPT(self) -> str:
        return """
ANALYSIS TASK: Communication Style Profiling

Map how this person expresses themselves and handles interpersonal dynamics.

THE COMMUNICATION STYLES:

1. ASSERTIVE (healthy):
   - Direct expression of needs/feelings without aggression
   - Uses "I statements" ("I feel", "I need", "I want")
   - Can say no, can disagree respectfully
   - Listens to understand, not just to respond

2. PASSIVE:
   - Avoids direct expression, indirect communication
   - Doesn't advocate for own needs
   - Agrees publicly, resents privately
   - "It's fine", "whatever you want", excessive accommodation

3. AGGRESSIVE:
   - Expresses needs at the expense of others
   - Blame language, "you always/never"
   - Dominating, interrupting, dismissing

4. PASSIVE-AGGRESSIVE:
   - Indirect expression of hostility
   - Sarcasm, silent treatment, subtle undermining
   - "I'm FINE" when clearly not fine
   - Doing the opposite of what was asked while technically complying

5. COLLABORATIVE:
   - Actively seeks to understand other perspective
   - "What do you think?", "help me understand"
   - Problem-solves together rather than winning

CONFLICT APPROACH:
- Avoidance: walks away, changes subject, refuses to engage
- Accommodation: gives in to end conflict
- Competition: must win the argument
- Compromise: meets in the middle
- Collaboration: works to solve the underlying problem
"""

    @property
    def OUTPUT_INSTRUCTIONS(self) -> str:
        return """
Return this exact JSON structure:
{
  "rating": 0.0-1.0,
  "confidence": 0.0-1.0,
  "label": "PRIMARY COMMUNICATION STYLE",
  "summary": "2-3 sentences",
  "evidence_quotes": ["quote1", "quote2", "quote3"],
  "evidence_chunk_ids": ["id1"],
  "patterns_found": ["pattern1"],
  "counter_evidence": [],
  "contradictions_internal": [],
  "structured_data": {
    "primary_style": "ASSERTIVE | PASSIVE | AGGRESSIVE | PASSIVE_AGGRESSIVE | COLLABORATIVE",
    "style_distribution": {"assertive": 0.0, "passive": 0.0, "aggressive": 0.0, "passive_aggressive": 0.0, "collaborative": 0.0},
    "conflict_approach": "AVOIDANCE | ACCOMMODATION | COMPETITION | COMPROMISE | COLLABORATION",
    "i_statement_frequency": "LOW | MODERATE | HIGH",
    "listening_signals": "POOR | MODERATE | GOOD | EXCELLENT",
    "context_style_switching": ["style in professional context", "style in personal conflict"],
    "communication_strengths": ["strength1"],
    "communication_gaps": ["gap1"]
  }
}"""

    def parse_structured_data(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "primary_style": raw.get("primary_style", "UNKNOWN"),
            "style_distribution": raw.get("style_distribution", {}),
            "conflict_approach": raw.get("conflict_approach", "UNKNOWN"),
            "i_statement_frequency": raw.get("i_statement_frequency", "MODERATE"),
            "listening_signals": raw.get("listening_signals", "MODERATE"),
            "context_style_switching": raw.get("context_style_switching", []),
            "communication_strengths": raw.get("communication_strengths", []),
            "communication_gaps": raw.get("communication_gaps", []),
        }


class SocialInfluenceAgent(MicroAgent):
    """
    L1-24 | Social Influence Agent
    ──────────────────────────────────────────────────────────────────
    OWNS: Susceptibility to social influence — conformity pressure,
          authority deference, social proof reliance. AND influence exerted.

    WHAT IT LOOKS FOR:
    - Changing behavior/opinion because of social pressure
    - "Everyone is doing it / saying it / thinks so"
    - Deference to authority figures
    - Independent thinking vs crowd-following
    - How they try to influence others
    """

    AGENT_ID = "L1-24-social-influence"
    AGENT_NAME = "Social Influence Agent"
    DIMENSION = "Social influence susceptibility: conformity, authority, social proof, plus influence style"
    RAG_QUERIES = [
        "everyone thinks people say everyone does popular",
        "expert said authority research studies proven",
        "changed my mind because they said others think",
        "I don't care what others think independent my own",
        "peer pressure following going along not standing out",
        "I convinced persuaded argued made them see",
        "FOMO missing out trend viral what others are doing",
    ]
    TOP_K = 7

    @property
    def ANALYSIS_PROMPT(self) -> str:
        return """
ANALYSIS TASK: Social Influence Assessment

Assess how much this person's behavior and beliefs are shaped by social forces,
AND how they try to influence others.

INFLUENCE SUSCEPTIBILITY:

1. CONFORMITY PRESSURE:
   Do they change behavior to fit in?
   Do they adopt opinions because others hold them?
   Signals: "everyone thinks", "nobody does that anymore", group identity language

2. AUTHORITY DEFERENCE:
   Do they accept claims primarily because of who said them?
   Blind authority deference vs healthy respect for expertise
   Signals: "the experts say", "studies show" without critical evaluation

3. SOCIAL PROOF:
   Does "everyone doing X" make X more appealing?
   FOMO language, trend-following, popularity as justification

4. INDEPENDENT THINKING:
   Evidence of holding unpopular opinions
   Explicitly going against consensus
   "I know this isn't popular but..."

5. PEER INFLUENCE:
   Specific individuals whose opinions carry disproportionate weight
   Changing decisions based on friend/partner/family input

INFLUENCE EXERTION (how THEY influence others):
- Rational argument: "here's why you should"
- Emotional appeal: making others feel
- Authority claims: "I know better"
- Social proof: "everyone agrees with me"
- Manipulation: guilt, pressure, obligation
"""

    @property
    def OUTPUT_INSTRUCTIONS(self) -> str:
        return """
Return this exact JSON structure:
{
  "rating": 0.0-1.0,
  "confidence": 0.0-1.0,
  "label": "Brief influence profile",
  "summary": "2-3 sentences",
  "evidence_quotes": ["quote1", "quote2", "quote3"],
  "evidence_chunk_ids": ["id1"],
  "patterns_found": ["pattern1"],
  "counter_evidence": [],
  "contradictions_internal": [],
  "structured_data": {
    "conformity_susceptibility": 0.0-1.0,
    "authority_deference": 0.0-1.0,
    "social_proof_reliance": 0.0-1.0,
    "independent_thinking": 0.0-1.0,
    "influence_style": "RATIONAL | EMOTIONAL | AUTHORITY | SOCIAL_PROOF | MANIPULATIVE | MIXED",
    "key_influencers": ["people/sources that disproportionately shape their views"],
    "fomo_signals": true or false,
    "nonconformity_instances": ["times they went against consensus"]
  }
}"""

    def parse_structured_data(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "conformity_susceptibility": raw.get("conformity_susceptibility", 0.5),
            "authority_deference": raw.get("authority_deference", 0.5),
            "social_proof_reliance": raw.get("social_proof_reliance", 0.5),
            "independent_thinking": raw.get("independent_thinking", 0.5),
            "influence_style": raw.get("influence_style", "MIXED"),
            "key_influencers": raw.get("key_influencers", []),
            "fomo_signals": raw.get("fomo_signals", False),
            "nonconformity_instances": raw.get("nonconformity_instances", []),
        }


# ══════════════════════════════════════════════════════════════════════
# IDENTITY LAYER
# ══════════════════════════════════════════════════════════════════════

class SelfConceptAgent(MicroAgent):
    """
    L1-25 | Self-Concept Agent
    ──────────────────────────────────────────────────────────────────
    OWNS: The person's self-image, self-labels, self-worth signals,
          and the thermostat that keeps behavior consistent with identity.

    WHY THIS MATTERS: Identity is the most powerful behavioral force.
    People do not act inconsistently with how they see themselves.
    The self-concept acts as a thermostat: any behavior that feels
    "not like me" creates discomfort that drives back toward the identity.
    Changing behavior without changing identity is temporary.

    WHAT IT LOOKS FOR:
    - Direct self-descriptions ("I'm someone who...", "I've always been...")
    - Self-labels (positive and negative)
    - Confidence signals across different domains
    - Consistency of self-description across contexts
    - "Not like me" rejections (reveals identity boundaries)
    - Self-worth signals — what makes them feel valuable
    """

    AGENT_ID = "L1-25-self-concept"
    AGENT_NAME = "Self-Concept Agent"
    DIMENSION = "Self-image, self-labels, confidence, and identity thermostat"
    RAG_QUERIES = [
        "I'm someone who I am I've always been I'm the type",
        "I'm not good at I can't do I don't do",
        "confident proud good at capable strong",
        "insecure doubt myself not sure I can",
        "that's not me not like me I don't do that",
        "I identify as I see myself I think of myself",
        "I'm proud of I know I'm good at definitely",
    ]
    TOP_K = 9

    @property
    def ANALYSIS_PROMPT(self) -> str:
        return """
ANALYSIS TASK: Self-Concept Mapping

Extract the person's self-model — how they construct and maintain their identity.

WHAT TO MAP:

1. EXPLICIT SELF-LABELS:
   Direct statements: "I'm introverted", "I'm a hard worker", "I'm not technical"
   Each label is an identity anchor that resists change.
   List all explicit self-labels — positive and negative.

2. CONFIDENCE DISTRIBUTION:
   Where is this person confident? Where are they insecure?
   High confidence domains: speak with authority, no hedging, expertise language
   Low confidence domains: excessive hedging, apology, minimization, avoidance

3. IDENTITY BOUNDARY MARKERS:
   "That's not really me", "I don't do X", "I'm not that kind of person"
   These mark the edges of the self-concept — what's excluded from identity.
   They're often the most important boundaries because they block growth.

4. SELF-WORTH SOURCE:
   What makes this person feel worthwhile?
   Achievement? Being liked? Being right? Being helpful? Looking good?
   Their worth source determines their emotional vulnerabilities.

5. CONSISTENCY vs FRAGMENTATION:
   Is the self-concept stable and coherent?
   OR highly context-dependent (different person in different contexts)?
   Fragmentation = identity hasn't been fully integrated.

6. IDEAL SELF vs ACTUAL SELF GAP:
   What do they aspire to be that they're not yet?
   Language: "I want to be", "I'm working on becoming", "I wish I were"
   Large gap = source of either motivation or shame
"""

    @property
    def OUTPUT_INSTRUCTIONS(self) -> str:
        return """
Return this exact JSON structure:
{
  "rating": 0.0-1.0,
  "confidence": 0.0-1.0,
  "label": "Brief self-concept characterization",
  "summary": "2-3 sentences",
  "evidence_quotes": ["quote1", "quote2", "quote3"],
  "evidence_chunk_ids": ["id1"],
  "patterns_found": ["pattern1"],
  "counter_evidence": [],
  "contradictions_internal": [],
  "structured_data": {
    "positive_self_labels": ["hardworking", "honest"],
    "negative_self_labels": ["not creative", "bad with people"],
    "confidence_map": {"work": 0.8, "social": 0.4, "creative": 0.3},
    "identity_boundaries": ["things they explicitly exclude from self"],
    "self_worth_source": "achievement | approval | competence | relationships | meaning | appearance",
    "self_concept_stability": "FRAGMENTED | UNSTABLE | MODERATE | STABLE | VERY_STABLE",
    "ideal_self_gap": "SMALL | MODERATE | LARGE",
    "self_compassion_level": 0.0-1.0
  }
}"""

    def parse_structured_data(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "positive_self_labels": raw.get("positive_self_labels", []),
            "negative_self_labels": raw.get("negative_self_labels", []),
            "confidence_map": raw.get("confidence_map", {}),
            "identity_boundaries": raw.get("identity_boundaries", []),
            "self_worth_source": raw.get("self_worth_source", "UNKNOWN"),
            "self_concept_stability": raw.get("self_concept_stability", "MODERATE"),
            "ideal_self_gap": raw.get("ideal_self_gap", "MODERATE"),
            "self_compassion_level": raw.get("self_compassion_level", 0.5),
        }


class NarrativeIdentityAgent(MicroAgent):
    """
    L1-26 | Narrative Identity Agent
    ──────────────────────────────────────────────────────────────────
    OWNS: The story the person tells about their life.
          Protagonist vs victim framing. Agency level. Growth arc.
          How they construct meaning from their experiences.

    WHY THIS MATTERS: Dan McAdams's research established that people
    construct identity through narrative. The story you tell about your
    life determines how you respond to future events. A person with a
    redemption narrative (bad things made me stronger) is resilient.
    A person with a contamination narrative (good things always go wrong)
    is vulnerable to depression and learned helplessness.

    WHAT IT LOOKS FOR:
    - How stories about their life are structured
    - Protagonist (agent, makes things happen) vs victim (things happen to them)
    - Redemption arcs (bad → good) vs contamination arcs (good → bad)
    - Meaning-making: do they extract lessons from experience?
    - Self-compassion vs self-criticism in life narratives
    - How they explain their own history to others
    """

    AGENT_ID = "L1-26-narrative"
    AGENT_NAME = "Narrative Identity Agent"
    DIMENSION = "Life story framing, protagonist vs victim role, growth arc, meaning-making"
    RAG_QUERIES = [
        "story of my life that's how it always goes",
        "because of what happened that led me to shaped me",
        "I made it happen I chose I decided I created",
        "it happened to me I had no choice was done to me",
        "learned from that experience taught me changed me",
        "I've been through a lot despite everything survived",
        "my life is a mess everything goes wrong bad luck",
        "I'm where I am because of what I did worked for",
    ]
    TOP_K = 9

    @property
    def ANALYSIS_PROMPT(self) -> str:
        return """
ANALYSIS TASK: Narrative Identity Analysis

Analyze the story this person tells about their life and themselves.

NARRATIVE DIMENSIONS:

1. AGENCY vs VICTIMHOOD:
   HIGH AGENCY: "I made this happen", "I chose to", "I decided"
   LOW AGENCY / VICTIM: "It happened to me", "I had no choice", "they did this to me"
   Both can be accurate — look for the systematic tendency

2. STORY ARC TYPE:
   GROWTH narrative: challenges lead to development, things are getting better
   REDEMPTION narrative: suffering transformed into something meaningful
   CONTAMINATION narrative: good things always get ruined or go bad
   DECLINE narrative: things have been getting worse, loss of something
   STUCK narrative: things don't change, same patterns repeat

3. MEANING-MAKING CAPACITY:
   After hard experiences, do they extract meaning, lessons, or growth?
   OR do they remain in raw pain without narrative integration?
   "That experience taught me..." = active meaning-making
   "That experience ruined everything" = no meaning-making

4. PROTAGONIST ROLE:
   Are they the main character driving their own story?
   OR are they a secondary character in someone else's story?
   (People with low agency often narrate their lives around other people's actions)

5. SELF-COMPASSION IN NARRATIVE:
   How do they talk about their past self?
   With compassion (understanding why they did what they did)?
   With harsh judgment (stupid, weak, should have known better)?

6. NARRATIVE COHERENCE:
   Does their life story hang together as a coherent arc?
   OR is it fragmented, contradictory, without clear through-lines?
"""

    @property
    def OUTPUT_INSTRUCTIONS(self) -> str:
        return """
Return this exact JSON structure:
{
  "rating": 0.0-1.0,
  "confidence": 0.0-1.0,
  "label": "Primary narrative identity characterization",
  "summary": "2-3 sentences",
  "evidence_quotes": ["quote1", "quote2", "quote3"],
  "evidence_chunk_ids": ["id1"],
  "patterns_found": ["pattern1"],
  "counter_evidence": [],
  "contradictions_internal": [],
  "structured_data": {
    "primary_role": "PROTAGONIST | VICTIM | HERO | OBSERVER | MIXED",
    "agency_level": 0.0-1.0,
    "story_arc": "growth | redemption | contamination | decline | stuck | mixed",
    "meaning_making_capacity": "NONE | LOW | MODERATE | HIGH",
    "self_compassion_in_narrative": 0.0-1.0,
    "narrative_coherence": "FRAGMENTED | LOW | MODERATE | HIGH",
    "blame_attribution": "SELF | OTHERS | SITUATION | MIXED",
    "key_narrative_themes": ["theme1", "theme2"],
    "narrative_evidence": ["quote that reveals life story framing"]
  }
}"""

    def parse_structured_data(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "primary_role": raw.get("primary_role", "MIXED"),
            "agency_level": raw.get("agency_level", 0.5),
            "story_arc": raw.get("story_arc", "mixed"),
            "meaning_making_capacity": raw.get("meaning_making_capacity", "MODERATE"),
            "self_compassion_in_narrative": raw.get("self_compassion_in_narrative", 0.5),
            "narrative_coherence": raw.get("narrative_coherence", "MODERATE"),
            "blame_attribution": raw.get("blame_attribution", "MIXED"),
            "key_narrative_themes": raw.get("key_narrative_themes", []),
            "narrative_evidence": raw.get("narrative_evidence", []),
        }


# ══════════════════════════════════════════════════════════════════════
# AGENT REGISTRY
# ══════════════════════════════════════════════════════════════════════

def build_all_agents(vector_store, client=None) -> dict:
    """
    Instantiate all 26 micro agents and return as a dict keyed by agent_id.
    Pass in your shared vector_store and Anthropic client.
    """
    agents = [
        # Biology
        StressSignalAgent(vector_store, client),
        EnergyVitalityAgent(vector_store, client),
        # Perception
        AttentionFocusAgent(vector_store, client),
        InterpretationBiasAgent(vector_store, client),
        ConfirmationBiasAgent(vector_store, client),
        # Dual-Process
        AutomaticReactionAgent(vector_store, client),
        CognitiveBiasMapAgent(vector_store, client),
        # Emotion
        EmotionalRangeAgent(vector_store, client),
        EmotionalRegulationAgent(vector_store, client),
        EmotionalThemeAgent(vector_store, client),
        # Cognition
        ThinkingStyleAgent(vector_store, client),
        LanguageComplexityAgent(vector_store, client),
        MemoryPatternAgent(vector_store, client),
        # Beliefs
        CoreBeliefAgent(vector_store, client),
        ValuesHierarchyAgent(vector_store, client),
        WorldviewAgent(vector_store, client),
        # Motivation
        NeedsSignalAgent(vector_store, client),
        DriveTypeAgent(vector_store, client),
        PersistenceAgent(vector_store, client),
        # Behavior
        HabitLoopAgent(vector_store, client),
        DecisionStyleAgent(vector_store, client),
        # Social
        AttachmentStyleAgent(vector_store, client),
        CommunicationStyleAgent(vector_store, client),
        SocialInfluenceAgent(vector_store, client),
        # Identity
        SelfConceptAgent(vector_store, client),
        NarrativeIdentityAgent(vector_store, client),
    ]

    registry = {agent.AGENT_ID: agent for agent in agents}
    print(f"[PSYCHE] Initialized {len(registry)} micro agents.")
    return registry


# ── DOMAIN GROUPS (for Layer 2 routing) ──────────────────────────────
DOMAIN_GROUPS = {
    "biology": ["L1-01-stress", "L1-02-energy"],
    "perception_cognition": [
        "L1-03-attention", "L1-04-interpretation",
        "L1-05-confirmation", "L1-06-system1", "L1-07-biases"
    ],
    "emotional": ["L1-08-emotional-range", "L1-09-regulation", "L1-10-emotional-themes"],
    "cognitive_beliefs": [
        "L1-11-thinking", "L1-12-language", "L1-13-memory",
        "L1-14-core-beliefs", "L1-15-values", "L1-16-worldview"
    ],
    "motivation_behavior": [
        "L1-17-needs", "L1-18-drive", "L1-19-persistence",
        "L1-20-habits", "L1-21-decisions"
    ],
    "social_identity": [
        "L1-22-attachment", "L1-23-communication",
        "L1-24-social-influence", "L1-25-self-concept", "L1-26-narrative"
    ]
}

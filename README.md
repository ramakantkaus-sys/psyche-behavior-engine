# 🧠 PSYCHE OS — Psychological Profiling Multi-Agent System

> *A 26-agent AI system that reads someone's chat messages and produces a comprehensive, evidence-based psychological profile.*

---

## Table of Contents

1. [What Is PSYCHE OS?](#what-is-psyche-os)
2. [Why Does This Exist?](#why-does-this-exist)
3. [System Architecture — The Big Picture](#system-architecture--the-big-picture)
4. [The 9-Step Pipeline](#the-9-step-pipeline)
5. [Deep Dive: Each Component](#deep-dive-each-component)
6. [The 26 Micro-Agents — Complete Index](#the-26-micro-agents--complete-index)
7. [The Psychological Theory Behind It](#the-psychological-theory-behind-it)
8. [How The Debate System Works](#how-the-debate-system-works)
9. [Output Format: The Psychological Profile](#output-format-the-psychological-profile)
10. [Project File Structure](#project-file-structure)
11. [Tech Stack](#tech-stack)
12. [How to Run](#how-to-run)
13. [Future Roadmap](#future-roadmap)

---

## What Is PSYCHE OS?

**PSYCHE OS** (Psychological Cognition & Heuristic Evaluation Operating System) is a **multi-agentic AI system** that analyzes a person's text-based conversations — WhatsApp chats, ChatGPT conversations, or any written dialogue — and produces a **comprehensive psychological profile** of that person.

It is NOT a chatbot. It is a **diagnostic engine**. You feed it someone's chat history, and 26 specialized AI agents work in parallel to analyze every psychological dimension of that person's behaviour — from stress levels to attachment style, from cognitive biases to core beliefs, from emotional regulation to narrative identity.

### The Key Insight

People reveal their psychology in how they write, not just in what they write. Word choice, sentence structure, emotional patterns, topics they return to, how they handle conflict, what they catastrophize about, what they dismiss — all of these are **signals** that trained analysts can read. PSYCHE OS automates this at scale.

---

## Why Does This Exist?

Traditional psychological assessment requires:
- Hours of in-person interviews
- Trained professionals interpreting responses
- Expensive standardized tests
- Willing participation from the subject

PSYCHE OS offers a complementary approach:
- Analyzes **naturalistic text** (real conversations, not test responses)
- Works on **existing data** (no special session needed)
- Produces **evidence-based output** (every claim linked to direct quotes)
- Provides **structured, quantitative ratings** alongside qualitative analysis
- Can identify patterns across **thousands of messages** that human analysts would miss

---

## System Architecture — The Big Picture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PSYCHE OS ARCHITECTURE                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  INPUT:  Chat file (.txt WhatsApp / .json ChatGPT / .txt generic)  │
│                              │                                      │
│                              ▼                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                     PIPELINE LAYER                           │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────────┐  │   │
│  │  │  Parser   │→│ Profiler  │→│ Chunker  │→│Vector Store│  │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └────────────┘  │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    LAYER 1: MICRO-AGENTS                     │   │
│  │                                                              │   │
│  │  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐    │   │
│  │  │Stress│ │Energy│ │Attn  │ │Bias  │ │Conf  │ │Sys1  │    │   │
│  │  │Signal│ │Vital │ │Focus │ │Interp│ │Bias  │ │React │    │   │
│  │  └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘    │   │
│  │     │        │        │        │        │        │         │   │
│  │  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐    │   │
│  │  │CogBi │ │EmoRng│ │EmoReg│ │EmoThm│ │Think │ │LangCx│    │   │
│  │  │MapAgt│ │Agent │ │Agent │ │Agent │ │Style │ │Agent │    │   │
│  │  └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘    │   │
│  │     │        │        │        │        │        │         │   │
│  │  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐    │   │
│  │  │Memory│ │CoreBe│ │Values│ │World │ │Needs │ │Drive │    │   │
│  │  │Pattrn│ │liefs │ │Hier  │ │view  │ │Signal│ │Type  │    │   │
│  │  └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘    │   │
│  │     │        │        │        │        │        │         │   │
│  │  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐    │   │
│  │  │Persis│ │Habit │ │Decis │ │Attach│ │Comm  │ │Social│    │   │
│  │  │tence │ │Loop  │ │Style │ │Style │ │Style │ │Inflnc│    │   │
│  │  └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘    │   │
│  │     │        │        │        │        │        │         │   │
│  │  ┌──────┐ ┌──────┐                                         │   │
│  │  │SelfCn│ │Narrat│      ALL 26 AGENTS RUN IN PARALLEL      │   │
│  │  │cept  │ │ident │                                         │   │
│  │  └──┬───┘ └──┬───┘                                         │   │
│  └─────┼────────┼──────────────────────────────────────────────┘   │
│        │        │                                                   │
│        ▼        ▼                                                   │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │               CONFLICT DETECTION + DEBATE NODE               │   │
│  │  When agents disagree, an LLM mediates the conflict          │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                  LAYER 2: DOMAIN SYNTHESIS                   │   │
│  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐    │   │
│  │  │Biology │ │Percept │ │Emotion │ │Cogn &  │ │Motiv & │    │   │
│  │  │Domain  │ │& Cog   │ │Domain  │ │Beliefs │ │Behavior│    │   │
│  │  └────────┘ └────────┘ └────────┘ └────────┘ └────────┘    │   │
│  │  ┌────────┐                                                  │   │
│  │  │Social &│  6 DOMAIN SYNTHESIZERS merge L1 signals          │   │
│  │  │Identity│                                                  │   │
│  │  └────────┘                                                  │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                  LAYER 3: FINAL REPORT                       │   │
│  │  Merges all domain syntheses into a comprehensive            │   │
│  │  psychological profile with cross-domain insights            │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  OUTPUT:  JSON Report + Markdown Report + Behavioral Predictions   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## The 9-Step Pipeline

Here's exactly what happens when you run `python main.py chat.txt --speaker "John"`:

### Step 1: Parse Chat File
**File:** `pipeline/chat_parser.py`

The parser auto-detects the file format:
- **.txt** → Tries WhatsApp format first (matches timestamp patterns like `[DD/MM/YYYY, HH:MM]`), falls back to generic paragraph-based parsing
- **.json** → Parses ChatGPT export format (navigates the nested `mapping` structure)

Each message becomes a `Message` object with: `id`, `timestamp`, `speaker`, `text`, `source`, `word_count`, `char_count`.

### Step 2: Compute Statistical Profile
**File:** `pipeline/stat_profiler.py`

Scans all messages from the target speaker and computes:
- **Word ratios:** first-person pronoun usage, positive/negative word ratios
- **Behavioral signals:** question ratio, exclamation ratio
- **Vocabulary metrics:** unique word count, vocabulary richness
- **Temporal patterns:** message frequency by hour, time span, average response time
- **Topic extraction:** most frequently discussed content words

This produces a `StatProfile` object that every agent can reference.

### Step 3: Chunk Conversations
**File:** `pipeline/chunker.py`

Splits the raw message stream into overlapping **chunks** of ~30 messages each, with 5-message overlap between chunks. Each chunk gets metadata:
- **Topic:** auto-extracted from most frequent content words
- **Emotional tone:** positive / negative / neutral / mixed (keyword-based)
- **Conversation type:** casual / question / vent / debate / reflection / plan
- **Time period:** early / middle / recent in the conversation timeline
- **Target speaker ratio:** how much of the chunk is the target person talking

### Step 4: Index Into Vector Store
**File:** `pipeline/vector_store.py`

All chunks are embedded using **ChromaDB's built-in embedding model** (all-MiniLM-L6-v2) and indexed for semantic retrieval. This is what enables **RAG** — each agent can query for only the chunks most relevant to its specific dimension.

### Step 5: Layer 1 — 26 Micro-Agents
**Files:** `psyche_base_agent.py`, `psyche_layer1_agents.py`, `psyche_layer1_runner.py`

All 26 agents launch **in parallel** using a `ThreadPoolExecutor` with 26 threads. Each agent:

1. **Queries the vector store** with its specialized RAG queries (e.g., the Stress Agent queries for "exhausted burned out can't sleep overwhelmed")
2. **Retrieves the top-k most relevant chunks** (typically 7-10 chunks per agent)
3. **Builds a context** combining the retrieved chunks + statistical profile
4. **Constructs a specialized prompt** with its unique analysis instructions
5. **Calls Claude Haiku** (fast, cost-efficient LLM) with the prompt
6. **Parses the JSON response** into a standardized `MicroSignal` with rating, confidence, evidence quotes, and structured data

Each agent returns a `MicroSignal` containing:
- `rating` (0.0–1.0): severity/intensity score
- `confidence` (0.0–1.0): how confident the agent is in its assessment
- `label`: human-readable classification
- `summary`: 2-3 sentence description
- `evidence_quotes`: direct quotes supporting the finding
- `patterns_found`: identified behavioral patterns
- `counter_evidence`: evidence that contradicts the main finding
- `contradictions_internal`: internal inconsistencies found
- `structured_data`: dimension-specific structured output

### Step 6: Conflict Detection
**File:** `psyche_layer1_runner.py` (method: `detect_conflicts`)

After all agents return, the system checks for:
- **Intra-domain conflicts:** Two agents in the same domain producing ratings that differ by > 0.30 (e.g., high stress but high energy)
- **Cross-domain conflicts:** Logically inconsistent combinations across domains (e.g., "high stress" + "high vitality")

Each conflict is flagged with the two agents involved, the rating difference, and whether it needs debate resolution.

### Step 7: Debate Node
**File:** `agents/debate_node.py`

For each flagged conflict, an LLM mediator:
1. Examines both agents' evidence quality and quantity
2. Weighs confidence scores
3. Considers if both can be simultaneously true
4. Produces a resolution: `AGENT_A_WINS`, `AGENT_B_WINS`, `COMPROMISE`, or `BOTH_VALID`

### Step 8: Layer 2 — Domain Synthesis
**File:** `agents/layer2/domain_synthesis.py`

The 26 agents are grouped into **6 psychological domains**:

| Domain | Agents |
|---|---|
| Biology | Stress, Energy |
| Perception & Cognition | Attention, Interpretation Bias, Confirmation Bias, System 1, Cognitive Bias Map |
| Emotional | Emotional Range, Regulation, Themes |
| Cognition & Beliefs | Thinking Style, Language, Memory, Core Beliefs, Values, Worldview |
| Motivation & Behavior | Needs, Drive Type, Persistence, Habits, Decision Style |
| Social & Identity | Attachment, Communication, Social Influence, Self-Concept, Narrative |

For each domain, a **Domain Synthesizer** agent reads all the L1 signals, resolves remaining tensions, and produces a unified domain-level narrative with key findings and confidence ratings.

### Step 9: Layer 3 — Final Report
**File:** `agents/layer3/final_report.py`

The capstone synthesis. A more powerful LLM (**Claude Sonnet**) receives:
- All 6 domain syntheses
- Debate resolutions
- Statistical profile
- All individual L1 signals

It produces the **final psychological profile** with:
- Executive summary
- 7-section psychological profile (emotional, cognitive, beliefs, motivation, social, identity, biological)
- Cross-domain insights (how patterns in one area explain another)
- Risk areas with severity ratings
- Identified strengths
- Growth areas with recommended approaches
- Behavioral predictions
- Confidence assessment with caveats

---

## Deep Dive: Each Component

### The Base Agent Architecture
**File:** `psyche_base_agent.py` (263 lines)

Every micro-agent inherits from `MicroAgent`, an abstract base class that handles:

```
MicroAgent (Abstract Base)
    │
    ├── Properties each agent must define:
    │   ├── AGENT_ID       → "L1-01-stress"
    │   ├── AGENT_NAME     → "Stress Signal Agent"
    │   ├── DIMENSION      → "chronic stress, burnout, overwhelm"
    │   ├── RAG_QUERIES    → ["exhausted burned out...", ...]
    │   ├── TOP_K          → 8 (how many chunks to retrieve)
    │   ├── ANALYSIS_PROMPT → (the specialized prompt)
    │   └── OUTPUT_INSTRUCTIONS → (JSON structure to return)
    │
    ├── Methods inherited:
    │   ├── retrieve_chunks()  → queries vector store with RAG_QUERIES
    │   ├── build_context()    → formats chunks + stats into prompt
    │   ├── build_prompt()     → combines system prompt + context + instructions
    │   ├── call_llm()         → sends to Claude Haiku
    │   ├── parse_output()     → extracts JSON from LLM response
    │   └── run()              → orchestrates the full pipeline
    │
    └── Error handling:
        └── Returns low-confidence "AGENT_FAILED" signal instead of crashing
```

### The RAG System
Each agent doesn't see the entire conversation — it only sees **what's relevant to its dimension**. This is critical because:
- A Stress Agent doesn't need to see messages about career planning
- A Core Beliefs Agent doesn't need to see logistics discussions
- A 10,000-message chat would overflow any LLM context window

**How RAG works per agent:**
1. Agent defines 5-10 **semantic search queries** specific to its dimension
2. For each query, the vector store returns the most similar chunks
3. Results are deduplicated and ranked by relevance
4. Top-k chunks are passed to the LLM as context

### The Vector Store
**ChromaDB** provides:
- Automatic embedding using `all-MiniLM-L6-v2` (sentence-transformers model)
- Cosine similarity search
- Persistent storage option for large datasets
- Fast retrieval even with thousands of chunks

---

## The 26 Micro-Agents — Complete Index

### 🫀 Biology Layer
| ID | Agent | Dimension | What It Detects |
|---|---|---|---|
| L1-01 | Stress Signal | Chronic stress, burnout | Sleep disruption, overwhelm language, somatic symptoms |
| L1-02 | Energy Vitality | Baseline energy | Fatigue patterns, vitality fluctuations, energy sustainability |

### 👁️ Perception Layer
| ID | Agent | Dimension | What It Detects |
|---|---|---|---|
| L1-03 | Attention Focus | Attention patterns | Internal vs external focus, rumination, detail orientation |
| L1-04 | Interpretation Bias | Meaning-making lens | Negative vs positive interpretation of ambiguous events |
| L1-05 | Confirmation Bias | Evidence filtering | Selective attention to confirming vs disconfirming evidence |

### ⚡ Dual-Process Layer
| ID | Agent | Dimension | What It Detects |
|---|---|---|---|
| L1-06 | Automatic Reaction | System 1 patterns | Snap judgments, emotional hijacks, impulsive reactions |
| L1-07 | Cognitive Bias Map | 30+ cognitive biases | Anchoring, availability heuristic, sunk cost, dunning-kruger, etc. |

### 💜 Emotion Layer
| ID | Agent | Dimension | What It Detects |
|---|---|---|---|
| L1-08 | Emotional Range | Emotional vocabulary | Range and diversity of expressed emotions |
| L1-09 | Emotional Regulation | Self-regulation | Suppression, reappraisal, flooding, avoidance strategies |
| L1-10 | Emotional Theme | Recurring themes | Persistent emotional undercurrents (shame, guilt, anger, joy) |

### 🧠 Cognition Layer
| ID | Agent | Dimension | What It Detects |
|---|---|---|---|
| L1-11 | Thinking Style | Cognitive approach | Analytical vs intuitive, abstract vs concrete, creative vs rigid |
| L1-12 | Language Complexity | Linguistic sophistication | Vocabulary richness, sentence complexity, hedging patterns |
| L1-13 | Memory Pattern | Memory selectivity | What's remembered, distorted, or emphasized in recollection |

### 🔮 Beliefs Layer
| ID | Agent | Dimension | What It Detects |
|---|---|---|---|
| L1-14 | Core Beliefs | Deepest assumptions | "I am worthless / lovable / competent" — implicit rules |
| L1-15 | Values Hierarchy | Actual priorities | Stated vs behavioral values, sacrifice patterns, guilt triggers |
| L1-16 | Worldview | Fundamental world model | Scarcity vs abundance, trust vs threat, fair vs unfair world |

### 🔥 Motivation Layer
| ID | Agent | Dimension | What It Detects |
|---|---|---|---|
| L1-17 | Needs Signal | Unmet needs | Maslow + Self-Determination Theory — what's craved but missing |
| L1-18 | Drive Type | Motivation source | Intrinsic vs extrinsic, fear vs growth, shame-based drive |
| L1-19 | Persistence | Grit & resilience | Setback response, abandonment patterns, recovery speed |

### 🎯 Behavior Layer
| ID | Agent | Dimension | What It Detects |
|---|---|---|---|
| L1-20 | Habit Loop | Automated behaviors | Cue → Routine → Reward loops, emotional triggers |
| L1-21 | Decision Style | Choice-making patterns | Impulsive vs deliberate, risk orientation, analysis paralysis |

### 🤝 Social Layer
| ID | Agent | Dimension | What It Detects |
|---|---|---|---|
| L1-22 | Attachment Style | Relational wiring | Secure / anxious / avoidant / disorganized attachment |
| L1-23 | Communication | Expression patterns | Assertive / passive / aggressive / collaborative style |
| L1-24 | Social Influence | Influence patterns | Conformity, authority deference, social proof reliance |

### 🪞 Identity Layer
| ID | Agent | Dimension | What It Detects |
|---|---|---|---|
| L1-25 | Self-Concept | Self-image | Self-labels, confidence map, identity boundaries, self-worth |
| L1-26 | Narrative Identity | Life story framing | Protagonist vs victim, redemption vs contamination arc |

---

## The Psychological Theory Behind It

PSYCHE OS is grounded in established psychological frameworks:

| Framework | Used For | Agents |
|---|---|---|
| **CBT** (Cognitive Behavioral Therapy) | Core beliefs, cognitive distortions | Core Beliefs, Cognitive Bias Map |
| **ACT** (Acceptance & Commitment Therapy) | Values identification, defusion | Values Hierarchy |
| **Attachment Theory** (Bowlby/Ainsworth) | Relationship patterns | Attachment Style |
| **Self-Determination Theory** (Deci & Ryan) | Intrinsic motivation, needs | Needs Signal, Drive Type |
| **Maslow's Hierarchy** | Need identification | Needs Signal |
| **Dual Process Theory** (Kahneman) | System 1 vs System 2 thinking | Automatic Reaction, Thinking Style |
| **Dan McAdams' Narrative Identity** | Life story construction | Narrative Identity |
| **Habit Loop Model** (Duhigg) | Behavioral automation | Habit Loop |
| **Schema Therapy** (Jeffrey Young) | Core schemas, early maladaptive schemas | Core Beliefs, Worldview |
| **Emotion Regulation Theory** (Gross) | Regulation strategies | Emotional Regulation |
| **Grit Research** (Angela Duckworth) | Persistence and resilience | Persistence |

---

## How The Debate System Works

When two agents disagree significantly (rating difference > 0.30 within the same domain), the **Debate Node** activates:

```
     Agent A                Agent B
   Rating: 0.85           Rating: 0.40
   Evidence: 5 quotes     Evidence: 2 quotes
   Confidence: 0.90       Confidence: 0.60
        │                      │
        └──────────┬───────────┘
                   │
           ┌───────▼───────┐
           │  DEBATE NODE   │
           │  (LLM Mediator)│
           └───────┬───────┘
                   │
              ┌────▼────┐
              │RESOLUTION│
              └─────────┘

    Possible outcomes:
    ├── AGENT_A_WINS  (stronger evidence)
    ├── AGENT_B_WINS  (stronger evidence)
    ├── COMPROMISE    (split the difference)
    └── BOTH_VALID    (measuring different things)
```

**Example from the test run:**
- Stress Agent (0.85) vs Energy Agent (0.32) → **BOTH_VALID** (high stress AND low energy can coexist — they're measuring different things)

---

## Output Format: The Psychological Profile

The final output includes two files:

### JSON Report (machine-readable)
```json
{
  "executive_summary": "...",
  "psychological_profile": {
    "emotional_foundation": "...",
    "cognitive_architecture": "...",
    "belief_system": "...",
    "motivational_engine": "...",
    "social_wiring": "...",
    "identity_narrative": "...",
    "biological_baseline": "..."
  },
  "cross_domain_insights": ["..."],
  "risk_areas": [{"area": "...", "severity": "HIGH", ...}],
  "strengths": [{"strength": "...", ...}],
  "growth_areas": [{"area": "...", "recommended_approach": "...", ...}],
  "behavioral_predictions": ["..."],
  "confidence_assessment": {"overall_confidence": 0.84, ...}
}
```

### Markdown Report (human-readable)
A formatted document with sections for executive summary, each psychological dimension, risk areas, strengths, growth recommendations, and behavioral predictions.

---

## Project File Structure

```
Human Behaviour signal system/
│
├── 📋 Core Files (ORIGINAL — untouched)
│   ├── psyche_schemas.py          (178 lines)  — All Pydantic data models
│   ├── psyche_base_agent.py       (263 lines)  — Abstract MicroAgent base class
│   ├── psyche_layer1_agents.py    (3094 lines) — All 26 specialized agents
│   └── psyche_layer1_runner.py    (269 lines)  — Parallel execution engine
│
├── 🚀 Entry Point
│   └── main.py                    (220 lines)  — CLI + 9-step pipeline orchestrator
│
├── 📦 Package Structure (re-export modules)
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── models.py              — Re-exports from psyche_schemas
│   └── agents/
│       ├── __init__.py
│       ├── base_agent.py          — Re-exports from psyche_base_agent
│       └── layer1/
│           ├── __init__.py
│           └── all_agents.py      — Re-exports from psyche_layer1_agents
│
├── 🔧 Pipeline Components (NEW)
│   └── pipeline/
│       ├── __init__.py
│       ├── chat_parser.py         (200 lines)  — WhatsApp / ChatGPT / generic parser
│       ├── chunker.py             (200 lines)  — Conversation chunker + metadata
│       ├── stat_profiler.py       (170 lines)  — Statistical profile generator
│       └── vector_store.py         (90 lines)  — ChromaDB wrapper
│
├── 🧠 Higher-Layer Agents (NEW)
│   └── agents/
│       ├── debate_node.py         (160 lines)  — Conflict resolution engine
│       ├── layer2/
│       │   ├── __init__.py
│       │   └── domain_synthesis.py (160 lines) — 6 domain synthesizers
│       └── layer3/
│           ├── __init__.py
│           └── final_report.py    (250 lines)  — Final report generator
│
├── 📊 Generated Reports
│   └── reports/
│       ├── psyche_report_*.json   — Full structured report
│       └── psyche_report_*.md     — Human-readable markdown report
│
├── ⚙️ Configuration
│   ├── .env                       — API keys
│   ├── .env.example               — API key template
│   └── requirements.txt           — Python dependencies
│
└── 🐍 Environment
    └── venv/                      — Python 3.10 virtual environment
```

---

## Tech Stack

| Component | Technology | Why |
|---|---|---|
| **Language** | Python 3.10 | Async support, type hints, rich ecosystem |
| **LLM (L1/L2)** | Claude Haiku | Fast, cheap, good at structured JSON output |
| **LLM (L3)** | Claude Sonnet | Higher quality for final synthesis |
| **Vector Store** | ChromaDB | Simple, embedded, free, good default embeddings |
| **Embeddings** | all-MiniLM-L6-v2 | Lightweight, fast, runs locally |
| **Data Models** | Pydantic | Validation, serialization, type safety |
| **Parallelism** | ThreadPoolExecutor | 26 agents run simultaneously |
| **Config** | python-dotenv | API key management |

**Estimated Cost Per Analysis:**
- ~26 Claude Haiku calls (L1) ≈ $0.02
- ~6 Claude Haiku calls (L2 + debate) ≈ $0.005
- ~1 Claude Sonnet call (L3) ≈ $0.01
- **Total: ~$0.035 per full analysis** (extremely cost-efficient)

---

## How to Run

### Prerequisites
1. Python 3.10+
2. Anthropic API key

### Setup
```bash
# Activate virtual environment
.\venv\Scripts\activate    # Windows
source venv/bin/activate    # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Add your API key
echo ANTHROPIC_API_KEY=sk-ant-your-key-here > .env
```

### Run Analysis
```bash
# On a WhatsApp chat export
python main.py whatsapp_chat.txt --speaker "John Doe"

# On a ChatGPT export
python main.py conversations.json --speaker "user"

# With custom options
python main.py chat.txt --speaker "Alice" --output results/ --chunk-size 40 --timeout 300
```

### Flags
| Flag | Default | Description |
|---|---|---|
| `--speaker` | (required) | Name of the person to analyze |
| `--output` | `reports/` | Output directory |
| `--chunk-size` | 30 | Messages per chunk |
| `--workers` | 26 | Max parallel agents |
| `--timeout` | 90 | Per-agent timeout (seconds) |

---

## Future Roadmap

- [ ] **Layer 2 Debate Enhancement** — Multi-round debate between conflicting L2 domains
- [ ] **Temporal Analysis** — Track how psychological patterns change over time
- [ ] **Comparison Mode** — Compare two people's profiles (e.g., in a relationship)
- [ ] **Instagram/Twitter Parsing** — Expand data sources beyond chat
- [ ] **Web Dashboard** — Interactive UI to explore results visually
- [ ] **Custom Agent Creation** — Let users define their own analysis dimensions
- [ ] **Multi-language Support** — Analyze conversations in Hindi, Spanish, etc.
- [ ] **Streaming Output** — Show agents completing in real-time in the terminal
- [ ] **Report Diff** — Show how a profile changes with new data
- [ ] **API Server** — REST API to run analyses programmatically

---

*Built with ❤️ using Python, Claude (Anthropic), ChromaDB, and 26 specialized psychological micro-agents.*

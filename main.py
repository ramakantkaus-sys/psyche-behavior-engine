"""
PSYCHE OS — Main Entry Point
End-to-end pipeline: Chat file → Psychological Profile Report

Usage:
    python main.py <chat_file> --speaker "Person Name"
    python main.py chat.txt --speaker "John" --output reports/
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from anthropic import Anthropic

from psyche_schemas import StatProfile
from pipeline.chat_parser import ingest_chat
from pipeline.chunker import chunk_messages
from pipeline.stat_profiler import compute_stat_profile
from pipeline.vector_store import PsycheVectorStore
from psyche_layer1_runner import Layer1Runner
from agents.layer2.domain_synthesis import Layer2Runner
from agents.debate_node import DebateNode
from agents.layer3.final_report import FinalReportGenerator


def print_banner():
    """Print the PSYCHE OS startup banner."""
    print("""
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║   ██████╗ ███████╗██╗   ██╗ ██████╗██╗  ██╗███████╗     ║
║   ██╔══██╗██╔════╝╚██╗ ██╔╝██╔════╝██║  ██║██╔════╝     ║
║   ██████╔╝███████╗ ╚████╔╝ ██║     ███████║█████╗       ║
║   ██╔═══╝ ╚════██║  ╚██╔╝  ██║     ██╔══██║██╔══╝       ║
║   ██║     ███████║   ██║   ╚██████╗██║  ██║███████╗     ║
║   ╚═╝     ╚══════╝   ╚═╝    ╚═════╝╚═╝  ╚═╝╚══════╝     ║
║                                                          ║
║        Psychological Profiling Multi-Agent System        ║
║                   v1.0 — Layer 1-3                       ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
    """)


def run_pipeline(
    chat_file: str,
    target_speaker: str,
    output_dir: str = "reports",
    chunk_size: int = 30,
    max_workers: int = 26,
    timeout: int = 90,
):
    """
    Execute the full PSYCHE OS pipeline.

    Pipeline:
        1. Parse chat file → List[Message]
        2. Compute statistical profile → StatProfile
        3. Chunk messages → List[Chunk]
        4. Index chunks into vector store
        5. Run Layer 1: 26 micro agents in parallel
        6. Detect conflicts between agents
        7. Run Debate Node on conflicts
        8. Run Layer 2: Domain synthesis
        9. Run Layer 3: Final report
       10. Save and display report
    """
    total_start = time.time()

    # ── STEP 1: Parse ──────────────────────────────────────────────
    print("\n━━━ STEP 1/9: Parsing chat file ━━━")
    messages = ingest_chat(chat_file, speaker_name=target_speaker)
    if not messages:
        print("[ERROR] No messages parsed. Check your chat file format.")
        return

    # Identify unique speakers
    speakers = set(m.speaker for m in messages)
    print(f"  Speakers found: {speakers}")
    print(f"  Total messages: {len(messages)}")

    if target_speaker not in speakers:
        # Try case-insensitive match
        match = [s for s in speakers if s.lower() == target_speaker.lower()]
        if match:
            target_speaker = match[0]
        else:
            print(f"  [WARNING] Speaker '{target_speaker}' not found in: {speakers}")
            print(f"  Using first speaker: {list(speakers)[0]}")
            target_speaker = list(speakers)[0]

    # ── STEP 2: Statistical Profile ────────────────────────────────
    print("\n━━━ STEP 2/9: Computing statistical profile ━━━")
    stat_profile = compute_stat_profile(messages, target_speaker)
    print(f"  Total words: {stat_profile.total_words}")
    print(f"  Vocab size: {stat_profile.vocabulary_size}")
    print(f"  Time span: {stat_profile.time_span_days} days")
    print(f"  Top topics: {', '.join(stat_profile.most_common_topics[:5])}")

    # ── STEP 3: Chunk ──────────────────────────────────────────────
    print("\n━━━ STEP 3/9: Chunking conversations ━━━")
    chunks = chunk_messages(messages, target_speaker, chunk_size=chunk_size)
    if not chunks:
        print("[ERROR] No chunks created. Not enough messages.")
        return

    # ── STEP 4: Vector Store ───────────────────────────────────────
    print("\n━━━ STEP 4/9: Indexing into vector store ━━━")
    vector_store = PsycheVectorStore()
    vector_store.add_chunks(chunks)
    print(f"  Indexed {vector_store.count()} chunks")

    # ── STEP 5: Layer 1 — Run all 26 agents ────────────────────────
    print("\n━━━ STEP 5/9: Running Layer 1 (26 micro agents) ━━━")
    client = Anthropic()
    l1_runner = Layer1Runner(
        vector_store=vector_store,
        client=client,
        max_workers=max_workers,
        timeout_seconds=timeout,
    )
    signals = l1_runner.run_all(chunks, stat_profile, target_speaker)

    # ── STEP 6: Conflict detection ─────────────────────────────────
    print("\n━━━ STEP 6/9: Detecting conflicts ━━━")
    conflicts = l1_runner.detect_conflicts(signals)
    grouped = l1_runner.group_by_domain(signals)

    # ── STEP 7: Debate Node ────────────────────────────────────────
    print("\n━━━ STEP 7/9: Running debate node ━━━")
    debate_node = DebateNode(client)
    resolutions = debate_node.resolve_all_conflicts(conflicts, signals, target_speaker)

    # ── STEP 8: Layer 2 — Domain Synthesis ─────────────────────────
    print("\n━━━ STEP 8/9: Running Layer 2 (domain synthesis) ━━━")
    l2_runner = Layer2Runner(client)
    domain_syntheses = l2_runner.run(grouped, conflicts, target_speaker)

    # ── STEP 9: Layer 3 — Final Report ─────────────────────────────
    print("\n━━━ STEP 9/9: Generating final report ━━━")
    report_gen = FinalReportGenerator(client)
    report = report_gen.generate(
        domain_syntheses=domain_syntheses,
        debate_resolutions=resolutions,
        stat_profile=stat_profile,
        all_signals=signals,
        target_speaker=target_speaker,
    )

    # Add metadata
    report["metadata"] = {
        "total_messages_parsed": len(messages),
        "total_chunks": len(chunks),
        "agents_run": len(signals),
        "agents_failed": sum(1 for s in signals.values() if s.label == "AGENT_FAILED"),
        "conflicts_detected": len(conflicts),
        "conflicts_debated": len(resolutions),
        "total_time_seconds": round(time.time() - total_start, 1),
    }

    # Save JSON report
    json_path = report_gen.save_report(report, output_dir)

    # Save markdown report
    markdown = report_gen.render_markdown(report)
    md_path = json_path.replace(".json", ".md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(markdown)
    print(f"[REPORT] Markdown saved to {md_path}")

    # Print summary
    elapsed = time.time() - total_start
    print(f"""
╔══════════════════════════════════════════════════╗
║           PSYCHE OS — Analysis Complete          ║
╠══════════════════════════════════════════════════╣
║  Target: {target_speaker:<39} ║
║  Messages analyzed: {len(messages):<28} ║
║  Chunks created: {len(chunks):<31} ║
║  Agents run: {len(signals):<35} ║
║  Conflicts resolved: {len(resolutions):<27} ║
║  Total time: {elapsed:.1f}s{' ' * (33 - len(f'{elapsed:.1f}s'))} ║
║                                                  ║
║  JSON Report: {json_path:<34} ║
║  MD Report:   {md_path:<34} ║
╚══════════════════════════════════════════════════╝
    """)

    return report


def main():
    parser = argparse.ArgumentParser(
        description="PSYCHE OS — Psychological Profiling Multi-Agent System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py whatsapp_chat.txt --speaker "John Doe"
  python main.py chatgpt_export.json --speaker "user" --output results/
  python main.py conversation.txt --speaker "Alice" --chunk-size 40
        """
    )
    parser.add_argument("chat_file", help="Path to chat file (.txt for WhatsApp, .json for ChatGPT)")
    parser.add_argument("--speaker", required=True, help="Name of the person to analyze")
    parser.add_argument("--output", default="reports", help="Output directory for reports (default: reports/)")
    parser.add_argument("--chunk-size", type=int, default=30, help="Messages per chunk (default: 30)")
    parser.add_argument("--workers", type=int, default=26, help="Max parallel agent workers (default: 26)")
    parser.add_argument("--timeout", type=int, default=90, help="Per-agent timeout in seconds (default: 90)")

    args = parser.parse_args()

    if not os.path.exists(args.chat_file):
        print(f"[ERROR] File not found: {args.chat_file}")
        sys.exit(1)

    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("[ERROR] ANTHROPIC_API_KEY not set. Add it to your .env file or environment.")
        print("  Create a .env file with: ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)

    print_banner()
    run_pipeline(
        chat_file=args.chat_file,
        target_speaker=args.speaker,
        output_dir=args.output,
        chunk_size=args.chunk_size,
        max_workers=args.workers,
        timeout=args.timeout,
    )


if __name__ == "__main__":
    main()

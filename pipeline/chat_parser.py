"""
PSYCHE OS — Chat Ingestion Pipeline
Parses WhatsApp exports, ChatGPT JSON exports, and generic text
into a list of Message objects.
"""

from __future__ import annotations
import json
import os
import re
import uuid
from datetime import datetime
from typing import List, Optional

from psyche_schemas import Message


# ─────────────────────────────────────────────
# WHATSAPP PARSER
# ─────────────────────────────────────────────

# Common WhatsApp export formats:
# [DD/MM/YYYY, HH:MM:SS] Speaker: text
# DD/MM/YYYY, HH:MM - Speaker: text
# MM/DD/YY, HH:MM AM/PM - Speaker: text

WHATSAPP_PATTERNS = [
    # Format: [DD/MM/YYYY, HH:MM:SS] Speaker: message
    re.compile(
        r"\[(\d{1,2}/\d{1,2}/\d{2,4}),\s*(\d{1,2}:\d{2}(?::\d{2})?)\]\s*(.+?):\s*(.*)"
    ),
    # Format: DD/MM/YYYY, HH:MM - Speaker: message
    re.compile(
        r"(\d{1,2}/\d{1,2}/\d{2,4}),\s*(\d{1,2}:\d{2}(?:\s*[APap][Mm])?)\s*-\s*(.+?):\s*(.*)"
    ),
    # Format: MM/DD/YY, HH:MM AM/PM - Speaker: message
    re.compile(
        r"(\d{1,2}/\d{1,2}/\d{2,4}),\s*(\d{1,2}:\d{2}:\d{2}\s*[APap][Mm])\s*-\s*(.+?):\s*(.*)"
    ),
]

DATE_FORMATS = [
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M",
    "%m/%d/%y %I:%M %p",
    "%m/%d/%y %I:%M:%S %p",
    "%d/%m/%y %H:%M",
    "%d/%m/%y %H:%M:%S",
    "%m/%d/%Y %I:%M %p",
]


def _parse_timestamp(date_str: str, time_str: str) -> Optional[datetime]:
    """Try multiple date formats to parse the timestamp."""
    combined = f"{date_str} {time_str}".strip()
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(combined, fmt)
        except ValueError:
            continue
    return None


def parse_whatsapp(file_path: str) -> List[Message]:
    """
    Parse a WhatsApp chat export file (.txt) into Message objects.
    Handles multi-line messages by appending continuation lines.
    Skips system messages (encryption notices, group changes, etc.)
    """
    messages: List[Message] = []
    current_msg = None

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            matched = False

            for pattern in WHATSAPP_PATTERNS:
                match = pattern.match(line)
                if match:
                    # Save previous message
                    if current_msg:
                        messages.append(current_msg)

                    date_str, time_str, speaker, text = match.groups()
                    timestamp = _parse_timestamp(date_str, time_str)

                    # Skip system messages
                    system_keywords = [
                        "Messages and calls are end-to-end encrypted",
                        "created group",
                        "added",
                        "removed",
                        "left",
                        "changed the subject",
                        "changed this group",
                        "changed the group",
                        "<Media omitted>",
                        "message was deleted",
                        "You deleted this message",
                    ]
                    if any(kw.lower() in text.lower() for kw in system_keywords):
                        current_msg = None
                        matched = True
                        break

                    words = text.split()
                    current_msg = Message(
                        id=str(uuid.uuid4()),
                        timestamp=timestamp,
                        speaker=speaker.strip(),
                        text=text.strip(),
                        source="whatsapp",
                        word_count=len(words),
                        char_count=len(text),
                    )
                    matched = True
                    break

            if not matched and current_msg and line.strip():
                # Continuation line for multi-line message
                current_msg.text += "\n" + line.strip()
                words = current_msg.text.split()
                current_msg.word_count = len(words)
                current_msg.char_count = len(current_msg.text)

    # Don't forget the last message
    if current_msg:
        messages.append(current_msg)

    print(f"[PARSER] WhatsApp: parsed {len(messages)} messages from {os.path.basename(file_path)}")
    return messages


# ─────────────────────────────────────────────
# CHATGPT EXPORT PARSER
# ─────────────────────────────────────────────

def parse_chatgpt_export(file_path: str) -> List[Message]:
    """
    Parse a ChatGPT conversation export (JSON format).
    Expects the standard ChatGPT export structure:
    [{"title": "...", "mapping": {...}}]
    """
    messages: List[Message] = []

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    conversations = data if isinstance(data, list) else [data]

    for convo in conversations:
        mapping = convo.get("mapping", {})
        for node_id, node in mapping.items():
            msg_data = node.get("message")
            if not msg_data:
                continue

            role = msg_data.get("author", {}).get("role", "unknown")
            if role == "system":
                continue

            parts = msg_data.get("content", {}).get("parts", [])
            text = " ".join(str(p) for p in parts if isinstance(p, str)).strip()
            if not text:
                continue

            create_time = msg_data.get("create_time")
            timestamp = datetime.fromtimestamp(create_time) if create_time else None

            words = text.split()
            messages.append(Message(
                id=msg_data.get("id", str(uuid.uuid4())),
                timestamp=timestamp,
                speaker=role,
                text=text,
                source="chatgpt",
                word_count=len(words),
                char_count=len(text),
            ))

    # Sort by timestamp
    messages.sort(key=lambda m: m.timestamp or datetime.min)

    print(f"[PARSER] ChatGPT: parsed {len(messages)} messages from {os.path.basename(file_path)}")
    return messages


# ─────────────────────────────────────────────
# GENERIC TEXT PARSER
# ─────────────────────────────────────────────

def parse_generic_text(file_path: str, speaker_name: str = "User") -> List[Message]:
    """
    Parse a plain text file as messages.
    Each paragraph (separated by blank lines) becomes one message.
    All messages attributed to the given speaker_name.
    """
    messages: List[Message] = []

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    paragraphs = re.split(r"\n\s*\n", content)

    for para in paragraphs:
        text = para.strip()
        if not text or len(text) < 5:
            continue

        words = text.split()
        messages.append(Message(
            id=str(uuid.uuid4()),
            timestamp=None,
            speaker=speaker_name,
            text=text,
            source="generic",
            word_count=len(words),
            char_count=len(text),
        ))

    print(f"[PARSER] Generic: parsed {len(messages)} messages from {os.path.basename(file_path)}")
    return messages


# ─────────────────────────────────────────────
# AUTO-DETECT AND PARSE
# ─────────────────────────────────────────────

def ingest_chat(file_path: str, speaker_name: str = "User") -> List[Message]:
    """
    Auto-detect file type and parse appropriately.
    - .json → ChatGPT export
    - .txt  → try WhatsApp first, fallback to generic
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".json":
        return parse_chatgpt_export(file_path)
    elif ext == ".txt":
        # Try WhatsApp parsing first
        messages = parse_whatsapp(file_path)
        if len(messages) >= 5:
            return messages
        # Fallback to generic
        print("[PARSER] WhatsApp format not detected, falling back to generic parser.")
        return parse_generic_text(file_path, speaker_name)
    else:
        print(f"[PARSER] Unknown file type '{ext}', treating as generic text.")
        return parse_generic_text(file_path, speaker_name)

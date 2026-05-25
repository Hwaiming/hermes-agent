#!/usr/bin/env python3
"""
Near-Memory Cache — L1 fast-recall layer for session_search.

Analogy: human short-term memory before a full hippocampus search.
Only meaningful exchanges are cached; transient noise (config tweaks,
tool confirmations, stray keystrokes) is filtered out.

Cache file: ~/.hermes/sessions/recent_cache.json
Max entries: 50 (FIFO)
"""

import json
import logging
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Paths & limits
# ---------------------------------------------------------------------------
_HERMES_HOME = os.path.expanduser(os.environ.get("HERMES_HOME", "~/.hermes"))
_CACHE_PATH = os.path.join(_HERMES_HOME, "sessions", "recent_cache.json")
_MAX_ENTRIES = 50

# ---------------------------------------------------------------------------
# Meaningful-content filter
# ---------------------------------------------------------------------------

# Phrases that indicate the response is purely mechanical / valueless for recall.
_NOISE_PATTERNS: list = [
    # Pure acknowledgments
    r"^(好的|收到|OK|done|明白|了解|已?执行|已?完成|已?处理|搞定)[\s。.]*$",
    r"^(Got it|Alright|Understood|Done|Will do)[\s.]*$",
    # Config-only responses (model/skin/theme swaps)
    r"^(已切|已更换|已设置|已配置|已调整)",
    r"(模型|model|皮肤|skin|主题|theme).*(已切|已换|已更|OK)",
    # Single-word / emoji-only
    r"^[👍✅👌🆗]{1,3}$",
    r"^.{0,3}$",  # <= 3 chars
]

# Minimum character count for a response to be considered meaningful.
_MIN_MEANINGFUL_CHARS = 200


def is_meaningful(content: str) -> bool:
    """Return True if *content* looks worth caching for future recall."""
    if not content or not content.strip():
        return False
    stripped = content.strip()

    if len(stripped) < _MIN_MEANINGFUL_CHARS:
        return False

    for pattern in _NOISE_PATTERNS:
        if re.search(pattern, stripped, re.IGNORECASE):
            return False

    return True


# ---------------------------------------------------------------------------
# Topic & summary extraction (rule-based, no LLM – fast path)
# ---------------------------------------------------------------------------

def _extract_topic(content: str, max_chars: int = 30) -> str:
    """Extract a short topic from the response content."""
    # Use the first substantive sentence as the topic.
    # Strip markdown headers / bold markers.
    cleaned = re.sub(r"^#+\s*", "", content.strip())
    cleaned = re.sub(r"\*\*(.*?)\*\*", r"\1", cleaned)
    # Take first sentence (up to 。, ！, \n, or period)
    for delim in ("。", "！", "\n", ". ", "!"):
        if delim in cleaned:
            cleaned = cleaned.split(delim)[0]
            break
    # Truncate
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars]
    return cleaned.strip()


def _extract_summary(content: str, max_chars: int = 120) -> str:
    """Extract a short summary from the response content."""
    # Take the first 2-3 sentences, compress.
    cleaned = content.strip()
    # Remove markdown formatting noise
    cleaned = re.sub(r"^#+\s*", "", cleaned)
    cleaned = re.sub(r"\*\*(.*?)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"```.*?```", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)

    sentences = re.split(r"[。！\n]", cleaned)
    summary = ""
    for s in sentences:
        s = s.strip()
        if len(s) < 5:
            continue
        if len(summary) + len(s) > max_chars:
            break
        summary += s + "。"
    return summary.strip()


# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------

def load_cache() -> Dict[str, Any]:
    """Load cache from disk. Returns empty dict if not found or corrupt."""
    if not os.path.exists(_CACHE_PATH):
        return {"entries": [], "max_entries": _MAX_ENTRIES}
    try:
        with open(_CACHE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {"entries": [], "max_entries": _MAX_ENTRIES}
        data.setdefault("entries", [])
        data.setdefault("max_entries", _MAX_ENTRIES)
        return data
    except (json.JSONDecodeError, IOError) as exc:
        logging.warning("near_memory_cache: failed to load cache: %s", exc)
        return {"entries": [], "max_entries": _MAX_ENTRIES}


def save_cache(data: Dict[str, Any]) -> None:
    """Save cache to disk. Creates parent directory if needed."""
    os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
    with open(_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_recent(limit: int = 10) -> List[Dict[str, Any]]:
    """Return the *limit* most recent cache entries."""
    data = load_cache()
    return data["entries"][:limit]


def search_cache(query: str) -> List[Dict[str, Any]]:
    """Fuzzy-search cache entries (topic + summary) for *query*.

    Simple substring matching — fast enough for 50 entries.
    Returns entries sorted by relevance (number of keyword hits, descending).
    """
    if not query or not query.strip():
        return get_recent(10)

    data = load_cache()
    keywords = query.strip().lower().split()
    scored: List[tuple] = []

    for entry in data["entries"]:
        text = (entry.get("topic", "") + " " + entry.get("summary", "")).lower()
        score = sum(1 for kw in keywords if kw in text)
        if score > 0:
            scored.append((score, entry))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [entry for _, entry in scored]


# ---------------------------------------------------------------------------
# Cache population (called from reflection state)
# ---------------------------------------------------------------------------

def cache_session(
    session_id: str,
    messages: list,
    date: str = None,
    *,
    use_llm: bool = False,
) -> Dict[str, Any]:
    """Screen a session's messages and cache meaningful exchanges.

    Parameters
    ----------
    session_id : str
        The Hermes session ID.
    messages : list
        List of ``{role, content}`` dicts from the full conversation.
    date : str, optional
        ISO date string.  Defaults to today.
    use_llm : bool
        If True, call the LLM for better topic/summary.  Slow path for
        offline / low-value sessions.

    Returns
    -------
    dict with keys: ``cached``, ``skipped``, ``entries``
    """
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")

    assistant_messages = [
        m for m in messages
        if m.get("role") == "assistant" and is_meaningful(m.get("content", ""))
    ]

    data = load_cache()
    new_entries: List[Dict[str, Any]] = []
    cached = 0
    skipped = len(messages) - len(assistant_messages)

    now = datetime.now().isoformat(timespec="seconds")

    for msg in assistant_messages:
        content = msg["content"]
        if not is_meaningful(content):
            skipped += 1
            continue

        if use_llm:
            # LLM path: delegate to _summarize_via_llm
            topic, summary = _summarize_via_llm(content)
        else:
            topic = _extract_topic(content)
            summary = _extract_summary(content)

        if not topic:
            topic = "（无标题）"

        entry = {
            "session_id": session_id,
            "date": date,
            "topic": topic,
            "summary": summary,
            "cached_at": now,
        }
        new_entries.append(entry)
        cached += 1

    # Prepend new entries to keep most recent first, then trim
    data["entries"] = new_entries + data["entries"]
    if len(data["entries"]) > data["max_entries"]:
        data["entries"] = data["entries"][: data["max_entries"]]

    save_cache(data)

    result = {"cached": cached, "skipped": skipped, "entries": new_entries}
    logging.info(
        "near_memory_cache: session %s → cached %d entries, skipped %d",
        session_id,
        cached,
        skipped,
    )
    return result


def _summarize_via_llm(content: str) -> tuple:
    """Generate topic + summary via LLM (slow path).

    Returns (topic, summary) tuple.  Falls back to rule-based on error.
    """
    try:
        from agent.auxiliary_client import async_call_llm, extract_content_or_reasoning
        from model_tools import _run_async

        prompt = (
            "Extract from this AI response:\n"
            "1. Topic (≤20 Chinese characters — the single core subject)\n"
            "2. Summary (≤80 Chinese characters — what was done / decided)\n\n"
            "Return ONLY: topic|summary\n\n"
            f"RESPONSE:\n{content[:2000]}"
        )

        async def _call():
            resp = await async_call_llm(
                task="session_search",
                messages=[
                    {"role": "system", "content": "You extract short topic+summary. Be concise."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=200,
                timeout=10.0,
            )
            return extract_content_or_reasoning(resp)

        result = _run_async(_call(), timeout=15.0)
        if result and "|" in result:
            parts = result.split("|", 1)
            return parts[0].strip()[:30], parts[1].strip()[:120]
    except Exception as exc:
        logging.warning("near_memory_cache: LLM summarisation failed: %s", exc)

    # Fallback to rule-based
    return _extract_topic(content), _extract_summary(content)

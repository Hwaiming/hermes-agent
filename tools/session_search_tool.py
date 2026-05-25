#!/usr/bin/env python3
"""
Session Search Tool - Long-Term Conversation Recall

Single-shape tool with three calling modes (inferred from args, no explicit
mode parameter):

  1. DISCOVERY — pass ``query``. Runs FTS5, dedupes hits by session lineage,
     returns top N sessions each with: snippet, ±5 message window around the
     match, plus bookend_start (first 3 user+assistant msgs of session) and
     bookend_end (last 3). Zero LLM cost.

  2. SCROLL — pass ``session_id`` + ``around_message_id``. Returns a window
     of ±window messages centered on the anchor, no FTS5, no bookends. To
     scroll forward / backward, re-anchor on the last / first message id of
     the returned window.

  3. BROWSE — no args. Returns recent sessions chronologically (titles,
     previews, timestamps).

All three modes operate on the SQLite session DB via the FTS5 index and
the get_anchored_view / get_messages_around primitives in hermes_state.
No LLM calls anywhere — every shape returns actual messages from the DB.

History: PR #20238 (JabberELF) seeded a fast/summary dual-mode split; the
toolkit expansion in PR #26419 (yoniebans) added the anchored drill-down,
bookends, and sort. This module merges all of that into a single calling
shape with no mode parameter, no summary LLM path, and explicit scroll
support.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Union

# Sources that are excluded from session browsing/searching by default.
# Third-party integrations tag their sessions with HERMES_SESSION_SOURCE=tool
# so they don't clutter the user's session history.
_HIDDEN_SESSION_SOURCES = ("tool",)


def _get_session_search_timeout(default: float = 15.0) -> float:
    """Read auxiliary.session_search.timeout from config (seconds).

    The config value is the user-facing timeout — the LLM summarisation
    deadline.  If unset, defaults to 15 s so the tool returns promptly
    with whatever it has (including FTS5 raw snippets for unfinished
    summaries) rather than blocking the agent loop for minutes.
    """
    try:
        from hermes_cli.config import load_config
        config = load_config()
    except ImportError:
        return default
    aux = config.get("auxiliary", {}) if isinstance(config, dict) else {}
    task_config = aux.get("session_search", {}) if isinstance(aux, dict) else {}
    if not isinstance(task_config, dict):
        return default
    raw = task_config.get("timeout")
    if raw is None:
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    return max(5.0, min(value, 120.0))


# ── Query-vagueness heuristic ────────────────────────────────────────────
# If the query is too vague, a full FTS5 + LLM summarisation pipeline is
# wasteful — the user can narrow down with a single glance at recent
# sessions.  We surface the near-memory cache (or recent session list)
# and ask the user for more specific keywords instead.

_VAGUE_QUERY_MIN_CHARS = 4  # shorter than this → definitely vague

_VAGUE_PATTERNS: list = [
    # Pronouns / demonstratives with no content
    r"^(上次|上回|之前|那个|那些|这个|这些|咱|我们).{0,3}$",
    r"^(怎么|如何|怎样).{0,5}$",
    r"^(帮我|替我|查查|找找|看看).{0,5}$",
    r"^(那个|哪).{0,6}(修|改|做|处理|搞|弄)?.{0,3}$",
    # Single generic noun without qualifiers
    r"^(bug|错误|问题|文章|代码|方案|项目|任务|配置|文件|脚本)$",
]


def _is_query_too_vague(query: str) -> bool:
    """Return True if *query* is too vague for an FTS5 + LLM pipeline."""
    q = query.strip()
    if len(q) < _VAGUE_QUERY_MIN_CHARS:
        return True
    for pattern in _VAGUE_PATTERNS:
        if re.search(pattern, q, re.IGNORECASE):
            return True
    return False


# ── Inline near-memory cache (no external imports) ─────────────────────
# This avoids module-reload issues in long-running Hermes agent processes.
# The near_memory_cache.py module is the canonical API for writes; here we
# just read the same JSON file directly.

import os as _os

_NEAR_CACHE_PATH = _os.path.join(
    _os.path.expanduser(_os.environ.get("HERMES_HOME", "~/.hermes")),
    "sessions",
    "recent_cache.json",
)


def _load_near_cache() -> List[Dict[str, Any]]:
    """Return cache entries list (empty if file missing/corrupt)."""
    if not _os.path.exists(_NEAR_CACHE_PATH):
        return []
    try:
        with open(_NEAR_CACHE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data.get("entries", [])
    except Exception:
        return []
    return []


def _search_near_cache_inline(query: str) -> List[Dict[str, Any]]:
    """Fuzzy-search cache entries by keyword overlap."""
    entries = _load_near_cache()
    if not entries:
        return []
    keywords = query.strip().lower().split()
    scored = []
    for e in entries:
        text = (e.get("topic", "") + " " + e.get("summary", "")).lower()
        score = sum(1 for kw in keywords if kw in text)
        if score > 0:
            scored.append((score, e))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [entry for _, entry in scored]


def _format_timestamp(ts: Union[int, float, str, None]) -> str:
    """Convert a Unix timestamp (float/int) or ISO string to a human-readable date.

    Returns "unknown" for None, str(ts) if conversion fails.
    """
    if ts is None:
        return "unknown"
    try:
        if isinstance(ts, (int, float)):
            from datetime import datetime
            dt = datetime.fromtimestamp(ts)
            return dt.strftime("%B %d, %Y at %I:%M %p")
        if isinstance(ts, str):
            if ts.replace(".", "").replace("-", "").isdigit():
                from datetime import datetime
                dt = datetime.fromtimestamp(float(ts))
                return dt.strftime("%B %d, %Y at %I:%M %p")
            return ts
    except (ValueError, OSError, OverflowError) as e:
        logging.debug("Failed to format timestamp %s: %s", ts, e, exc_info=True)
    except Exception as e:
        logging.debug("Unexpected error formatting timestamp %s: %s", ts, e, exc_info=True)
    return str(ts)


def _format_conversation(messages: List[Dict[str, Any]]) -> str:
    """Format session messages into a readable transcript for summarization."""
    parts = []
    for msg in messages:
        role = msg.get("role", "unknown").upper()
        content = msg.get("content") or ""
        tool_name = msg.get("tool_name")

        if role == "TOOL" and tool_name:
            # Truncate long tool outputs
            if len(content) > 500:
                content = content[:250] + "\n...[truncated]...\n" + content[-250:]
            parts.append(f"[TOOL:{tool_name}]: {content}")
        elif role == "ASSISTANT":
            # Include tool call names if present
            tool_calls = msg.get("tool_calls")
            if tool_calls and isinstance(tool_calls, list):
                tc_names = []
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        name = tc.get("name") or tc.get("function", {}).get("name", "?")
                        tc_names.append(name)
                if tc_names:
                    parts.append(f"[ASSISTANT]: [Called: {', '.join(tc_names)}]")
                if content:
                    parts.append(f"[ASSISTANT]: {content}")
            else:
                parts.append(f"[ASSISTANT]: {content}")
        else:
            parts.append(f"[{role}]: {content}")

    return "\n\n".join(parts)


def _truncate_around_matches(
    full_text: str, query: str, max_chars: int = MAX_SESSION_CHARS
) -> str:
    """
    Truncate a conversation transcript to *max_chars*, choosing a window
    that maximises coverage of positions where the *query* actually appears.

    Strategy (in priority order):
    1. Try to find the full query as a phrase (case-insensitive).
    2. If no phrase hit, look for positions where all query terms appear
       within a 200-char proximity window (co-occurrence).
    3. Fall back to individual term positions.

    Once candidate positions are collected the function picks the window
    start that covers the most of them.
    """
    if len(full_text) <= max_chars:
        return full_text

    text_lower = full_text.lower()
    query_lower = query.lower().strip()
    match_positions: list[int] = []

    # --- 1. Full-phrase search ------------------------------------------------
    phrase_pat = re.compile(re.escape(query_lower))
    match_positions = [m.start() for m in phrase_pat.finditer(text_lower)]

    # --- 2. Proximity co-occurrence of all terms (within 200 chars) -----------
    if not match_positions:
        terms = query_lower.split()
        if len(terms) > 1:
            # Collect every occurrence of each term
            term_positions: dict[str, list[int]] = {}
            for t in terms:
                term_positions[t] = [
                    m.start() for m in re.finditer(re.escape(t), text_lower)
                ]
            # Slide through positions of the rarest term and check proximity
            rarest = min(terms, key=lambda t: len(term_positions.get(t, [])))
            for pos in term_positions.get(rarest, []):
                if all(
                    any(abs(p - pos) < 200 for p in term_positions.get(t, []))
                    for t in terms
                    if t != rarest
                ):
                    match_positions.append(pos)

    # --- 3. Individual term positions (last resort) ---------------------------
    if not match_positions:
        terms = query_lower.split()
        for t in terms:
            for m in re.finditer(re.escape(t), text_lower):
                match_positions.append(m.start())

    if not match_positions:
        # Nothing at all — take from the start
        truncated = full_text[:max_chars]
        suffix = "\n\n...[later conversation truncated]..." if max_chars < len(full_text) else ""
        return truncated + suffix

    # --- Pick window that covers the most match positions ---------------------
    match_positions.sort()

    best_start = 0
    best_count = 0
    for candidate in match_positions:
        ws = max(0, candidate - max_chars // 4)  # bias: 25% before, 75% after
        we = ws + max_chars
        if we > len(full_text):
            ws = max(0, len(full_text) - max_chars)
            we = len(full_text)
        count = sum(1 for p in match_positions if ws <= p < we)
        if count > best_count:
            best_count = count
            best_start = ws

    start = best_start
    end = min(len(full_text), start + max_chars)

    truncated = full_text[start:end]
    prefix = "...[earlier conversation truncated]...\n\n" if start > 0 else ""
    suffix = "\n\n...[later conversation truncated]..." if end < len(full_text) else ""
    return prefix + truncated + suffix


async def _summarize_session(
    conversation_text: str, query: str, session_meta: Dict[str, Any],
    timeout: float = None,
) -> Optional[str]:
    """Summarize a single session conversation focused on the search query.

    Accepts an optional *timeout* (seconds) for the LLM call.  When the
    deadline is exceeded the caller should treat the result as a failure
    and fall back to an FTS5 raw snippet — no retries here because the
    caller's deadline is the user's deadline, not ours.
    """
    system_prompt = (
        "You are reviewing a past conversation transcript to help recall what happened. "
        "Summarize the conversation with a focus on the search topic. Include:\n"
        "1. What the user asked about or wanted to accomplish\n"
        "2. What actions were taken and what the outcomes were\n"
        "3. Key decisions, solutions found, or conclusions reached\n"
        "4. Any specific commands, files, URLs, or technical details that were important\n"
        "5. Anything left unresolved or notable\n\n"
        "Be thorough but concise. Preserve specific details (commands, paths, error messages) "
        "that would be useful to recall. Write in past tense as a factual recap."
    )

    source = session_meta.get("source", "unknown")
    started = _format_timestamp(session_meta.get("started_at"))

    user_prompt = (
        f"Search topic: {query}\n"
        f"Session source: {source}\n"
        f"Session date: {started}\n\n"
        f"CONVERSATION TRANSCRIPT:\n{conversation_text}\n\n"
        f"Summarize this conversation with focus on: {query}"
    )

    max_retries = 1  # no retries — caller deadline is user's deadline
    for attempt in range(max_retries):
        try:
            response = await async_call_llm(
                task="session_search",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=MAX_SUMMARY_TOKENS,
                timeout=timeout,
            )
            content = extract_content_or_reasoning(response)
            if content:
                return content
            # Reasoning-only / empty — let the retry loop handle it
            logging.warning("Session search LLM returned empty content (attempt %d/%d)", attempt + 1, max_retries)
            if attempt < max_retries - 1:
                await asyncio.sleep(1 * (attempt + 1))
                continue
            return content
        except RuntimeError:
            logging.warning("No auxiliary model available for session summarization")
            return None
        except Exception as e:
            logging.debug("Error resolving parent for %s: %s", cur, e, exc_info=True)
            break
    return cur


def _shape_message(m: Dict[str, Any], anchor_id: Optional[int] = None) -> Dict[str, Any]:
    """Slim a message row for the tool response. Keeps content even if empty."""
    entry = {
        "id": m.get("id"),
        "role": m.get("role"),
        "content": m.get("content"),
        "timestamp": m.get("timestamp"),
    }
    if m.get("tool_name"):
        entry["tool_name"] = m.get("tool_name")
    if m.get("tool_calls"):
        entry["tool_calls"] = m.get("tool_calls")
    if m.get("tool_call_id"):
        entry["tool_call_id"] = m.get("tool_call_id")
    if anchor_id is not None and m.get("id") == anchor_id:
        entry["anchor"] = True
    # Strip None values to keep payload tight, but always keep content
    # (absent content is meaningful — tool-call-only assistant turns).
    return {k: v for k, v in entry.items() if v is not None or k in ("content",)}


def _list_recent_sessions(db, limit: int, current_session_id: str = None) -> str:
    """Return metadata for the most recent sessions (no LLM calls, no FTS5)."""
    try:
        sessions = db.list_sessions_rich(
            limit=limit + 5,
            exclude_sources=list(_HIDDEN_SESSION_SOURCES),
            order_by_last_active=True,
        )  # fetch extra so we can skip current

        current_root = _resolve_to_parent(db, current_session_id) if current_session_id else None

        results = []
        for s in sessions:
            sid = s.get("id", "")
            if current_root and (sid == current_root or sid == current_session_id):
                continue
            # Skip child / delegation sessions
            if s.get("parent_session_id"):
                continue
            results.append({
                "session_id": sid,
                "title": s.get("title") or None,
                "source": s.get("source", ""),
                "started_at": s.get("started_at", ""),
                "last_active": s.get("last_active", ""),
                "message_count": s.get("message_count", 0),
                "preview": s.get("preview", ""),
            })
            if len(results) >= limit:
                break

        return json.dumps({
            "success": True,
            "mode": "browse",
            "results": results,
            "count": len(results),
            "message": f"Showing {len(results)} most recent sessions. Pass a query= to search, or session_id+around_message_id to scroll.",
        }, ensure_ascii=False)
    except Exception as e:
        logging.error("Error listing recent sessions: %s", e, exc_info=True)
        return tool_error(f"Failed to list recent sessions: {e}", success=False)


def _scroll(
    db,
    session_id: str,
    around_message_id: int,
    window: int = 5,
    current_session_id: str = None,
) -> str:
    """Scroll shape: return a window of messages centered on an anchor.

    No FTS5, no bookends — just the slice. The discovery shape's lineage
    fixup is preserved: if the anchor doesn't live in the named session
    but does live in a child session in the same lineage, rebind silently.
    """
    if not isinstance(session_id, str) or not session_id.strip():
        return tool_error("scroll requires session_id", success=False)
    session_id = session_id.strip()

    try:
        around_message_id = int(around_message_id)
    except (TypeError, ValueError):
        return tool_error("scroll requires integer around_message_id", success=False)

    # Window clamp [1, 20]
    if not isinstance(window, int):
        try:
            window = int(window)
        except (TypeError, ValueError):
            window = 5
    window = max(1, min(window, 20))

    # Reject scrolling inside the active session lineage — those messages are
    # already in context.
    if current_session_id:
        a_root = _resolve_to_parent(db, session_id)
        c_root = _resolve_to_parent(db, current_session_id)
        if a_root and c_root and a_root == c_root:
            return tool_error(
                "scroll rejected: anchor lives in the current session lineage (already in your active context)",
                success=False,
            )

    # Session existence check
    try:
        session_meta = db.get_session(session_id) or {}
    except Exception as e:
        logging.debug("get_session failed for %s: %s", session_id, e, exc_info=True)
        session_meta = {}
    if not session_meta:
        return tool_error(f"session_id not found: {session_id}", success=False)

    # ── L1 · Near-memory cache lookup ──────────────────────────────────
    # Before hitting FTS5 + LLM (expensive), check the lightweight cache.
    # - Hit → return immediately (no LLM cost).
    # - Miss + vague query → surface recent cache to help user narrow down.
    # - Miss + specific query → fall through to FTS5 pipeline below.
    cache_hits = _search_near_cache_inline(query)
    if cache_hits:
        return json.dumps({
            "success": True,
            "mode": "cache",
            "query": query,
            "results": [
                {
                    "session_id": e["session_id"],
                    "date": e.get("date", ""),
                    "topic": e.get("topic", ""),
                    "summary": e.get("summary", ""),
                    "cached_at": e.get("cached_at", ""),
                }
                for e in cache_hits[:limit]
            ],
            "count": min(len(cache_hits), limit),
            "source": "near_memory_cache",
        }, ensure_ascii=False)

    if _is_query_too_vague(query):
        recent = _load_near_cache()[:10]
        return json.dumps({
            "success": True,
            "mode": "vague_redirect",
            "query": query,
            "message": (
                "查询太模糊，我无法确定你想找哪次会话。"
                "以下是最近 10 次有实质内容的会话，请告诉我大致时间或关键词，我来精准搜索。"
            ),
            "recent_sessions": [
                {
                    "session_id": e["session_id"],
                    "date": e.get("date", ""),
                    "topic": e.get("topic", ""),
                    "summary": e.get("summary", ""),
                }
                for e in recent
            ],
            "count": len(recent),
        }, ensure_ascii=False)

    try:
        view = db.get_messages_around(session_id, around_message_id, window=window)
    except Exception as e:
        logging.error("get_messages_around failed: %s", e, exc_info=True)
        return tool_error(f"failed to load messages: {e}", success=False)

    messages = view.get("window") or []

    # Lineage rebind: caller may have paired a parent session_id with a
    # message id that lives in a descendant (compaction / delegation creates
    # child sessions). Locate the real owning session and refetch.
    rebind_warning = None
    if not messages:
        owning = None
        try:
            conn = getattr(db, "_conn", None)
            if conn is not None:
                row = conn.execute(
                    "SELECT session_id FROM messages WHERE id = ?",
                    (around_message_id,),
                ).fetchone()
                owning = row[0] if row else None
        except Exception as e:
            logging.debug("owning-session lookup failed: %s", e, exc_info=True)
            owning = None
        if owning and owning != session_id:
            a_root = _resolve_to_parent(db, session_id)
            o_root = _resolve_to_parent(db, owning)
            if a_root and o_root and a_root == o_root:
                try:
                    rebind_view = db.get_messages_around(owning, around_message_id, window=window)
                    messages = rebind_view.get("window") or []
                    if messages:
                        view = rebind_view
                        rebind_warning = (
                            f"around_message_id {around_message_id} lives in {owning} "
                            f"(child of {session_id}); rebound transparently"
                        )
                        try:
                            session_meta = db.get_session(owning) or session_meta
                        except Exception:
                            pass
                        session_id = owning
                except Exception as e:
                    logging.debug("rebind get_messages_around failed: %s", e, exc_info=True)

    if not messages:
        return tool_error(
            f"around_message_id {around_message_id} not in session_id {session_id}",
            success=False,
        )

    response = {
        "success": True,
        "mode": "scroll",
        "session_id": session_id,
        "around_message_id": around_message_id,
        "session_meta": {
            "when": _format_timestamp(session_meta.get("started_at")),
            "source": session_meta.get("source"),
            "model": session_meta.get("model"),
            "title": session_meta.get("title"),
        },
        "window": window,
        "messages": [_shape_message(m, anchor_id=around_message_id) for m in messages],
        "messages_before": view.get("messages_before", 0),
        "messages_after": view.get("messages_after", 0),
    }
    if rebind_warning:
        response["warning"] = rebind_warning
    return json.dumps(response, ensure_ascii=False)


def _discover(
    db,
    query: str,
    role_filter: Optional[List[str]],
    limit: int,
    sort: Optional[str],
    current_session_id: str = None,
) -> str:
    """Discovery shape: FTS5 + anchored window + bookends per hit. Single call."""
    role_list = role_filter if role_filter else ["user", "assistant"]

    try:
        raw_results = db.search_messages(
            query=query,
            role_filter=role_list,
            exclude_sources=list(_HIDDEN_SESSION_SOURCES),
            limit=50,  # widen so dedup-by-lineage can find distinct sessions
            offset=0,
            sort=sort,
        )
    except Exception as e:
        logging.error("FTS5 search failed: %s", e, exc_info=True)
        return tool_error(f"Search failed: {e}", success=False)

    if not raw_results:
        return json.dumps({
            "success": True,
            "mode": "discover",
            "query": query,
            "results": [],
            "count": 0,
            "message": "No matching sessions found.",
        }, ensure_ascii=False)

    current_lineage_root = _resolve_to_parent(db, current_session_id) if current_session_id else None

    # Dedupe by lineage. Keep the raw owning session_id on the surviving
    # row — only that pairs validly with the FTS5 match id for the anchored
    # window. parent_session_id is exposed separately when different.
    seen_sessions = {}
    for r in raw_results:
        raw_sid = r["session_id"]
        resolved_sid = _resolve_to_parent(db, raw_sid)
        # Skip the current session lineage
        if current_lineage_root and resolved_sid == current_lineage_root:
            continue
        if current_session_id and raw_sid == current_session_id:
            continue
        if resolved_sid not in seen_sessions:
            row = dict(r)
            row["_lineage_root"] = resolved_sid
            seen_sessions[resolved_sid] = row
        if len(seen_sessions) >= limit:
            break

    results = []
    for lineage_root, match_info in seen_sessions.items():
        hit_sid = match_info.get("session_id") or lineage_root
        msg_id = match_info.get("id")
        try:
            view = db.get_anchored_view(hit_sid, msg_id, window=5, bookend=3)
        except Exception as e:
            logging.warning("get_anchored_view failed for %s/%s: %s", hit_sid, msg_id, e, exc_info=True)
            continue

        try:
            session_meta = db.get_session(lineage_root) or {}
        except Exception:
            session_meta = {}

        entry = {
            "session_id": hit_sid,
            "when": _format_timestamp(
                session_meta.get("started_at") or match_info.get("session_started")
            ),
            "source": session_meta.get("source") or match_info.get("source", "unknown"),
            "model": session_meta.get("model") or match_info.get("model") or "unknown",
            "title": session_meta.get("title") or None,
            "matched_role": match_info.get("role"),
            "match_message_id": msg_id,
            "snippet": match_info.get("snippet") or "",
            "bookend_start": [_shape_message(m) for m in (view.get("bookend_start") or [])],
            "messages": [_shape_message(m, anchor_id=msg_id) for m in (view.get("window") or [])],
            "bookend_end": [_shape_message(m) for m in (view.get("bookend_end") or [])],
            "messages_before": view.get("messages_before", 0),
            "messages_after": view.get("messages_after", 0),
        }
        if lineage_root and lineage_root != hit_sid:
            entry["parent_session_id"] = lineage_root
        results.append(entry)

    return json.dumps({
        "success": True,
        "mode": "discover",
        "query": query,
        "results": results,
        "count": len(results),
        "sessions_searched": len(seen_sessions),
    }, ensure_ascii=False)


def session_search(
    query: str = "",
    role_filter: str = None,
    limit: int = 3,
    db=None,
    current_session_id: str = None,
    # Scroll shape
    session_id: str = None,
    around_message_id: int = None,
    window: int = 5,
    # Discovery shape
    sort: str = None,
) -> str:
    """Single-shape tool. Mode inferred from which args are set.

        # Prepare all sessions for parallel summarization
        tasks = []
        for session_id, match_info in seen_sessions.items():
            try:
                messages = db.get_messages_as_conversation(session_id)
                if not messages:
                    continue
                session_meta = db.get_session(session_id) or {}
                conversation_text = _format_conversation(messages)
                conversation_text = _truncate_around_matches(conversation_text, query)
                tasks.append((session_id, match_info, conversation_text, session_meta))
            except Exception as e:
                logging.warning(
                    "Failed to prepare session %s: %s",
                    session_id,
                    e,
                    exc_info=True,
                )

        # Summarize all sessions with a firm deadline and graceful degradation.
        # asyncio.wait() returns (done, pending) so we can deliver completed
        # summaries immediately and fall back to FTS5 raw snippets for the rest
        # — the user gets *something* on time instead of waiting minutes.
        search_timeout = _get_session_search_timeout()
        max_concurrency = min(_get_session_search_max_concurrency(), max(1, len(tasks)))
        semaphore = asyncio.Semaphore(max_concurrency)

        async def _bounded_summary(text: str, meta: Dict[str, Any]) -> Optional[str]:
            async with semaphore:
                return await _summarize_session(text, query, meta, timeout=search_timeout)

        async def _summarize_all_with_deadline() -> List[Optional[str]]:
            coros = [
                _bounded_summary(text, meta)
                for _, _, text, meta in tasks
            ]
            # Python ≥3.12: asyncio.wait requires Task objects, not raw coroutines.
            tasks_ = [asyncio.ensure_future(c) for c in coros]
            done, pending = await asyncio.wait(tasks_, timeout=search_timeout)
            # Build results in task order.  Completed → summary; timed-out → None
            # (caller fills in FTS5 raw snippet as fallback).
            # Key on Task objects, not coroutines — asyncio.wait returns Tasks.
            task_to_idx = {task: i for i, task in enumerate(tasks_)}
            results = [None] * len(tasks_)
            for finished in done:
                idx = task_to_idx.get(finished)
                if idx is not None:
                    exc = finished.exception()
                    if exc is not None:
                        logging.warning(
                            "Session summary call %d failed: %s", idx, exc, exc_info=True,
                        )
                    else:
                        results[idx] = finished.result()
            if pending:
                logging.info(
                    "Session summarisation deadline reached (%s s): %d/%d summaries "
                    "completed, %d will use FTS5 raw snippets.",
                    search_timeout,
                    len(done),
                    len(tasks_),
                    len(pending),
                )
            return results

    Scroll wins over discovery when both are set — the agent has explicitly
    asked for a slice of a known session.
    """
    if db is None:
        try:
            # Use _run_async() which properly manages event loops across
            # CLI, gateway, and worker-thread contexts.  The previous
            # pattern (asyncio.run() in a ThreadPoolExecutor) created a
            # disposable event loop that conflicted with cached
            # AsyncOpenAI/httpx clients bound to a different loop,
            # causing deadlocks in gateway mode (#2681).
            from model_tools import _run_async
            results = _run_async(_summarize_all_with_deadline(),
                                 timeout=search_timeout + 5)
        except concurrent.futures.TimeoutError:
            logging.warning(
                "Session summarisation timed out after %.0f seconds (event-loop deadline)",
                search_timeout + 5,
                exc_info=True,
            )
            return json.dumps({
                "success": False,
                "error": (
                    f"Session summarisation timed out ({search_timeout:.0f}s deadline). "
                    "Try a more specific query or reduce the limit."
                ),
            }, ensure_ascii=False)

        summaries = []
        for (session_id, match_info, conversation_text, session_meta), result in zip(tasks, results):
            # result is None when the LLM call timed out (asyncio.wait deadline)
            # or failed — we fall back to an FTS5 raw snippet below.

            # Prefer resolved parent session metadata over FTS5 match metadata.
            # match_info carries source/model from the *child* session that contained
            # the FTS5 hit; after _resolve_to_parent() the session_id points to the
            # root, so session_meta has the authoritative platform/source for the
            # session the user actually cares about (#15909).
            entry = {
                "session_id": session_id,
                "when": _format_timestamp(
                    session_meta.get("started_at") or match_info.get("session_started")
                ),
                "source": session_meta.get("source") or match_info.get("source", "unknown"),
                "model": session_meta.get("model") or match_info.get("model"),
            }

    # Scroll shape takes precedence — explicit anchor beats any query.
    if (isinstance(session_id, str) and session_id.strip()) and around_message_id is not None:
        return _scroll(
            db=db,
            session_id=session_id,
            around_message_id=around_message_id,
            window=window,
            current_session_id=current_session_id,
        )

    # Limit clamp [1, 10]
    if not isinstance(limit, int):
        try:
            limit = int(limit)
        except (TypeError, ValueError):
            limit = 3
    limit = max(1, min(limit, 10))

    # Browse shape: no query → recent sessions.
    if not query or not isinstance(query, str) or not query.strip():
        return _list_recent_sessions(db, limit, current_session_id)

    # Parse role_filter
    role_list: Optional[List[str]] = None
    if isinstance(role_filter, str) and role_filter.strip():
        role_list = [r.strip() for r in role_filter.split(",") if r.strip()]

    # Normalise sort
    sort_norm: Optional[str] = None
    if isinstance(sort, str):
        candidate = sort.strip().lower()
        if candidate in ("newest", "oldest"):
            sort_norm = candidate

    return _discover(
        db=db,
        query=query.strip(),
        role_filter=role_list,
        limit=limit,
        sort=sort_norm,
        current_session_id=current_session_id,
    )


def check_session_search_requirements() -> bool:
    """Requires the SQLite state database."""
    try:
        from hermes_state import DEFAULT_DB_PATH
        return DEFAULT_DB_PATH.parent.exists()
    except ImportError:
        return False


SESSION_SEARCH_SCHEMA = {
    "name": "session_search",
    "description": (
        "Search past sessions stored in the local session DB, or scroll inside one. "
        "FTS5-backed retrieval over the SQLite message store. No LLM calls — every "
        "shape returns actual messages from the DB.\n\n"
        "THREE CALLING SHAPES\n\n"
        "  1) DISCOVERY — pass `query`:\n"
        "     session_search(query=\"auth refactor\", limit=3)\n"
        "     Runs FTS5, dedupes hits by session lineage, returns the top N sessions. "
        "Each result carries:\n"
        "       - session_id, title, when, source\n"
        "       - snippet: FTS5-highlighted match excerpt\n"
        "       - bookend_start: first 3 user+assistant messages of the session "
        "(the goal / kickoff)\n"
        "       - messages: ±5 messages around the FTS5 match, with the anchor message "
        "flagged (the hit in context)\n"
        "       - bookend_end: last 3 user+assistant messages of the session "
        "(the resolution / decisions)\n"
        "       - match_message_id, messages_before, messages_after\n"
        "     Bookends + window together let you reconstruct goal → match → resolution "
        "without paying for the whole transcript.\n\n"
        "  2) SCROLL — pass `session_id` + `around_message_id`:\n"
        "     session_search(session_id=\"...\", around_message_id=12345, window=10)\n"
        "     Returns a window of ±`window` messages centered on the anchor. No FTS5, "
        "no bookends — just the slice. Use after a discovery call when you need more "
        "context than the ±5 default window.\n"
        "       - To scroll FORWARD: pass messages[-1].id back as around_message_id.\n"
        "       - To scroll BACKWARD: pass messages[0].id back as around_message_id.\n"
        "       - The boundary message appears in both windows — orientation marker.\n"
        "       - When messages_before or messages_after is < window, you're at the "
        "start or end of the session.\n\n"
        "  3) BROWSE — no args:\n"
        "     session_search()\n"
        "     Returns recent sessions chronologically: titles, previews, timestamps. "
        "Use when the user asks \"what was I working on\" without naming a topic.\n\n"
        "FTS5 SYNTAX\n\n"
        "  AND is the default — multi-word queries require all terms. Use OR explicitly "
        "for broader recall (`alpha OR beta OR gamma`), quoted phrases for exact match "
        "(`\"docker networking\"`), boolean (`python NOT java`), or prefix wildcards "
        "(`deploy*`).\n\n"
        "WHEN TO USE\n\n"
        "  Reach for this on any \"what did we do about X\" / \"where did we leave Y\" / "
        "\"find the session where Z\" question — before gh, web search, or filesystem "
        "inspection. The session DB carries what was said when; external tools show "
        "current world state."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Search query (discovery shape). Keywords, phrases, or boolean "
                    "expressions to find in past sessions. Omit to browse recent "
                    "sessions. Ignored when session_id + around_message_id are set "
                    "(scroll shape)."
                ),
            },
            "limit": {
                "type": "integer",
                "description": (
                    "Discovery shape only. Max sessions to return (default 3, max 10). "
                    "Bump to 5–10 when the topic likely spans several sessions and you "
                    "want to pick the right one to scroll into."
                ),
                "default": 3,
            },
            "sort": {
                "type": "string",
                "enum": ["newest", "oldest"],
                "description": (
                    "Discovery shape only. Temporal bias on top of FTS5 ranking. Omit "
                    "to keep relevance-only ordering (suitable for exploratory recall — "
                    "\"what do we know about X\"). Set 'newest' for recency-shaped "
                    "questions (\"where did we leave X\"). Set 'oldest' for "
                    "origin-shaped questions (\"how did X start\"). Ignored in scroll "
                    "and browse shapes."
                ),
            },
            "session_id": {
                "type": "string",
                "description": (
                    "Scroll shape. Session to read inside. Use the session_id returned "
                    "from a prior discovery call. Must be paired with "
                    "around_message_id."
                ),
            },
            "around_message_id": {
                "type": "integer",
                "description": (
                    "Scroll shape. Message id to center the window on. From a discovery "
                    "result use match_message_id, or any id seen in a prior window. To "
                    "scroll forward pass the last window message's id; to scroll "
                    "backward pass the first."
                ),
            },
            "window": {
                "type": "integer",
                "description": (
                    "Scroll shape only. Messages to return on each side of the anchor "
                    "(anchor itself always included). Clamped to [1, 20]. Default 5."
                ),
                "default": 5,
            },
            "role_filter": {
                "type": "string",
                "description": (
                    "Optional. Comma-separated roles to include. Discovery defaults to "
                    "'user,assistant' (tool output is usually noise). Pass "
                    "'user,assistant,tool' to include tool output (debugging tool "
                    "behaviour) or 'tool' to search tool output only."
                ),
            },
        },
        "required": [],
    },
}


# --- Registry ---
from tools.registry import registry, tool_error

registry.register(
    name="session_search",
    toolset="session_search",
    schema=SESSION_SEARCH_SCHEMA,
    handler=lambda args, **kw: session_search(
        query=args.get("query") or "",
        role_filter=args.get("role_filter"),
        limit=args.get("limit", 3),
        session_id=args.get("session_id"),
        around_message_id=args.get("around_message_id"),
        window=args.get("window", 5),
        sort=args.get("sort"),
        db=kw.get("db"),
        current_session_id=kw.get("current_session_id"),
    ),
    check_fn=check_session_search_requirements,
    emoji="🔍",
)

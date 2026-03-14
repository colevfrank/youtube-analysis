"""
Agentic Content Triage Pipeline
================================
Deep evaluation of YouTube videos using Claude's tool-use capability.

Unlike score.py (which applies a rubric to metadata), this pipeline gives
Claude a set of tools and lets it autonomously decide how to investigate
each video. The model can:

    - fetch a video transcript for content analysis
    - analyze channel-level patterns (consistency, topical focus)
    - compute engagement anomaly scores (is engagement organic or inflated?)
    - cross-reference topic claims against known sources

This mirrors the architecture of a real content triage system: rather than
applying a fixed heuristic, an agent gathers evidence through multiple
tools and synthesizes a judgment. The tool-use loop is the key design
pattern — the model reasons about *what information it needs* before
making an assessment.

Architecture:
    1. Select a video for deep evaluation
    2. Send initial context + tool definitions to Claude
    3. Claude calls tools as needed (agentic loop)
    4. Each tool result is fed back for further reasoning
    5. Claude produces a final structured assessment

Usage:
    cd youtube-analysis && uv run triage.py [--video-id ID] [--sample N]

Requires ANTHROPIC_API_KEY and YOUTUBE_API_KEY in .env.
"""

import csv
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

DATA_DIR = Path(__file__).parent
WATCH_HISTORY_CSV = DATA_DIR / "watch-history.csv"
VIDEO_METADATA_CSV = DATA_DIR / "video-metadata.csv"
TRIAGE_REPORTS_CSV = DATA_DIR / "triage-reports.csv"

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-sonnet-4-6"
MAX_TOOL_ROUNDS = 6  # cap agentic loop iterations


# ---------------------------------------------------------------------------
# Tool Definitions (sent to Claude)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "fetch_transcript",
        "description": (
            "Fetch the transcript/captions of a YouTube video. Returns the "
            "full text of auto-generated or manual captions. Use this to "
            "analyze the actual spoken content of a video rather than relying "
            "solely on metadata. Returns first ~3000 chars of transcript."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "video_id": {
                    "type": "string",
                    "description": "YouTube video ID",
                }
            },
            "required": ["video_id"],
        },
    },
    {
        "name": "analyze_channel_patterns",
        "description": (
            "Analyze the viewing patterns for a specific channel in the user's "
            "watch history. Returns: total videos watched from this channel, "
            "date range of engagement, frequency, and category distribution. "
            "Use this to understand whether a channel is a one-off watch or "
            "a consistent part of the user's media diet."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "channel_name": {
                    "type": "string",
                    "description": "Name of the YouTube channel",
                }
            },
            "required": ["channel_name"],
        },
    },
    {
        "name": "compute_engagement_metrics",
        "description": (
            "Compute derived engagement metrics for a video. Returns: "
            "like-to-view ratio, comment-to-view ratio, estimated engagement "
            "percentile relative to other videos in the user's history, "
            "and flags for potential anomalies (e.g., unusually high "
            "like ratio suggesting inorganic engagement). Use this to assess "
            "whether a video's popularity metrics reflect genuine audience "
            "interest."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "video_id": {
                    "type": "string",
                    "description": "YouTube video ID",
                }
            },
            "required": ["video_id"],
        },
    },
    {
        "name": "get_category_context",
        "description": (
            "Get context about a YouTube category in the user's history. "
            "Returns: how many videos the user has watched in this category, "
            "average quality scores (if scored), top channels in this category "
            "from the user's history, and median engagement metrics. Use this "
            "to situate a video within the user's broader consumption patterns."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "category_id": {
                    "type": "string",
                    "description": "YouTube category ID (e.g., '27' for Education)",
                }
            },
            "required": ["category_id"],
        },
    },
    {
        "name": "submit_assessment",
        "description": (
            "Submit the final triage assessment for this video. Call this "
            "AFTER you have gathered enough evidence from the other tools. "
            "You MUST call this tool to complete the evaluation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "triage_label": {
                    "type": "string",
                    "enum": ["deep", "useful", "mixed", "shallow"],
                    "description": "Overall quality triage label",
                },
                "confidence": {
                    "type": "string",
                    "enum": ["high", "medium", "low"],
                    "description": (
                        "Confidence in the assessment. 'low' if metadata was "
                        "sparse and transcript unavailable."
                    ),
                },
                "key_signals": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "List of 3-5 key signals that informed the assessment, "
                        "e.g. 'transcript shows structured argument with citations' "
                        "or 'engagement ratio suggests clickbait pattern'"
                    ),
                },
                "summary": {
                    "type": "string",
                    "description": (
                        "2-3 sentence summary of the assessment and reasoning."
                    ),
                },
                "content_type": {
                    "type": "string",
                    "enum": [
                        "educational", "analysis", "news", "tutorial",
                        "entertainment", "music", "personal", "promotional",
                        "documentary", "other",
                    ],
                    "description": "Primary content type classification",
                },
            },
            "required": [
                "triage_label", "confidence", "key_signals",
                "summary", "content_type",
            ],
        },
    },
]


SYSTEM_PROMPT = """\
You are an agentic content triage system evaluating a YouTube video for quality
and substance. You have access to several tools to investigate the video.

Your goal: determine whether this video represents substantive, high-quality
content ("deep" or "useful") or lower-quality content ("mixed" or "shallow").

## Evaluation Strategy

1. Start by examining the metadata provided. Form an initial hypothesis.
2. Use tools to gather additional evidence. Prioritize:
   - fetch_transcript if the video might have substantive spoken content
   - analyze_channel_patterns to understand viewing context
   - compute_engagement_metrics to check for anomalies
3. Synthesize your evidence and submit a final assessment.

## Important Calibration Notes

- Not all content needs to be "educational" to score well. A well-crafted
  music video or comedy sketch can be "useful" if it shows real craft.
- Be skeptical of high engagement metrics alone — they can indicate
  clickbait as easily as genuine quality.
- Channel patterns matter: a video from a channel the user watches
  regularly may indicate sustained value, not just algorithmic capture.
- When evidence is limited (no transcript, sparse metadata), be honest
  about your confidence level.

After gathering evidence, you MUST call submit_assessment to finalize.
"""


# ---------------------------------------------------------------------------
# Tool Implementations
# ---------------------------------------------------------------------------

def _load_watch_history() -> pd.DataFrame:
    """Load watch history (cached after first call)."""
    if not hasattr(_load_watch_history, "_cache"):
        _load_watch_history._cache = pd.read_csv(
            WATCH_HISTORY_CSV, parse_dates=["timestamp"]
        )
    return _load_watch_history._cache


def _load_metadata() -> pd.DataFrame:
    """Load video metadata (cached after first call)."""
    if not hasattr(_load_metadata, "_cache"):
        _load_metadata._cache = pd.read_csv(VIDEO_METADATA_CSV)
    return _load_metadata._cache


def tool_fetch_transcript(video_id: str) -> str:
    """Fetch video transcript via youtube-transcript-api or captions API."""
    try:
        # Try youtube_transcript_api if available
        from youtube_transcript_api import YouTubeTranscriptApi

        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        full_text = " ".join(seg["text"] for seg in transcript)
        # Return first ~3000 chars
        if len(full_text) > 3000:
            full_text = full_text[:3000] + f"... [truncated, {len(full_text)} chars total]"
        return json.dumps({
            "status": "success",
            "video_id": video_id,
            "transcript_length": len(full_text),
            "transcript": full_text,
        })
    except ImportError:
        return json.dumps({
            "status": "unavailable",
            "video_id": video_id,
            "reason": "youtube_transcript_api not installed",
            "suggestion": "Evaluate based on metadata and other signals",
        })
    except Exception as e:
        return json.dumps({
            "status": "error",
            "video_id": video_id,
            "reason": str(e),
            "suggestion": "Transcript unavailable; evaluate based on metadata",
        })


def tool_analyze_channel_patterns(channel_name: str) -> str:
    """Analyze user's watching patterns for a given channel."""
    watch = _load_watch_history()
    channel_vids = watch[watch["channel_name"] == channel_name]

    if channel_vids.empty:
        return json.dumps({
            "channel": channel_name,
            "total_watches": 0,
            "note": "Channel not found in watch history",
        })

    # Compute patterns
    total = len(channel_vids)
    first_watch = channel_vids["timestamp"].min()
    last_watch = channel_vids["timestamp"].max()
    span_days = max((last_watch - first_watch).days, 1)
    frequency = total / (span_days / 30)  # watches per month

    # Unique videos vs rewatches
    unique_vids = channel_vids["video_id"].nunique()
    rewatch_ratio = total / unique_vids if unique_vids > 0 else 0

    # Temporal pattern: is engagement increasing, decreasing, or stable?
    if span_days > 60:
        midpoint = first_watch + pd.Timedelta(days=span_days / 2)
        first_half = len(channel_vids[channel_vids["timestamp"] < midpoint])
        second_half = total - first_half
        if second_half > first_half * 1.5:
            trend = "increasing"
        elif first_half > second_half * 1.5:
            trend = "decreasing"
        else:
            trend = "stable"
    else:
        trend = "insufficient_data"

    return json.dumps({
        "channel": channel_name,
        "total_watches": total,
        "unique_videos": unique_vids,
        "rewatch_ratio": round(rewatch_ratio, 2),
        "first_watch": first_watch.isoformat(),
        "last_watch": last_watch.isoformat(),
        "span_days": span_days,
        "watches_per_month": round(frequency, 1),
        "engagement_trend": trend,
    })


def tool_compute_engagement_metrics(video_id: str) -> str:
    """Compute derived engagement metrics and anomaly flags."""
    meta = _load_metadata()
    row = meta[meta["video_id"] == video_id]

    if row.empty:
        return json.dumps({
            "video_id": video_id,
            "status": "not_found",
        })

    row = row.iloc[0]
    views = pd.to_numeric(row.get("view_count"), errors="coerce")
    likes = pd.to_numeric(row.get("like_count"), errors="coerce")
    comments = pd.to_numeric(row.get("comment_count"), errors="coerce")
    duration = pd.to_numeric(row.get("duration_seconds"), errors="coerce")

    result = {"video_id": video_id, "status": "success"}

    if pd.notna(views) and views > 0:
        like_ratio = likes / views if pd.notna(likes) else None
        comment_ratio = comments / views if pd.notna(comments) else None
        result["views"] = int(views)
        result["like_to_view_ratio"] = round(like_ratio, 6) if like_ratio else None
        result["comment_to_view_ratio"] = round(comment_ratio, 6) if comment_ratio else None

        # Anomaly detection: compare to dataset-wide distributions
        all_views = pd.to_numeric(meta["view_count"], errors="coerce").dropna()
        all_like_ratios = (
            pd.to_numeric(meta["like_count"], errors="coerce")
            / all_views
        ).dropna()

        if like_ratio is not None:
            percentile = (all_like_ratios < like_ratio).mean() * 100
            result["like_ratio_percentile"] = round(percentile, 1)
            # Flag if in top 1% — possible inorganic engagement
            if percentile > 99:
                result["anomaly_flag"] = "unusually_high_like_ratio"
            elif percentile < 5:
                result["anomaly_flag"] = "unusually_low_engagement"
            else:
                result["anomaly_flag"] = "none"

        # Views percentile in user's history
        view_percentile = (all_views < views).mean() * 100
        result["view_percentile"] = round(view_percentile, 1)
    else:
        result["note"] = "View count unavailable"

    if pd.notna(duration):
        result["duration_seconds"] = int(duration)
        result["duration_minutes"] = round(duration / 60, 1)

    return json.dumps(result)


def tool_get_category_context(category_id: str) -> str:
    """Get context about a YouTube category in the user's history."""
    from analyze import CATEGORY_MAP

    watch = _load_watch_history()
    meta = _load_metadata()

    cat_name = CATEGORY_MAP.get(str(category_id), f"Category {category_id}")

    # Filter metadata for this category
    cat_meta = meta[meta["category_id"].astype(str) == str(category_id)]
    cat_video_ids = set(cat_meta["video_id"])

    # Watch events for this category
    cat_watches = watch[watch["video_id"].isin(cat_video_ids)]

    if cat_watches.empty:
        return json.dumps({
            "category_id": category_id,
            "category_name": cat_name,
            "total_watches": 0,
        })

    # Top channels in this category
    top_channels = (
        cat_watches["channel_name"]
        .value_counts()
        .head(5)
        .to_dict()
    )

    # Engagement stats for this category
    views = pd.to_numeric(cat_meta["view_count"], errors="coerce")
    likes = pd.to_numeric(cat_meta["like_count"], errors="coerce")

    # Check if quality scores exist
    quality_note = "No quality scores available yet"
    scores_csv = DATA_DIR / "quality-scores.csv"
    if scores_csv.exists():
        scores = pd.read_csv(scores_csv)
        cat_scores = scores[scores["video_id"].isin(cat_video_ids)]
        if not cat_scores.empty:
            quality_note = {
                "scored_count": len(cat_scores),
                "avg_composite": round(cat_scores["composite_score"].mean(), 2),
                "label_distribution": cat_scores["triage_label"].value_counts().to_dict(),
            }

    return json.dumps({
        "category_id": category_id,
        "category_name": cat_name,
        "total_watches": len(cat_watches),
        "unique_videos": cat_watches["video_id"].nunique(),
        "top_channels": top_channels,
        "median_views": int(views.median()) if not views.empty else None,
        "median_likes": int(likes.median()) if not likes.empty else None,
        "quality_scores": quality_note,
    })


# Tool dispatch
TOOL_HANDLERS = {
    "fetch_transcript": lambda inp: tool_fetch_transcript(inp["video_id"]),
    "analyze_channel_patterns": lambda inp: tool_analyze_channel_patterns(inp["channel_name"]),
    "compute_engagement_metrics": lambda inp: tool_compute_engagement_metrics(inp["video_id"]),
    "get_category_context": lambda inp: tool_get_category_context(inp["category_id"]),
}


# ---------------------------------------------------------------------------
# Agentic Loop
# ---------------------------------------------------------------------------

def run_triage(api_key: str, video_row: dict) -> dict | None:
    """Run the agentic triage loop for a single video.

    The loop:
        1. Send video context + tools to Claude
        2. If Claude calls a tool, execute it and feed the result back
        3. Repeat until Claude calls submit_assessment or we hit MAX_TOOL_ROUNDS
        4. Return the final assessment

    Returns:
        dict with assessment fields, or None on failure.
    """
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    # Build initial context from metadata
    from score import build_video_prompt
    initial_context = build_video_prompt(video_row)

    messages = [{"role": "user", "content": initial_context}]
    tools_used = []
    assessment = None

    for round_num in range(MAX_TOOL_ROUNDS):
        payload = {
            "model": MODEL,
            "max_tokens": 1024,
            "system": SYSTEM_PROMPT,
            "tools": TOOLS,
            "messages": messages,
        }

        try:
            resp = requests.post(
                ANTHROPIC_API_URL, headers=headers, json=payload, timeout=60
            )
            if resp.status_code == 429:
                retry_after = float(resp.headers.get("retry-after", 10))
                print(f"    Rate limited, waiting {retry_after:.0f}s...")
                time.sleep(retry_after)
                continue

            resp.raise_for_status()
            data = resp.json()

        except requests.RequestException as e:
            print(f"    API error in round {round_num + 1}: {e}")
            return None

        stop_reason = data.get("stop_reason")
        content_blocks = data.get("content", [])

        # Add assistant response to messages
        messages.append({"role": "assistant", "content": content_blocks})

        # Check for tool use
        tool_uses = [b for b in content_blocks if b.get("type") == "tool_use"]

        if not tool_uses:
            # No tool calls — model is done (or confused)
            break

        # Process each tool call
        tool_results = []
        for tool_call in tool_uses:
            tool_name = tool_call["name"]
            tool_input = tool_call["input"]
            tool_id = tool_call["id"]

            if tool_name == "submit_assessment":
                assessment = tool_input
                tools_used.append("submit_assessment")
                # Still need to send a tool_result to complete the turn
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": json.dumps({"status": "assessment_recorded"}),
                })
            elif tool_name in TOOL_HANDLERS:
                print(f"    → {tool_name}({json.dumps(tool_input)[:60]})")
                result = TOOL_HANDLERS[tool_name](tool_input)
                tools_used.append(tool_name)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": result,
                })
            else:
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": json.dumps({"error": f"Unknown tool: {tool_name}"}),
                    "is_error": True,
                })

        messages.append({"role": "user", "content": tool_results})

        if assessment:
            break

        time.sleep(1)  # rate limiting between rounds

    if assessment:
        assessment["tools_used"] = tools_used
        assessment["tool_rounds"] = len([t for t in tools_used if t != "submit_assessment"])

    return assessment


# ---------------------------------------------------------------------------
# Batch Processing
# ---------------------------------------------------------------------------

REPORT_FIELDNAMES = [
    "video_id",
    "title",
    "channel",
    "triage_label",
    "confidence",
    "content_type",
    "summary",
    "key_signals",
    "tools_used",
    "tool_rounds",
    "evaluated_at",
]


def load_already_triaged() -> set[str]:
    """Return set of video IDs already in triage-reports.csv."""
    if not TRIAGE_REPORTS_CSV.exists():
        return set()
    triaged = set()
    with open(TRIAGE_REPORTS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            triaged.add(row["video_id"])
    return triaged


def append_reports(rows: list[dict]) -> None:
    """Append triage reports to CSV."""
    file_exists = TRIAGE_REPORTS_CSV.exists()
    with open(TRIAGE_REPORTS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def run_batch_triage(
    video_id: str | None = None,
    sample_size: int | None = None,
) -> None:
    """Run deep triage on one or more videos.

    Args:
        video_id: Evaluate a specific video.
        sample_size: Evaluate a random sample of N videos.
    """
    load_dotenv(DATA_DIR / ".env")
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY in .env or environment")
        sys.exit(1)

    meta = pd.read_csv(VIDEO_METADATA_CSV)
    meta = meta[meta["api_title"] != "[unavailable]"]

    if video_id:
        meta = meta[meta["video_id"] == video_id]
        if meta.empty:
            print(f"Video {video_id} not found in metadata")
            sys.exit(1)
    else:
        # Skip already triaged
        already = load_already_triaged()
        meta = meta[~meta["video_id"].isin(already)]
        if sample_size and sample_size < len(meta):
            meta = meta.sample(n=sample_size, random_state=42)

    total = len(meta)
    print(f"\nDeep triage: {total} video{'s' if total != 1 else ''}\n")

    results = []
    for i, (_, row) in enumerate(meta.iterrows(), 1):
        title = row.get("api_title", "?")[:50]
        vid = row["video_id"]
        print(f"[{i}/{total}] {title}")
        print(f"  Video ID: {vid}")

        assessment = run_triage(api_key, row.to_dict())

        if assessment is None:
            print("  ✗ Failed to evaluate\n")
            continue

        label = assessment.get("triage_label", "?").upper()
        confidence = assessment.get("confidence", "?")
        tools = assessment.get("tools_used", [])
        print(f"  ✓ {label} (confidence: {confidence})")
        print(f"    Tools used: {', '.join(tools)}")
        print(f"    Summary: {assessment.get('summary', '?')[:120]}")
        print()

        report = {
            "video_id": vid,
            "title": row.get("api_title", ""),
            "channel": row.get("api_channel_title", ""),
            "triage_label": assessment.get("triage_label"),
            "confidence": assessment.get("confidence"),
            "content_type": assessment.get("content_type"),
            "summary": assessment.get("summary"),
            "key_signals": "|".join(assessment.get("key_signals", [])),
            "tools_used": "|".join(assessment.get("tools_used", [])),
            "tool_rounds": assessment.get("tool_rounds", 0),
            "evaluated_at": pd.Timestamp.now().isoformat(),
        }
        results.append(report)

        # Flush periodically
        if len(results) >= 5:
            append_reports(results)
            results = []

    if results:
        append_reports(results)

    print(f"Done. Results saved to {TRIAGE_REPORTS_CSV}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Deep agentic triage of YouTube videos"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--video-id", type=str, default=None,
        help="Evaluate a specific video by ID",
    )
    group.add_argument(
        "--sample", type=int, default=None,
        help="Evaluate a random sample of N videos",
    )
    args = parser.parse_args()

    run_batch_triage(video_id=args.video_id, sample_size=args.sample)


if __name__ == "__main__":
    main()

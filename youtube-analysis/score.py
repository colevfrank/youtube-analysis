"""
Content Quality Scorer
======================
Scores YouTube videos along structured quality dimensions using Claude.

The core design challenge here mirrors what any content triage system faces:
"quality" is multidimensional and context-dependent. Rather than a single score,
we decompose quality into orthogonal dimensions with explicit rubrics, letting
the LLM apply judgment while keeping the evaluation framework transparent and
auditable.

Dimensions:
    - depth: intellectual/informational depth vs. surface-level content
    - rigor: methodological care, sourcing, evidence quality
    - novelty: original perspective vs. rehash of common knowledge
    - effort: production effort and craft (editing, research, structure)
    - engagement_quality: earns attention via substance vs. clickbait/outrage

Each dimension is scored 1-5 with a short justification. An overall
`triage_label` maps the composite into a categorical bucket:
    - "deep"    — substantive, worth investing time in
    - "useful"  — practical value, competently made
    - "mixed"   — some value but significant filler or low rigor
    - "shallow" — low informational value, pure entertainment or clickbait

Usage:
    cd youtube-analysis && uv run score.py [--sample N] [--resume]

Requires ANTHROPIC_API_KEY in .env (or environment).
"""

import csv
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

DATA_DIR = Path(__file__).parent
VIDEO_METADATA_CSV = DATA_DIR / "video-metadata.csv"
QUALITY_SCORES_CSV = DATA_DIR / "quality-scores.csv"

# ---------------------------------------------------------------------------
# Quality Taxonomy
# ---------------------------------------------------------------------------

QUALITY_DIMENSIONS = {
    "depth": {
        "description": "Intellectual or informational depth",
        "rubric": {
            1: "Surface-level, no substance beyond the title",
            2: "Touches on a topic without developing it",
            3: "Provides moderate depth; covers the basics competently",
            4: "Goes meaningfully deeper than surface; teaches or challenges",
            5: "Deeply researched, expert-level treatment of subject matter",
        },
    },
    "rigor": {
        "description": "Methodological care, sourcing, and evidence quality",
        "rubric": {
            1: "No sourcing, speculative or misleading claims",
            2: "Anecdotal; claims made without evidence",
            3: "Some sourcing but inconsistent; mixes opinion and fact",
            4: "Well-sourced, acknowledges uncertainty, generally careful",
            5: "Rigorous methodology, primary sources, transparent reasoning",
        },
    },
    "novelty": {
        "description": "Originality of perspective or information",
        "rubric": {
            1: "Pure rehash of widely known information",
            2: "Common take with minor personal angle",
            3: "Useful synthesis or a somewhat fresh framing",
            4: "Genuinely original perspective or lesser-known information",
            5: "Breaks new ground; first-of-kind analysis or reporting",
        },
    },
    "effort": {
        "description": "Production effort, research, and craft",
        "rubric": {
            1: "Minimal effort; talking head with no preparation",
            2: "Basic effort; some structure but little research",
            3: "Competent production; adequate research and editing",
            4: "High production value; clearly well-researched",
            5: "Exceptional craft; extensive research, editing, visuals",
        },
    },
    "engagement_quality": {
        "description": "Whether attention is earned via substance vs. manipulation",
        "rubric": {
            1: "Pure clickbait, outrage-bait, or algorithmic optimization",
            2: "Relies heavily on sensationalism or emotional manipulation",
            3: "Mixed; some substance but also relies on hooks",
            4: "Earns attention primarily through quality content",
            5: "Fully substance-driven; no manipulative tactics",
        },
    },
}

TRIAGE_LABELS = {
    "deep": "Substantive, worth investing time in",
    "useful": "Practical value, competently made",
    "mixed": "Some value but significant filler or low rigor",
    "shallow": "Low informational value, pure entertainment or clickbait",
}

SCORE_FIELDNAMES = [
    "video_id",
    "depth",
    "rigor",
    "novelty",
    "effort",
    "engagement_quality",
    "composite_score",
    "triage_label",
    "rationale",
    "model",
    "scored_at",
]


# ---------------------------------------------------------------------------
# Prompt Construction
# ---------------------------------------------------------------------------

def build_rubric_text() -> str:
    """Render the quality dimensions and rubric into a prompt-friendly format."""
    lines = []
    for dim, spec in QUALITY_DIMENSIONS.items():
        lines.append(f"### {dim}: {spec['description']}")
        for level, desc in spec["rubric"].items():
            lines.append(f"  {level} — {desc}")
        lines.append("")
    return "\n".join(lines)


SYSTEM_PROMPT = f"""\
You are a content quality analyst. Your job is to evaluate YouTube videos
based on their metadata (title, description, tags, category, channel,
duration, engagement metrics). You apply a structured rubric to produce
consistent, calibrated quality assessments.

## Quality Dimensions and Rubric

{build_rubric_text()}

## Triage Labels

Based on the composite score (mean of all dimensions), assign a label:
- "deep" — composite >= 4.0
- "useful" — composite >= 3.0
- "mixed" — composite >= 2.0
- "shallow" — composite < 2.0

## Output Format

Respond with ONLY a JSON object (no markdown, no preamble):
{{
  "depth": <int 1-5>,
  "rigor": <int 1-5>,
  "novelty": <int 1-5>,
  "effort": <int 1-5>,
  "engagement_quality": <int 1-5>,
  "triage_label": "<deep|useful|mixed|shallow>",
  "rationale": "<2-3 sentence justification covering key signals>"
}}

## Calibration Notes

- Music videos, memes, and short clips are not inherently "shallow" — evaluate
  on their own terms. A well-crafted music video can score high on effort.
- "Engagement quality" is about *how* attention is captured, not *whether* the
  content is entertaining. Entertainment with craft scores higher than
  entertainment via outrage.
- When metadata is sparse (short description, no tags), say so in rationale
  and score conservatively toward the middle (2-3), not low.
- Duration alone does not indicate quality. A tight 5-minute explainer can
  outscore a rambling 2-hour podcast.
"""


def _safe_int(val) -> str:
    """Format a value as a comma-separated int, or 'N/A' if missing/NaN."""
    try:
        f = float(val)
        if f != f:  # NaN check
            return "N/A"
        return f"{int(f):,}"
    except (TypeError, ValueError):
        return "N/A"


def build_video_prompt(row: dict) -> str:
    """Build the user message for a single video evaluation."""
    # Format engagement metrics
    views = _safe_int(row.get("view_count"))
    likes = _safe_int(row.get("like_count"))
    comments = _safe_int(row.get("comment_count"))

    # Duration
    dur_sec = row.get("duration_seconds", 0)
    try:
        dur_val = float(dur_sec)
        if dur_val != dur_val or dur_val <= 0:
            raise ValueError
        mins, secs = divmod(int(dur_val), 60)
        hrs, mins = divmod(mins, 60)
        duration = f"{hrs}h{mins:02d}m{secs:02d}s" if hrs else f"{mins}m{secs:02d}s"
    except (TypeError, ValueError):
        duration = "N/A"

    # Tags (NaN from CSV comes through as float)
    tags = row.get("tags", "")
    if not isinstance(tags, str) or not tags:
        tag_str = "None"
    else:
        tag_str = ", ".join(tags.split("|")[:15])

    # Truncate description (same NaN guard)
    desc = row.get("description", "")
    if not isinstance(desc, str):
        desc = ""

    return f"""\
Evaluate this YouTube video:

Title: {row.get('api_title', row.get('title', 'Unknown'))}
Channel: {row.get('api_channel_title', row.get('channel_name', 'Unknown'))}
Category: {row.get('category_id', 'N/A')}
Duration: {duration}
Views: {views} | Likes: {likes} | Comments: {comments}
Tags: {tag_str}
Description:
{desc}
"""


# ---------------------------------------------------------------------------
# API Interaction
# ---------------------------------------------------------------------------

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-sonnet-4-6"
RATE_LIMIT_DELAY = 1.0  # seconds between requests


@dataclass
class ScoringStats:
    """Track scoring progress."""
    total: int = 0
    scored: int = 0
    errors: int = 0
    skipped: int = 0
    label_counts: dict = field(default_factory=lambda: {
        "deep": 0, "useful": 0, "mixed": 0, "shallow": 0
    })


def call_claude(api_key: str, video_row: dict) -> dict | None:
    """Score a single video via the Anthropic API. Returns parsed scores or None."""
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": MODEL,
        "max_tokens": 400,
        "system": SYSTEM_PROMPT,
        "messages": [
            {"role": "user", "content": build_video_prompt(video_row)},
        ],
    }

    for attempt in range(3):
        try:
            resp = requests.post(
                ANTHROPIC_API_URL, headers=headers, json=payload, timeout=30
            )

            if resp.status_code == 429:
                retry_after = float(resp.headers.get("retry-after", 10))
                print(f"  Rate limited, waiting {retry_after:.0f}s...")
                time.sleep(retry_after)
                continue

            resp.raise_for_status()
            data = resp.json()

            # Extract text from response
            text = ""
            for block in data.get("content", []):
                if block.get("type") == "text":
                    text += block["text"]

            # Parse JSON from response (strip markdown fences if present)
            text = text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0]
            scores = json.loads(text)

            # Validate
            for dim in QUALITY_DIMENSIONS:
                if dim not in scores or not isinstance(scores[dim], int):
                    raise ValueError(f"Missing or invalid score for {dim}")
                scores[dim] = max(1, min(5, scores[dim]))

            # Compute composite
            dim_scores = [scores[dim] for dim in QUALITY_DIMENSIONS]
            scores["composite_score"] = round(sum(dim_scores) / len(dim_scores), 2)

            # Validate triage label
            if scores.get("triage_label") not in TRIAGE_LABELS:
                # Recompute from composite
                c = scores["composite_score"]
                if c >= 4.0:
                    scores["triage_label"] = "deep"
                elif c >= 3.0:
                    scores["triage_label"] = "useful"
                elif c >= 2.0:
                    scores["triage_label"] = "mixed"
                else:
                    scores["triage_label"] = "shallow"

            return scores

        except json.JSONDecodeError as e:
            print(f"  JSON parse error: {e}")
            return None
        except requests.RequestException as e:
            print(f"  API error: {e}")
            return None
        except (ValueError, KeyError) as e:
            print(f"  Validation error: {e}")
            return None

    return None


# ---------------------------------------------------------------------------
# Batch Processing
# ---------------------------------------------------------------------------

def load_already_scored() -> set[str]:
    """Return set of video IDs already in quality-scores.csv."""
    if not QUALITY_SCORES_CSV.exists():
        return set()
    scored = set()
    with open(QUALITY_SCORES_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            scored.add(row["video_id"])
    return scored


def append_scores(rows: list[dict]) -> None:
    """Append scored rows to quality-scores.csv."""
    file_exists = QUALITY_SCORES_CSV.exists()
    with open(QUALITY_SCORES_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SCORE_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def score_videos(sample_size: int | None = None, resume: bool = True) -> None:
    """Main scoring loop.

    Args:
        sample_size: If set, score a random sample of N videos. Otherwise
                     score all videos with metadata.
        resume: If True, skip videos already in quality-scores.csv.
    """
    load_dotenv(DATA_DIR / ".env")
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY in .env or environment")
        sys.exit(1)

    if not VIDEO_METADATA_CSV.exists():
        print(f"ERROR: {VIDEO_METADATA_CSV} not found. Run enrich.py first.")
        sys.exit(1)

    meta = pd.read_csv(VIDEO_METADATA_CSV)
    # Filter out unavailable videos
    meta = meta[meta["api_title"] != "[unavailable]"].copy()
    print(f"Loaded {len(meta)} videos with metadata")

    # Resume support
    already_scored = load_already_scored() if resume else set()
    if already_scored:
        meta = meta[~meta["video_id"].isin(already_scored)]
        print(f"Already scored: {len(already_scored)}, remaining: {len(meta)}")

    if meta.empty:
        print("All videos already scored!")
        return

    # Sample if requested
    if sample_size and sample_size < len(meta):
        meta = meta.sample(n=sample_size, random_state=42)
        print(f"Sampled {sample_size} videos for scoring")

    stats = ScoringStats(total=len(meta))
    buffer: list[dict] = []

    print(f"\nScoring {len(meta)} videos...\n")

    for i, (_, row) in enumerate(meta.iterrows(), 1):
        vid = row["video_id"]
        title = row.get("api_title", "?")[:60]
        print(f"  [{i}/{stats.total}] {title}...", end="", flush=True)

        scores = call_claude(api_key, row.to_dict())

        if scores is None:
            stats.errors += 1
            print(" ERROR")
            continue

        # Build output row
        out = {
            "video_id": vid,
            "depth": scores["depth"],
            "rigor": scores["rigor"],
            "novelty": scores["novelty"],
            "effort": scores["effort"],
            "engagement_quality": scores["engagement_quality"],
            "composite_score": scores["composite_score"],
            "triage_label": scores["triage_label"],
            "rationale": scores.get("rationale", ""),
            "model": MODEL,
            "scored_at": pd.Timestamp.now().isoformat(),
        }

        buffer.append(out)
        stats.scored += 1
        stats.label_counts[scores["triage_label"]] += 1

        label = scores["triage_label"].upper()
        composite = scores["composite_score"]
        print(f" {composite:.1f} [{label}]")

        # Flush every 10 scores
        if len(buffer) >= 10:
            append_scores(buffer)
            buffer = []

        # Rate limiting
        time.sleep(RATE_LIMIT_DELAY)

    # Final flush
    if buffer:
        append_scores(buffer)

    # Summary
    print(f"\n{'=' * 50}")
    print(f"  Scoring Complete")
    print(f"{'=' * 50}")
    print(f"  Scored:  {stats.scored}")
    print(f"  Errors:  {stats.errors}")
    print(f"  Distribution:")
    for label, count in sorted(stats.label_counts.items()):
        pct = (count / stats.scored * 100) if stats.scored else 0
        bar = "█" * int(pct / 2)
        print(f"    {label:<10} {count:>4}  ({pct:>5.1f}%)  {bar}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Score YouTube videos on content quality dimensions"
    )
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Score a random sample of N videos (default: all)",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Re-score all videos, ignoring previous results",
    )
    args = parser.parse_args()

    score_videos(sample_size=args.sample, resume=not args.no_resume)


if __name__ == "__main__":
    main()

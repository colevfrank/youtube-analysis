"""Enrich watch-history.csv with YouTube Data API v3 metadata."""

import csv
import re
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv
import os

DATA_DIR = Path(__file__).parent
WATCH_HISTORY_CSV = DATA_DIR / "watch-history.csv"
VIDEO_METADATA_CSV = DATA_DIR / "video-metadata.csv"

API_URL = "https://www.googleapis.com/youtube/v3/videos"
BATCH_SIZE = 50
FLUSH_EVERY = 10  # flush to CSV every N batches

FIELDNAMES = [
    "video_id",
    "api_title",
    "api_channel_title",
    "channel_id",
    "published_at",
    "description",
    "category_id",
    "tags",
    "default_language",
    "default_audio_language",
    "duration_iso",
    "duration_seconds",
    "definition",
    "caption",
    "view_count",
    "like_count",
    "comment_count",
    "topic_categories",
]


def load_api_key() -> str:
    """Load the YouTube API key from .env."""
    load_dotenv(DATA_DIR / ".env")
    key = os.getenv("YOUTUBE_API_KEY")
    if not key or key == "your_key_here":
        print("ERROR: Set YOUTUBE_API_KEY in youtube-analysis/.env")
        sys.exit(1)
    return key


def parse_duration(iso: str) -> int:
    """Convert ISO 8601 duration (PT1H2M3S) to total seconds."""
    m = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", iso)
    if not m:
        return 0
    hours = int(m.group(1) or 0)
    minutes = int(m.group(2) or 0)
    seconds = int(m.group(3) or 0)
    return hours * 3600 + minutes * 60 + seconds


def load_already_fetched() -> set[str]:
    """Read video-metadata.csv and return the set of already-fetched video IDs."""
    if not VIDEO_METADATA_CSV.exists():
        return set()
    fetched = set()
    with open(VIDEO_METADATA_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fetched.add(row["video_id"])
    return fetched


def extract_video_fields(item: dict) -> dict:
    """Flatten one API response item into a row dict."""
    vid = item["id"]
    snippet = item.get("snippet", {})
    content = item.get("contentDetails", {})
    stats = item.get("statistics", {})
    topics = item.get("topicDetails", {})

    duration_iso = content.get("duration", "")
    tags = snippet.get("tags", [])
    topic_cats = topics.get("topicCategories", [])

    return {
        "video_id": vid,
        "api_title": snippet.get("title", ""),
        "api_channel_title": snippet.get("channelTitle", ""),
        "channel_id": snippet.get("channelId", ""),
        "published_at": snippet.get("publishedAt", ""),
        "description": snippet.get("description", ""),
        "category_id": snippet.get("categoryId", ""),
        "tags": "|".join(tags),
        "default_language": snippet.get("defaultLanguage", ""),
        "default_audio_language": snippet.get("defaultAudioLanguage", ""),
        "duration_iso": duration_iso,
        "duration_seconds": parse_duration(duration_iso),
        "definition": content.get("definition", ""),
        "caption": content.get("caption", ""),
        "view_count": stats.get("viewCount", ""),
        "like_count": stats.get("likeCount", ""),
        "comment_count": stats.get("commentCount", ""),
        "topic_categories": "|".join(topic_cats),
    }


def fetch_batch(ids: list[str], key: str) -> list[dict]:
    """Call the YouTube API for up to 50 video IDs. Returns list of row dicts."""
    params = {
        "part": "snippet,contentDetails,statistics,topicDetails",
        "id": ",".join(ids),
        "key": key,
    }
    resp = requests.get(API_URL, params=params, timeout=30)

    if resp.status_code == 403:
        error = resp.json().get("error", {})
        errors = error.get("errors", [{}])
        reason = errors[0].get("reason", "") if errors else ""
        if reason == "quotaExceeded":
            print("\nQuota exceeded! Progress saved. Re-run tomorrow to continue.")
            return None  # signal quota exhaustion
        print(f"\n403 error: {error.get('message', resp.text)}")
        return None

    resp.raise_for_status()
    data = resp.json()

    # Build rows for videos that returned data
    returned_ids = set()
    rows = []
    for item in data.get("items", []):
        rows.append(extract_video_fields(item))
        returned_ids.add(item["id"])

    # Mark unavailable videos so we don't re-request them
    for vid in ids:
        if vid not in returned_ids:
            rows.append({
                "video_id": vid,
                "api_title": "[unavailable]",
                **{k: "" for k in FIELDNAMES if k not in ("video_id", "api_title")},
            })

    return rows


def _append_results(rows: list[dict]) -> None:
    """Append rows to video-metadata.csv, creating with headers if new."""
    file_exists = VIDEO_METADATA_CSV.exists()
    with open(VIDEO_METADATA_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def enrich() -> None:
    """Main enrichment loop."""
    key = load_api_key()

    # Load watch history video IDs
    if not WATCH_HISTORY_CSV.exists():
        print(f"ERROR: {WATCH_HISTORY_CSV} not found. Run main.py first.")
        sys.exit(1)

    import pandas as pd
    df = pd.read_csv(WATCH_HISTORY_CSV)
    all_ids = df["video_id"].dropna().unique().tolist()
    print(f"Total unique video IDs in watch history: {len(all_ids)}")

    # Subtract already-fetched
    fetched = load_already_fetched()
    remaining = [vid for vid in all_ids if vid not in fetched]
    print(f"Already fetched: {len(fetched)}")
    print(f"Remaining to fetch: {len(remaining)}")

    if not remaining:
        print("All videos already enriched!")
        return

    # Batch through API
    batches = [remaining[i : i + BATCH_SIZE] for i in range(0, len(remaining), BATCH_SIZE)]
    print(f"Will make {len(batches)} API calls (~{len(batches)} quota units)\n")

    buffer: list[dict] = []
    for i, batch_ids in enumerate(batches, 1):
        rows = fetch_batch(batch_ids, key)
        if rows is None:
            # Quota or error — flush what we have and stop
            if buffer:
                _append_results(buffer)
                print(f"Flushed {len(buffer)} rows before stopping.")
            return

        buffer.extend(rows)
        print(f"  Batch {i}/{len(batches)} — got {len(rows)} rows", end="")

        if i % FLUSH_EVERY == 0:
            _append_results(buffer)
            print(f"  [flushed {len(buffer)} rows]", end="")
            buffer = []
        print()

    # Final flush
    if buffer:
        _append_results(buffer)
        print(f"\nFlushed final {len(buffer)} rows.")

    total = len(load_already_fetched())
    print(f"\nDone! video-metadata.csv now has {total} rows.")


if __name__ == "__main__":
    enrich()

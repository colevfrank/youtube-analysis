import html
import re
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd

DATA_DIR = Path(__file__).parent
WATCH_HISTORY_HTML = DATA_DIR / "watch-history.html"
WATCH_HISTORY_CSV = DATA_DIR / "watch-history.csv"

# Regex to extract the first content-cell body from each outer-cell block
# DOTALL needed because some video titles contain newlines
CONTENT_CELL_RE = re.compile(
    r'class="content-cell mdl-cell mdl-cell--6-col mdl-typography--body-1">'
    r"(.*?)</div>",
    re.DOTALL,
)

# Regex to extract <a href="...">...</a> tags (DOTALL for multi-line titles)
LINK_RE = re.compile(r'<a href="([^"]+)">([^<]+)</a>', re.DOTALL)

# Timestamp pattern: e.g. "Feb 6, 2026, 2:02:26 PM EST"
TIMESTAMP_RE = re.compile(
    r"([A-Z][a-z]{2} \d{1,2}, \d{4}, \d{1,2}:\d{2}:\d{2}\s[AP]M\s\w+)"
)


def extract_video_id(url: str) -> str | None:
    """Extract video ID from a YouTube watch URL."""
    try:
        qs = parse_qs(urlparse(url).query)
        return qs.get("v", [None])[0]
    except Exception:
        return None


def parse_watch_history(filepath: Path) -> pd.DataFrame:
    """Parse watch-history.html into a DataFrame."""
    print(f"Reading {filepath.name}...")
    raw = filepath.read_text(encoding="utf-8")

    # Split into outer-cell blocks
    blocks = raw.split('class="outer-cell')
    # First block is the HTML head/CSS, skip it
    blocks = blocks[1:]
    print(f"Found {len(blocks)} entries")

    rows = []
    for block in blocks:
        # Get the first content-cell (contains video info + timestamp)
        m = CONTENT_CELL_RE.search(block)
        if not m:
            continue
        body = m.group(1)

        # Extract all links from the content cell
        links = LINK_RE.findall(body)

        # Extract timestamp
        ts_match = TIMESTAMP_RE.search(body)
        timestamp_str = ts_match.group(1) if ts_match else None

        # Determine entry type and extract fields
        if not links:
            continue  # skip entries with no links at all

        video_url = links[0][0]
        title = html.unescape(links[0][1]).replace("\n", " ").strip()

        if len(links) >= 2 and "youtube.com/channel" in links[1][0]:
            # Normal entry: has channel link
            channel_url = links[1][0]
            channel_name = html.unescape(links[1][1])
            entry_type = "normal"
        else:
            channel_url = None
            channel_name = None
            entry_type = "ad" if "From Google Ads" in block else "removed"

        video_id = extract_video_id(video_url)

        rows.append(
            {
                "title": title,
                "video_url": video_url,
                "video_id": video_id,
                "channel_name": channel_name,
                "channel_url": channel_url,
                "timestamp": timestamp_str,
                "type": entry_type,
            }
        )

    df = pd.DataFrame(rows)

    # Parse timestamps — strip timezone abbreviation since it's not standard
    df["timestamp"] = df["timestamp"].str.replace(
        r"\s[A-Z]{2,5}$", "", regex=True
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%b %d, %Y, %I:%M:%S %p")

    return df


def main():
    df = parse_watch_history(WATCH_HISTORY_HTML)

    print(f"\nParsed {len(df)} entries")
    print(f"\nType breakdown:")
    print(df["type"].value_counts().to_string())
    print(f"\nDate range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"\nSample entries:")
    print(df.head(3).to_string())

    df.to_csv(WATCH_HISTORY_CSV, index=False)
    print(f"\nSaved to {WATCH_HISTORY_CSV}")


if __name__ == "__main__":
    main()

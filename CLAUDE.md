# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Analysis of personal YouTube history data exported via Google Takeout, extended with LLM-based content quality scoring and an agentic triage pipeline. The raw data lives as two HTML files at the repo root and is symlinked/copied into the `youtube-analysis/` project directory:

- `watch-history.html` (~53 MB) — full watch history
- `search-history.html` (~5.8 MB) — full search history

Both are Google Takeout "My Activity" HTML exports using Material Design Lite markup. Each activity entry is a `<div class="outer-cell">` containing `<div class="content-cell">` elements with links, channel names, and timestamps.

## Development

The project uses **uv** for package management with Python 3.13. All code lives in `youtube-analysis/`.

```bash
# Run scripts
cd youtube-analysis && uv run main.py
cd youtube-analysis && uv run enrich.py
cd youtube-analysis && uv run analyze.py
cd youtube-analysis && uv run score.py --sample 50
cd youtube-analysis && uv run triage.py --sample 10

# Add a dependency
cd youtube-analysis && uv add <package>

# Sync environment from lockfile
cd youtube-analysis && uv sync
```

## Architecture

### Pipeline Stages

The project follows a progressive enrichment pipeline:

1. **Parse** (`main.py`) — parses watch-history.html → `watch-history.csv`
2. **Enrich** (`enrich.py`) — enriches video IDs via YouTube Data API v3 → `video-metadata.csv`
3. **Analyze** (`analyze.py`) — merges CSVs, prints terminal report, saves PNG charts
4. **Score** (`score.py`) — LLM-based quality scoring via Anthropic API → `quality-scores.csv`
5. **Triage** (`triage.py`) — agentic deep evaluation with tool-use → `triage-reports.csv`

### Score vs. Triage

`score.py` and `triage.py` address the same underlying problem (content quality assessment) but represent two different architectural approaches:

- **score.py** is a **batch classifier**: it applies a fixed rubric to metadata via a single LLM call per video. Fast, scalable, and good for broad coverage. Analogous to a first-pass filter.

- **triage.py** is an **agentic evaluator**: it gives Claude tools (transcript fetching, engagement analysis, channel pattern analysis) and lets the model autonomously decide what evidence to gather. Slower but deeper. Analogous to an expert reviewer investigating a flagged item.

### Data Files (git-ignored)

- `watch-history.csv` — one row per watch event (title, video_id, channel, timestamp, type)
- `video-metadata.csv` — one row per unique video ID (duration, views, likes, category, tags, etc.)
- `quality-scores.csv` — LLM quality scores per video (5 dimensions + composite + triage label)
- `triage-reports.csv` — deep agentic assessments (label, confidence, key signals, tools used)
- Merge for analysis: `watch.merge(meta, on="video_id", how="left")`

### Environment Variables (in .env)

- `YOUTUBE_API_KEY` — for YouTube Data API v3 (enrich.py)
- `ANTHROPIC_API_KEY` — for Claude API (score.py, triage.py)

## Data Notes

- The HTML files are large (especially watch-history.html). Use streaming/chunked parsing.
- Timestamps in the Takeout HTML are localized date strings (e.g., "Feb 6, 2025, 12:17:00 PM PST").
- Video links follow the pattern `https://www.youtube.com/watch?v=VIDEO_ID`.
- Some entries may say "Watched a video that has been removed" with no link.

## Quality Scoring Dimensions

The scoring rubric in score.py evaluates videos across five dimensions (each 1-5):
- **depth** — intellectual/informational depth
- **rigor** — sourcing, evidence quality, methodological care
- **novelty** — originality of perspective or information
- **effort** — production effort and craft
- **engagement_quality** — earns attention via substance vs. manipulation

Triage labels derived from composite score: deep (>=4.0), useful (>=3.0), mixed (>=2.0), shallow (<2.0).

## Tool-Use Architecture (triage.py)

The agentic pipeline defines tools that Claude can invoke autonomously:
- `fetch_transcript` — retrieves video captions for content analysis
- `analyze_channel_patterns` — examines user's engagement history with a channel
- `compute_engagement_metrics` — derives engagement ratios and anomaly flags
- `get_category_context` — situates a video within the user's category consumption
- `submit_assessment` — finalizes the triage evaluation

The agent loop runs up to 6 rounds, with Claude deciding which tools to call and in what order based on the evidence it needs.

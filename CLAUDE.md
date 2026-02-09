# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Analysis of personal YouTube history data exported via Google Takeout. The raw data lives as two HTML files at the repo root and is symlinked/copied into the `youtube-analysis/` project directory:

- `watch-history.html` (~53 MB) — full watch history
- `search-history.html` (~5.8 MB) — full search history

Both are Google Takeout "My Activity" HTML exports using Material Design Lite markup. Each activity entry is a `<div class="outer-cell">` containing `<div class="content-cell">` elements with links, channel names, and timestamps.

## Development

The project uses **uv** for package management with Python 3.13. All code lives in `youtube-analysis/`.

```bash
# Run scripts
cd youtube-analysis && uv run main.py

# Add a dependency
cd youtube-analysis && uv add <package>

# Sync environment from lockfile
cd youtube-analysis && uv sync
```

## Architecture

- `youtube-analysis/main.py` — entry point
- `youtube-analysis/pyproject.toml` — project config; current deps: pandas, requests

## Data Notes

- The HTML files are large (especially watch-history.html). Use streaming/chunked parsing (e.g., `html.parser` or BeautifulSoup with `lxml`) rather than loading entirely into memory when possible.
- Timestamps in the Takeout HTML are localized date strings (e.g., "Feb 6, 2025, 12:17:00 PM PST").
- Video links follow the pattern `https://www.youtube.com/watch?v=VIDEO_ID`.
- Some entries may say "Watched a video that has been removed" with no link.

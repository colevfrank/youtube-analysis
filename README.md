# YouTube Watch History Analysis

I've been watching YouTube since ~2010. That's roughly 15 years of data sitting in a Google Takeout export — what I was into in college, what I binge-watched during COVID, how my interests have shifted over time. This project started as curiosity about what that history actually looks like when you lay it all out, and evolved into an attempt to answer a harder question: of the thousands of videos I've watched, which ones were actually worth my time?

## The pipeline

**1. Parse** → `main.py`
Extracts structured data from Google Takeout's HTML export (~53 MB) into a clean CSV. Handles removed videos, ads, and multi-line titles.

**2. Enrich** → `enrich.py`
Pulls video metadata (duration, views, likes, comments, tags, category, description) from the YouTube Data API v3 in batches of 50. Supports resumption across quota limits — my history has a lot of videos.

**3. Analyze** → `analyze.py`
Descriptive analysis: watch time by month and year, category breakdowns, top channels. The basic portrait of what my consumption looks like.

**4. Score** → `score.py`
This is where it gets more interesting. I use the Anthropic API to score each video across five quality dimensions:

| Dimension | What it measures |
|---|---|
| **Depth** | Intellectual/informational substance |
| **Rigor** | Sourcing, evidence quality, methodological care |
| **Novelty** | Originality of perspective or information |
| **Effort** | Production craft and research investment |
| **Engagement quality** | Whether attention is earned via substance vs. manipulation |

Each dimension is scored 1–5, producing a composite and a triage label: `deep`, `useful`, `mixed`, or `shallow`. The rubric is explicit in the system prompt — the model is told that music videos aren't inherently "shallow," that duration doesn't indicate quality, and that sparse metadata should yield moderate scores rather than low ones. Getting these calibration details right turned out to be where most of the actual work was.

**5. Triage** → `triage.py`
A deeper evaluation using Claude's tool-use capability. Instead of scoring from metadata alone, the model gets a set of investigative tools and decides what to look into:

- `fetch_transcript` — pull captions for content analysis
- `analyze_channel_patterns` — check my engagement history with the channel (frequency, trend, rewatch ratio)
- `compute_engagement_metrics` — derive engagement ratios and flag anomalies against my full history
- `get_category_context` — situate the video within my broader consumption in that category
- `submit_assessment` — render a final judgment with confidence level and key signals

The agent runs up to 6 rounds, calling whatever tools it thinks are relevant before submitting its assessment. Which tools it reaches for — and in what order — turns out to be interesting in itself.

## Two ways of evaluating the same thing

`score.py` and `triage.py` attack the same question from different angles. The batch scorer is fast — one API call per video, fixed rubric, structured output. You could run it over everything. But it only sees metadata. The agentic evaluator is slower and more expensive, but it can actually investigate: read the transcript, check whether my engagement with a channel is sustained or a one-off, look at whether engagement metrics seem organic. In practice you'd want both — broad coverage from the first, depth on the interesting cases from the second.

## Running it

```bash
cd youtube-analysis

# 1. Parse your Takeout HTML
uv run main.py

# 2. Enrich with YouTube API metadata (needs YOUTUBE_API_KEY in .env)
uv run enrich.py

# 3. Descriptive analysis
uv run analyze.py

# 4. Quality scoring (needs ANTHROPIC_API_KEY in .env)
uv run score.py --sample 50
uv run score.py                     # all videos, slow

# 5. Deep triage
uv run triage.py --sample 10
uv run triage.py --video-id VIDEO_ID
```

Both scoring scripts support resumption — they skip already-processed videos on re-run.

## Data

Raw data is git-ignored. To reproduce with your own history, export via [Google Takeout](https://takeout.google.com/) and place the HTML files in the `youtube-analysis/` directory.

## Dependencies

Managed with [uv](https://github.com/astral-sh/uv). See `pyproject.toml`. Key deps: `pandas`, `matplotlib`, `requests`, `python-dotenv`, `youtube-transcript-api`.

"""Analyze YouTube watch history: terminal report + PNG charts."""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

DATA_DIR = Path(__file__).parent
WATCH_CSV = DATA_DIR / "watch-history.csv"
META_CSV = DATA_DIR / "video-metadata.csv"

# YouTube Data API category IDs → human-readable names
CATEGORY_MAP = {
    "1": "Film & Animation",
    "2": "Autos & Vehicles",
    "10": "Music",
    "15": "Pets & Animals",
    "17": "Sports",
    "18": "Short Movies",
    "19": "Travel & Events",
    "20": "Gaming",
    "21": "Videoblogging",
    "22": "People & Blogs",
    "23": "Comedy",
    "24": "Entertainment",
    "25": "News & Politics",
    "26": "Howto & Style",
    "27": "Education",
    "28": "Science & Technology",
    "29": "Nonprofits & Activism",
    "30": "Movies",
    "31": "Anime/Animation",
    "32": "Action/Adventure",
    "33": "Classics",
    "34": "Comedy",
    "35": "Documentary",
    "36": "Drama",
    "37": "Family",
    "38": "Foreign",
    "39": "Horror",
    "40": "Sci-Fi/Fantasy",
    "41": "Thriller",
    "42": "Shorts",
    "43": "Shows",
    "44": "Trailers",
}


def load_data() -> pd.DataFrame:
    """Load and merge watch history with video metadata."""
    watch = pd.read_csv(WATCH_CSV, parse_dates=["timestamp"])
    meta = pd.read_csv(META_CSV)

    # Convert float category_id (e.g. 24.0) to clean string ("24") for mapping
    cat = pd.to_numeric(meta["category_id"], errors="coerce")
    meta["category_id"] = cat.where(cat.isna(), cat.astype("Int64").astype(str))
    meta["duration_seconds"] = pd.to_numeric(meta["duration_seconds"], errors="coerce")

    df = watch.merge(
        meta[["video_id", "duration_seconds", "category_id"]],
        on="video_id",
        how="left",
    )
    df["duration_hours"] = df["duration_seconds"] / 3600
    df["category"] = df["category_id"].map(CATEGORY_MAP).fillna("Unknown")
    df["year"] = df["timestamp"].dt.year
    df["month"] = df["timestamp"].dt.to_period("M")
    return df


def print_report(df: pd.DataFrame) -> None:
    """Print a rich terminal summary."""
    total_events = len(df)
    total_hours = df["duration_hours"].sum()
    total_days = total_hours / 24
    date_min = df["timestamp"].min()
    date_max = df["timestamp"].max()

    print("=" * 50)
    print("  YouTube Watch Time Report")
    print("=" * 50)
    print(f"  Total watch events:      {total_events:,}")
    print(f"  Total estimated time:    {total_hours:,.0f} hours ({total_days:,.0f} days)")
    print(f"  Date range:              {date_min:%b %Y} – {date_max:%b %Y}")
    print()

    # By year
    print(f"{'Year':<6} {'Events':>8} {'Hours':>8} {'Hrs/Day':>8}")
    print("-" * 34)

    for year, group in df.groupby("year"):
        events = len(group)
        hours = group["duration_hours"].sum()

        # Calculate actual calendar days for this year in the data
        year_start = max(group["timestamp"].min(), pd.Timestamp(year, 1, 1))
        year_end = min(group["timestamp"].max(), pd.Timestamp(year, 12, 31))
        cal_days = max((year_end - year_start).days + 1, 1)

        hrs_per_day = hours / cal_days
        print(f"{year:<6} {events:>8,} {hours:>8,.0f} {hrs_per_day:>8.1f}")

    print()

    # Top categories
    print("Top 10 Categories by Hours:")
    print("-" * 34)
    cat_hours = df.groupby("category")["duration_hours"].sum().sort_values(ascending=False)
    for cat, hours in cat_hours.head(10).items():
        print(f"  {cat:<25} {hours:>7,.0f} hrs")
    print()


def chart_monthly_hours(df: pd.DataFrame) -> None:
    """Line chart of monthly watch hours over time."""
    monthly = df.groupby("month")["duration_hours"].sum()
    monthly.index = monthly.index.to_timestamp()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(monthly.index, monthly.values, linewidth=1.2, color="#c4302b")
    ax.fill_between(monthly.index, monthly.values, alpha=0.15, color="#c4302b")
    ax.set_title("Monthly YouTube Watch Hours", fontsize=14, fontweight="bold")
    ax.set_ylabel("Hours")
    ax.set_xlabel("")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    path = DATA_DIR / "watch-time-by-month.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path.name}")


def chart_events_by_year(df: pd.DataFrame) -> None:
    """Bar chart of total watch events per year."""
    yearly = df.groupby("year").size()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(yearly.index.astype(str), yearly.values, color="#c4302b", width=0.6)
    ax.set_title("Watch Events by Year", fontsize=14, fontweight="bold")
    ax.set_ylabel("Events")
    ax.set_xlabel("")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    path = DATA_DIR / "watch-events-by-year.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path.name}")


def chart_category_by_year(df: pd.DataFrame) -> None:
    """Stacked bar chart of hours per category per year (top 6 + Other)."""
    # Identify top 6 categories by total hours
    top_cats = (
        df.groupby("category")["duration_hours"]
        .sum()
        .sort_values(ascending=False)
        .head(6)
        .index.tolist()
    )

    df_cat = df.copy()
    df_cat["cat_group"] = df_cat["category"].where(df_cat["category"].isin(top_cats), "Other")

    pivot = df_cat.pivot_table(
        index="year", columns="cat_group", values="duration_hours", aggfunc="sum", fill_value=0
    )

    # Order columns: top categories sorted by total, then Other last
    col_order = [c for c in top_cats if c in pivot.columns]
    if "Other" in pivot.columns:
        col_order.append("Other")
    pivot = pivot[col_order]

    colors = ["#c4302b", "#ff6f61", "#ffa600", "#58508d", "#003f5c", "#bc5090", "#aaaaaa"]

    fig, ax = plt.subplots(figsize=(12, 6))
    pivot.plot.bar(stacked=True, ax=ax, color=colors[: len(pivot.columns)], width=0.7)
    ax.set_title("Watch Hours by Category & Year", fontsize=14, fontweight="bold")
    ax.set_ylabel("Hours")
    ax.set_xlabel("")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.legend(title="Category", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticklabels(pivot.index.astype(str), rotation=0)
    fig.tight_layout()
    path = DATA_DIR / "hours-by-category-year.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path.name}")


def main():
    print("Loading data...")
    df = load_data()
    print(f"Loaded {len(df):,} watch events, "
          f"{df['duration_seconds'].notna().sum():,} with duration metadata\n")

    print_report(df)
    chart_monthly_hours(df)
    chart_events_by_year(df)
    chart_category_by_year(df)
    print("\nDone!")


if __name__ == "__main__":
    main()

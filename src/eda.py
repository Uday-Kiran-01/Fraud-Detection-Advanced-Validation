from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from src.utils.io_utils import IO
from src.utils.logger import get_logger

logger = get_logger("eda")


def run_eda(cfg: dict):
    io = IO(cfg)
    df = io.read_raw()

    # Basic profile
    logger.info(f"Loaded {len(df):,} rows with columns: {list(df.columns)}")
    start, end = df["click_time"].min(), df["click_time"].max()
    logger.info(f"Time span: {start} → {end}")

    eda_dir = Path(cfg["paths"]["eda_dir"])
    eda_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------
    # Hourly distribution
    # -----------------------
    df["hour"] = df["click_time"].dt.hour

    # Observed hours (whatever appears)
    hourly_raw = df.groupby("hour").size().reset_index(name="clicks")
    ax = hourly_raw.plot(x="hour", y="clicks", kind="bar")
    ax.set_title("Hourly Clicks (observed)")
    fig = ax.get_figure()
    fig.tight_layout(); fig.savefig(eda_dir / "hourly_clicks.png"); plt.close(fig)

    # Filled 0..23 for readability
    hourly_filled = (
        hourly_raw.set_index("hour")
        .reindex(range(24), fill_value=0)
        .reset_index()
        .rename(columns={"index": "hour"})
    )
    hourly_filled.columns = ["hour", "clicks"]
    ax = hourly_filled.plot(x="hour", y="clicks", kind="bar")
    ax.set_title("Hourly Clicks (0–23, missing filled)")
    fig = ax.get_figure()
    fig.tight_layout(); fig.savefig(eda_dir / "hourly_clicks_filled.png"); plt.close(fig)

    io.save_df(hourly_raw, str(eda_dir / "hourly_clicks.csv"))
    io.save_df(hourly_filled, str(eda_dir / "hourly_clicks_filled.csv"))
 
    # Channel vs is_attributed (with denominators)

    by_ch = (
        df.groupby("channel")["is_attributed"]
          .agg(clicks="size", downloads="sum", download_rate="mean")
          .reset_index()
          .sort_values("channel")
    )
    io.save_df(by_ch, str(eda_dir / "download_rate_by_channel_full.csv"))

    # Filtered view for meaningful rates (default >=1000 clicks)
    min_clicks = int(cfg.get("eda", {}).get("min_channel_clicks", 1000))
    by_ch_filtered = by_ch.query("clicks >= @min_clicks").copy()
    io.save_df(
        by_ch_filtered.sort_values("download_rate", ascending=False),
        str(eda_dir / f"download_rate_by_channel_{min_clicks}plus.csv"),
    )

    # Plot Top-K channels by clicks (more readable than plotting all)
    top_k = int(cfg.get("report", {}).get("top_k_entities", 10))
    top_by_clicks = by_ch.sort_values("clicks", ascending=False).head(top_k)
    ax = top_by_clicks.plot(x="channel", y="download_rate", kind="bar")
    ax.set_title(f"Download Rate by Channel (Top {top_k} by clicks)")
    fig = ax.get_figure()
    fig.tight_layout(); fig.savefig(eda_dir / "download_rate_by_channel_topk.png"); plt.close(fig)

    # Tiny summary
    summary = {
        "rows": int(len(df)),
        "unique": {c: int(df[c].nunique()) for c in ["ip", "app", "device", "os", "channel"] if c in df.columns},
        "time_span": {"start": str(start), "end": str(end)},
        "hours_present": int(df["hour"].nunique()),
        "min_channel_clicks_filter": min_clicks,
        "top_k_plotted": top_k,
    }
    io.save_json(summary, str(eda_dir / "summary.json"))
    logger.info("EDA artifacts saved.")

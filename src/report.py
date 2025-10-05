from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from src.utils.io_utils import IO
from src.utils.logger import get_logger

logger = get_logger("report")


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p


def build_report(cfg: dict, date_str: str | None = None):
    io = IO(cfg)
    paths = cfg["paths"]
    feats_path = Path(paths["features_dir"]) / "X.csv"
    batch_csv = Path(paths["out_dir"]) / "batch_scores.csv"
    stream_csv = Path(paths["stream_dir"]) / "predictions_stream.csv"
    reports_dir = _ensure_dir(Path(paths["reports_dir"]))

    # Load features (for ip/device/os/channel etc.)
    X = io.load_df(str(feats_path))

    # Load predictions: prefer stream, else batch
    if stream_csv.exists():
        preds = pd.read_csv(stream_csv)  # idx, y_score, y_pred
        preds = preds.rename(columns={"y_score": "score", "y_pred": "pred"})
        # Join rows by original index
        df = X.loc[preds["idx"]].copy()
        df["score"] = preds["score"].to_numpy()
        df["pred"] = preds["pred"].to_numpy()
        source = "stream"
    else:
        preds = pd.read_csv(batch_csv)  # y_true, y_score, y_pred
        preds = preds.rename(columns={"y_score": "score", "y_pred": "pred"})
        if len(preds) != len(X):
            raise ValueError(
                f"batch_scores length {len(preds)} != features length {len(X)}. "
                "Re-run infer to regenerate batch_scores for this feature set."
            )
        df = X.copy()
        df["score"] = preds["score"].to_numpy()
        df["pred"] = preds["pred"].to_numpy()
        if "y_true" in preds.columns:
            df["y_true"] = preds["y_true"].to_numpy()
        source = "batch"

    # Basic tallies
    total = len(df)
    suspects = int(df["pred"].sum())
    flag_rate = suspects / total if total else 0.0

    # Top flagged IPs/devices
    flagged = df.loc[df["pred"] == 1]
    top_ips = (
        flagged.groupby("ip")
        .size().sort_values(ascending=False).head(10)
        .rename("flags").reset_index()
    )
    top_devs = (
        flagged.groupby(["device", "os"])
        .size().sort_values(ascending=False).head(10)
        .rename("flags").reset_index()
    )

    # Score distribution plot
    ax = df["score"].plot(kind="hist", bins=40, title=f"Fraud Score Distribution ({source})")
    fig = ax.get_figure()
    fig.tight_layout()
    plot_path = reports_dir / "fraud_score_hist.png"
    fig.savefig(plot_path)
    plt.close(fig)

    # Save tables
    io.save_df(top_ips, str(reports_dir / "top_ips.csv"))
    io.save_df(top_devs, str(reports_dir / "top_devices_os.csv"))

    # Markdown summary
    md = [
        "# Daily Fraud Report",
        f"**Source**: {source}",
        f"**Total clicks processed**: {total:,}",
        f"**Suspected fraud (using provided predictions)**: {suspects:,}  "
        f"(_flag rate: {flag_rate:.2%}_)",
        "## Top Flagged IPs",
        top_ips.to_markdown(index=False),
        "## Top Flagged Devices (device, os)",
        top_devs.to_markdown(index=False),
        f"![Fraud score hist]({plot_path.name})",
    ]
    md_path = reports_dir / "report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(md))

    logger.info(f"Report saved to {md_path} (source={source})")

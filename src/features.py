from pathlib import Path
import pandas as pd
from scipy.stats import entropy

from src.utils.io_utils import IO
from src.utils.logger import get_logger

logger = get_logger("features")


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    # Sort and reset to guarantee unique, aligned index labels
    df = df.sort_values(["ip", "click_time"]).reset_index(drop=True)

    # ---------------------------
    # Aggregates per IP
    # ---------------------------
    df["clicks_per_ip"] = (
        df.groupby("ip")["ip"].transform("size").astype("uint32")
    )
    df["unique_apps_per_ip"] = (
        df.groupby("ip")["app"].transform("nunique").astype("uint16")
    )

    # Time since last click (per IP)
    df["time_since_last_click"] = (
        df.groupby("ip")["click_time"]
          .diff()
          .dt.total_seconds()
          .fillna(9.99e8)
          .astype("float32")
    )
    # First-click flag and clip huge sentinel to a sane max (1h)
    df["is_first_click"] = (df["time_since_last_click"] >= 9.0e8).astype("uint8")
    df["time_since_last_click"] = df["time_since_last_click"].clip(0, 3600).astype("float32")

    # Rolling 60s count per IP (requires DatetimeIndex inside each group)
    def _roll_60s_count(group: pd.DataFrame) -> pd.Series:
        g = group.set_index("click_time")  # make DatetimeIndex
        c = g["ip"].rolling("60s").count()
        return pd.Series(c.to_numpy(), index=group.index)

    df["clicks_in_last_60s"] = (
        df.groupby("ip", group_keys=False).apply(_roll_60s_count).astype("uint16")
    )

    # ---------------------------
    # Label-free global feature (no leakage)
    # ---------------------------
    df["clicks_by_channel"] = (
        df.groupby("channel")["channel"].transform("size").astype("uint32")
    )

    # ---------------------------
    # App diversity per IP (entropy)
    # ---------------------------
    def app_entropy(g: pd.DataFrame) -> float:
        counts = g["app"].value_counts().to_numpy()
        return float(entropy(counts)) if counts.size else 0.0

    ent = df.groupby("ip", sort=False).apply(app_entropy).rename("app_diversity_score")
    df = df.join(ent, on="ip")
    df["app_diversity_score"] = df["app_diversity_score"].astype("float32")

    # Final NA guard (shouldn’t be needed, but safe)
    for col in [
        "time_since_last_click",
        "clicks_in_last_60s",
        "app_diversity_score",
        "clicks_by_channel",
        "unique_apps_per_ip",
        "clicks_per_ip",
    ]:
        df[col] = df[col].fillna(0)

    return df


def build_features(cfg: dict):
    io = IO(cfg)
    df = io.read_raw()
    feats = add_features(df)

    # Candidate model features (we'll intersect with what's present)
    desired_feature_cols = [
        "ip", "app", "device", "os", "channel",
        "clicks_per_ip", "unique_apps_per_ip", "clicks_in_last_60s",
        "time_since_last_click", "clicks_by_channel", "app_diversity_score",
        "is_first_click",
    ]
    feature_cols = [c for c in desired_feature_cols if c in feats.columns]
    if len(feature_cols) == 0:
        raise ValueError("No feature columns found after engineering. Check input schema and add_features().")

    # ---- Label handling (robust) ----
    training_cfg = cfg.get("training", {}) if isinstance(cfg, dict) else {}
    target = training_cfg.get("target", "is_attributed")
    target_aliases = training_cfg.get("target_aliases", ["label", "target", "y", "is_fraud"])

    label_col = None
    if target in feats.columns:
        label_col = target
    else:
        for alt in target_aliases:
            if alt in feats.columns:
                label_col = alt
                break

    # Build X/y
    X = feats[feature_cols].copy()
    y = None
    if label_col is not None:
        y = pd.to_numeric(feats[label_col], errors="coerce").fillna(0).astype(int)

    # Save artifacts
    out_dir = Path(io.paths["features_dir"])
    io.save_df(X, str(out_dir / "X.csv"))
    if y is not None:
        io.save_df(y.to_frame(name=label_col), str(out_dir / "y.csv"))
        logger.info(f"Saved features: X.shape={X.shape}, y.shape={y.shape} (label='{label_col}')")
    else:
        logger.info(
            f"Saved features: X.shape={X.shape} (no label column found; "
            f"looked for '{target}' or aliases {target_aliases})"
        )

    # NEW: persist ids if available (for downstream preds join)
    if "click_id" in feats.columns:
        ids = feats[["click_id"]].copy()
        io.save_df(ids, str(out_dir / "ids.csv"))

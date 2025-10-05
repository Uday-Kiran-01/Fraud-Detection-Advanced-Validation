# src/evaluation/ab_test.py
import os
import json
import yaml
import math
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

# ---------- Config ----------
ROOT = Path(__file__).resolve().parents[2]  # repo root
CFG_PATH = ROOT / "configs" / "config.yaml"
with open(CFG_PATH, "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

DATA_PATH = ROOT / "data" / "input" / "train_sample.csv"
OUT_DIR = ROOT / "data" / "output"
REPORT_DIR = OUT_DIR / "reports"
MODELS_DIR = OUT_DIR / "models"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

EVAL_THRESHOLD = float(CFG.get("evaluation", {}).get("threshold", 0.5))
BASELINE_CLICKS_PER_IP_PER_5M = int(CFG.get("baseline", {}).get("clicks_per_ip_per_5min", 50))
BASELINE_UNIQUE_CHANNELS_PER_5M = int(CFG.get("baseline", {}).get("unique_channels_per_5min", 40))

# ---------- Helpers ----------
def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def _read_preds(model_name: str) -> pd.DataFrame:
    """Load preds_test.csv for a model and return a df with columns:
       y_true, y_score, optional row_id (if present)."""
    f = MODELS_DIR / model_name / "preds_test.csv"
    if not f.exists():
        raise FileNotFoundError(f"Missing {f}")

    df = pd.read_csv(f)
    df = _norm_cols(df)

    # Detect y_true
    ytrue_candidates = ["y_true", "label", "target", "is_attributed", "y"]
    y_true_col = next((c for c in ytrue_candidates if c in df.columns), None)
    if y_true_col is None:
        # sometimes 0/1 truth is called "gt" or "truth"
        for c in df.columns:
            if c.lower() in {"gt", "truth"}:
                y_true_col = c
                break
    if y_true_col is None:
        raise ValueError(f"{f}: could not find ground-truth column among {df.columns.tolist()}")

    # Detect y_score (probability-like)
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    score_pref = ["y_score", "fraud_score", "proba", "probability", "prob", "score", "prediction", "pred_score"]
    y_score_col = None
    for c in numeric_cols:
        cl = c.lower()
        if any(s in cl for s in score_pref):
            y_score_col = c
            break
    if y_score_col is None:
        # Fallback to a binary pred if present
        for c in numeric_cols:
            if c.lower() in {"y_pred", "y_pred_tuned", "y_pred_0p5", "pred"}:
                y_score_col = c
                break
    if y_score_col is None:
        raise ValueError(f"{f}: could not find a score/probability column among {df.columns.tolist()}")

    # Optional row id
    rowid_candidates = ["row_id", "click_id", "id", "idx", "index", "Unnamed: 0"]
    rowid_col = next((c for c in rowid_candidates if c in df.columns), None)
    if rowid_col == "index":
        df = df.rename(columns={"index": "row_id"})
        rowid_col = "row_id"
    if rowid_col == "Unnamed: 0":
        df = df.rename(columns={"Unnamed: 0": "row_id"})
        rowid_col = "row_id"

    out = pd.DataFrame({
        "y_true": pd.to_numeric(df[y_true_col], errors="coerce").fillna(0).astype(int),
        "y_score": pd.to_numeric(df[y_score_col], errors="coerce").fillna(0).clip(0, 1)
    })
    if rowid_col is not None:
        out["row_id"] = df[rowid_col]
    return out

def _to_labels(scores: np.ndarray, thr: float) -> np.ndarray:
    return (scores >= thr).astype(int)

def _metrics(y_true: np.ndarray, y_score: np.ndarray, thr: float) -> Dict[str, float]:
    y_pred = _to_labels(y_score, thr)
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "flag_rate": float(y_pred.mean()),
    }
    try:
        out["roc_auc"] = roc_auc_score(y_true, y_score)
    except Exception:
        out["roc_auc"] = float("nan")
    return out

def _load_input_df() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Input data not found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    if "click_time" in df.columns:
        df["click_time"] = pd.to_datetime(df["click_time"], errors="coerce")
        df["window_5m"] = (df["click_time"].astype("int64") // (5 * 60 * 10**9))
        df["minute"] = (df["click_time"].astype("int64") // (60 * 10**9))
    else:
        df["window_5m"] = 0
        df["minute"] = 0
    if "row_id" not in df.columns:
        df = df.reset_index().rename(columns={"index": "row_id"})
    return df

def _baseline_flags_on_slice(df: pd.DataFrame) -> pd.Series:
    if "ip" not in df.columns:
        return pd.Series([0] * len(df), index=df.index)
    # Rule 1: burst per minute
    cpm = df.groupby(["ip", "minute"]).size().rename("cpm").reset_index()
    df = df.merge(cpm, on=["ip", "minute"], how="left")
    # Rule 2: unique channels/5m
    if "channel" in df.columns:
        uniq_ch = df.groupby(["ip","window_5m"])["channel"].nunique().rename("uniq_ch_5m").reset_index()
        df = df.merge(uniq_ch, on=["ip","window_5m"], how="left")
    else:
        df["uniq_ch_5m"] = 0
    flags = (
        (df["cpm"].fillna(0) >= BASELINE_CLICKS_PER_IP_PER_5M) |
        (df["uniq_ch_5m"].fillna(0) >= BASELINE_UNIQUE_CHANNELS_PER_5M)
    ).astype(int)
    return flags

# ---------- Main ----------
def run_ab_test() -> pd.DataFrame:
    model_names = ["logistic", "rf", "lgbm"]
    model_results = []
    has_any_model = False

    # 1) Evaluate models directly from preds_test.csv (uses y_true in that file)
    preds_per_model: Dict[str, pd.DataFrame] = {}
    for m in model_names:
        try:
            dfp = _read_preds(m)  # columns: y_true, y_score, optional row_id
            y_true = dfp["y_true"].to_numpy()
            y_score = dfp["y_score"].to_numpy()
            met = _metrics(y_true, y_score, thr=EVAL_THRESHOLD)
            model_results.append({"model": m, **met})
            preds_per_model[m] = dfp
            has_any_model = True
        except Exception as e:
            warnings.warn(f"Skipping {m}: {e}")

    if not has_any_model:
        raise RuntimeError("No usable model predictions found. Ensure preds_test.csv exists and has y_true & y_score.")

    # 2) Try to compute baseline on the same rows (only if we can map with row_id)
    baseline_added = False
    try:
        # pick the first model that has row_id
        any_with_id = next((dfp for dfp in preds_per_model.values() if "row_id" in dfp.columns), None)
        if any_with_id is not None:
            df_in = _load_input_df()
            eval_ids = set(pd.Series(any_with_id["row_id"]).astype(str))
            df_in = df_in.copy()
            df_in["row_id"] = df_in["row_id"].astype(str)
            df_slice = df_in[df_in["row_id"].isin(eval_ids)].copy()
            if len(df_slice) > 0 and "is_attributed" in df_slice.columns:
                base_scores = _baseline_flags_on_slice(df_slice).astype(float).to_numpy()
                base_truth = df_slice["is_attributed"].astype(int).to_numpy()
                met = _metrics(base_truth, base_scores, thr=0.5)  # baseline is 0/1, threshold irrelevant
                model_results.append({"model": "baseline_rules", **met})
                baseline_added = True
    except Exception as e:
        warnings.warn(f"Baseline skipped (no aligned rows): {e}")

    # 3) Save comparison
    comp = pd.DataFrame(model_results)
    # order columns
    cols = ["model", "accuracy", "precision", "recall", "f1", "roc_auc", "flag_rate"]
    comp = comp[[c for c in cols if c in comp.columns]]
    comp.to_csv(REPORT_DIR / "model_comparison.csv", index=False)

    # 4) Plot
    try:
        import matplotlib.pyplot as plt
        plt.figure()
        metrics_to_plot = ["precision", "recall", "f1", "roc_auc"]
        x = np.arange(len(comp["model"]))
        width = 0.18
        for i, met in enumerate(metrics_to_plot):
            plt.bar(x + i*width - width*1.5, comp[met].astype(float), width, label=met)
        plt.xticks(x, comp["model"], rotation=0)
        plt.ylabel("score")
        plt.title("Model Evaluation (direct from preds_test.csv)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(REPORT_DIR / "model_comparison.png", dpi=160)
        plt.close()
    except Exception as e:
        warnings.warn(f"Plotting failed: {e}")

    # 5) Markdown summary
    lines = [
        "# A/B Comparison — Model vs Baseline\n",
        f"- Threshold for models: **{EVAL_THRESHOLD}**\n",
        "- Evaluation computed directly from each `preds_test.csv` (uses its `y_true`).\n",
        f"- Baseline included: **{'Yes' if baseline_added else 'No (no aligned row_id)'}**\n",
        "\n## Results\n",
        comp.to_markdown(index=False),
        "\n\n*Saved:* `data/output/reports/model_comparison.csv`, `model_comparison.png`.\n"
    ]
    (REPORT_DIR / "ab_report.md").write_text("\n".join(lines), encoding="utf-8")
    return comp

if __name__ == "__main__":
    df = run_ab_test()
    print(df)

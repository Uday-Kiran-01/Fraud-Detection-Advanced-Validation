from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib

from src.utils.io_utils import IO
from src.utils.logger import get_logger

logger = get_logger("infer")


def _model_dir(cfg, model_key: str) -> Path:
    base = Path(cfg["paths"]["models_dir"])
    cand = base / (model_key or cfg.get("training", {}).get("model_type", "rf")).lower()
    return cand if cand.exists() else base  # back-compat to single-folder layout


def load_model(cfg: dict, model_key: str = None):
    mdir = _model_dir(cfg, model_key)
    model_path = mdir / "model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    model = joblib.load(model_path)

    thr = 0.5
    metapath = mdir / "metrics.json"
    if metapath.exists():
        with open(metapath, "r") as f:
            meta = json.load(f)
            thr = float(meta.get("best_threshold", 0.5))
    logger.info(f"Loaded model from {model_path} (thr={thr:.4f})")
    return model, thr


def _pick_threshold_from_counts(scores: np.ndarray, target_flags: int = None, flag_rate: float = None):
    n = len(scores)
    if target_flags is not None and target_flags > 0:
        k = min(max(1, int(target_flags)), n)
        # threshold = kth largest score
        idx = np.argpartition(scores, n - k)[n - k]
        thr = float(idx)
        # above line returns index; fix to value:
        thr = float(np.partition(scores, n - k)[n - k])
        return thr
    if flag_rate is not None and 0 < flag_rate < 1:
        k = int(round(n * flag_rate))
        k = max(1, min(k, n))
        return float(np.partition(scores, n - k)[n - k])
    return None


def run_inference(cfg: dict, model_key: str = None, threshold: float = None,
                  target_flags: int = None, flag_rate: float = None):
    io = IO(cfg)
    X = io.load_df(str(Path(cfg["paths"]["features_dir"]) / "X.csv"))
    y = io.load_df(str(Path(cfg["paths"]["features_dir"]) / "y.csv"))["is_attributed"]

    model, tuned_thr = load_model(cfg, model_key=model_key)

    # 1) get raw scores
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X)[:, 1]
    else:
        scores = model.predict(X)

    # 2) decide threshold (priority: explicit > target_flags > flag_rate > tuned)
    thr = float(threshold) if threshold is not None else None
    if thr is None:
        thr = _pick_threshold_from_counts(scores, target_flags=target_flags, flag_rate=flag_rate)
    if thr is None:
        thr = float(tuned_thr)

    preds = (scores >= thr).astype(int)

    out = Path(cfg["paths"]["out_dir"]) / "batch_scores.csv"
    df_out = pd.DataFrame({"y_true": y, "y_score": scores, "y_pred": preds})
    io.save_df(df_out, str(out))
    logger.info(f"Batch inference saved to {out} with thr={thr:.4f} "
                f" | flagged={int(preds.sum())} ({preds.mean():.4%})")

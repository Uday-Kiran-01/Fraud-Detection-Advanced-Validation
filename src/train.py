from pathlib import Path
import logging
import json
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.utils.io_utils import IO
from src.utils.logger import get_logger

logger = get_logger("train")

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False


def _model_from_type(model_type: str, cfg: dict):
    model_type = (model_type or "rf").lower()
    tcfg = cfg.get("training", {})
    n_estimators = int(tcfg.get("n_estimators", 200))
    max_depth = tcfg.get("max_depth", None)
    if max_depth in (-1, "None", None):
        max_depth = None
    else:
        max_depth = int(max_depth)
    class_weight = tcfg.get("class_weight", "balanced_subsample")
    random_state = int(tcfg.get("random_state", 42))

    if model_type == "rf":
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=-1,
        )

    if model_type == "logistic":
        # Simple linear baseline; works with our numeric ID features.
        return Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=False)),
                ("clf", LogisticRegression(
                    max_iter=1000,
                    class_weight=class_weight if class_weight != "balanced_subsample" else "balanced",
                    random_state=random_state,
                )),
            ]
        )

    if model_type == "lgbm" and HAS_LGBM:
        return LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=-1 if max_depth is None else max_depth,
            class_weight=class_weight if class_weight != "balanced_subsample" else "balanced",
            random_state=random_state,
            n_jobs=-1,
        )

    logger.warning("Unknown or unavailable model_type '%s'; falling back to RandomForest.", model_type)
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1,
    )


def _evaluate(y_true, y_score, y_pred):
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "report": classification_report(y_true, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def _threshold_for_precision(y_true, y_score, target_precision=0.9):
    # Find threshold achieving >= target_precision with max recall; fallback 0.5
    p, r, thr = precision_recall_curve(y_true, y_score)
    best = None
    for prec, rec, t in zip(p[:-1], r[:-1], thr):
        if prec >= float(target_precision):
            if best is None or rec > best[1]:
                best = (t, rec, prec)
    return float(best[0]) if best else 0.5


def train(cfg: dict):
    io = IO(cfg)

    # ---------- Load features ----------
    X = io.load_df(str(Path(cfg["paths"]["features_dir"]) / "X.csv"))
    y = io.load_df(str(Path(cfg["paths"]["features_dir"]) / "y.csv"))["is_attributed"]

    # ---------- Class balance guard ----------
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        raise RuntimeError(
            f"Need at least two classes to train; found {dict(zip(map(int, classes), map(int, counts)))}. "
            "Increase io.nrows or enable io.sample_frac in configs/config.yaml."
        )
    logger.info(f"class counts: {dict(zip(map(int, classes), map(int, counts)))}")

    # ---------- Split ----------
    tcfg = cfg.get("training", {})
    test_size = float(tcfg.get("test_size", 0.2))
    random_state = int(tcfg.get("random_state", 42))
    model_key = str(tcfg.get("model_type", "rf")).lower()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # ---------- Model ----------
    model = _model_from_type(model_key, cfg)
    model.fit(X_train, y_train)

    # ---------- Scores at 0.5 ----------
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        # Some models may not expose calibrated proba; use decision as score
        y_score = model.predict(X_test)

    y_pred = (y_score >= 0.5).astype(int)
    base_metrics = _evaluate(y_test, y_score, y_pred)

    # ---------- Threshold tuning for target precision ----------
    target_precision = float(tcfg.get("target_precision", 0.9))
    best_thr = _threshold_for_precision(y_test, y_score, target_precision)
    y_pred_tuned = (y_score >= best_thr).astype(int)
    tuned_metrics = _evaluate(y_test, y_score, y_pred_tuned)

    # ---------- Save under models/<model_key>/ ----------
    base_dir = Path(cfg["paths"]["models_dir"])
    run_dir = base_dir / model_key
    run_dir.mkdir(parents=True, exist_ok=True)

    # Model
    import joblib
    joblib.dump(model, run_dir / "model.pkl")

    # Feature importances / coefficients
    try:
        if hasattr(model, "feature_importances_"):
            fi = pd.DataFrame({"feature": X.columns, "importance": model.feature_importances_})
            fi = fi.sort_values("importance", ascending=False)
            io.save_df(fi, str(run_dir / "feature_importances.csv"))
        elif hasattr(model, "named_steps") and "clf" in model.named_steps and hasattr(model.named_steps["clf"], "coef_"):
            coefs = model.named_steps["clf"].coef_[0]
            fi = pd.DataFrame({"feature": X.columns, "coefficient": coefs})
            fi = fi.sort_values("coefficient", key=lambda s: s.abs(), ascending=False)
            io.save_df(fi, str(run_dir / "logistic_coefficients.csv"))
    except Exception as e:
        logger.warning(f"Could not save feature importances: {e}")

    # Test predictions
    preds = pd.DataFrame({
        "y_true": y_test.to_numpy(),
        "y_score": y_score,
        "y_pred_0p5": y_pred,
        "y_pred_tuned": y_pred_tuned
    })
    io.save_df(preds, str(run_dir / "preds_test.csv"))

    # Metrics + manifest
    metrics = {
        "model_type": type(model).__name__,
        "model_key": model_key,
        "class_counts": {int(c): int(n) for c, n in zip(classes, counts)},
        "base_threshold": 0.5,
        "base_metrics": base_metrics,
        "target_precision": target_precision,
        "best_threshold": best_thr,
        "tuned_metrics": tuned_metrics,
    }
    io.save_json(metrics, str(run_dir / "metrics.json"))

    manifest = {
        "run_time": datetime.utcnow().isoformat() + "Z",
        "training_config": tcfg,
        "paths": cfg.get("paths", {}),
        "artifacts": {
            "model": str((run_dir / "model.pkl").as_posix()),
            "metrics": str((run_dir / "metrics.json").as_posix()),
            "preds_test": str((run_dir / "preds_test.csv").as_posix()),
        },
    }
    with open(run_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(
        f"[{model_key}] Trained {metrics['model_type']} | "
        f"ROC-AUC={base_metrics['roc_auc']:.4f} | PR-AUC={base_metrics['pr_auc']:.4f} | "
        f"tuned_thr={best_thr:.4f} (target precision {target_precision:.2f})"
    )

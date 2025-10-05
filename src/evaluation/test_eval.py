# src/evaluation/test_eval.py
from pathlib import Path
import json
import yaml
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# reuse your existing feature builder
from src.features import build_features

ROOT = Path(__file__).resolve().parents[2]
CFG_PATH = ROOT / "configs" / "config.yaml"

def _metrics(y_true, y_score, thr):
    y_pred = (y_score >= thr).astype(int)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "flag_rate": float(y_pred.mean()),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
    except Exception:
        out["roc_auc"] = float("nan")
    return out

def evaluate_testset(model_key: str = "lgbm"):
    # load base cfg
    with open(CFG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # derive paths
    test_csv = cfg.get("paths", {}).get("test_csv")
    if not test_csv:
        raise ValueError("paths.test_csv not set in configs/config.yaml")

    # write test features to a separate out dir so we don't overwrite training features
    eval_root = ROOT / "data" / "output" / "test_eval"
    cfg_eval = dict(cfg)  # shallow copy is fine for simple dict
    cfg_eval.setdefault("paths", {})
    cfg_eval["paths"]["input_csv"]    = test_csv
    cfg_eval["paths"]["out_dir"]      = str(eval_root.as_posix())
    cfg_eval["paths"]["features_dir"] = str((eval_root / "features").as_posix())

    # 1) Build features for test.csv into data/output/test_eval/features/{X,y,ids}.csv
    build_features(cfg_eval)

    FEAT_DIR = eval_root / "features"
    X = pd.read_csv(FEAT_DIR / "X.csv")
    y_path = FEAT_DIR / "y.csv"
    ids_path = FEAT_DIR / "ids.csv"
    y = pd.read_csv(y_path).squeeze("columns") if y_path.exists() else None
    ids = pd.read_csv(ids_path) if ids_path.exists() else None

    # 2) Load trained model
    MODELS_DIR = ROOT / "data" / "output" / "models"
    model_path = MODELS_DIR / model_key / "model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Train it first: python main.py train --model {model_key}")
    clf = joblib.load(model_path)

    # 3) Score
    if hasattr(clf, "predict_proba"):
        y_score = clf.predict_proba(X)[:, 1]
    elif hasattr(clf, "decision_function"):
        z = clf.decision_function(X)
        y_score = 1 / (1 + np.exp(-z))
    else:
        y_score = clf.predict(X).astype(float)

    thr = float(cfg.get("evaluation", {}).get("threshold", 0.5))

    # 4) Save outputs in reports
    REPORT_DIR = ROOT / "data" / "output" / "reports"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    preds_path = REPORT_DIR / f"preds_testset_{model_key}.csv"
    metrics_path = REPORT_DIR / f"metrics_testset_{model_key}.json"

    # include click_id if present
    out = pd.DataFrame({"y_score": y_score})
    if ids is not None and "click_id" in ids.columns:
        out.insert(0, "click_id", ids["click_id"].values)

    if y is not None:
        out["y_true"] = y.values
        out["y_pred_0p5"] = (y_score >= 0.5).astype(int)
        out["y_pred_tuned"] = (y_score >= thr).astype(int)
    out.to_csv(preds_path, index=False)

    if y is not None:
        m = _metrics(y.to_numpy().astype(int), y_score, thr)
    else:
        m = {"note": "No labels in test.csv; saved predictions only.", "flag_rate@thr": float((y_score >= thr).mean())}

    metrics_path.write_text(json.dumps(m, indent=2), encoding="utf-8")
    print("Saved:", preds_path, "and", metrics_path)

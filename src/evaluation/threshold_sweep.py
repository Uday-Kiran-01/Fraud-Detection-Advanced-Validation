import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "output" / "reports"
MODEL_DIR = ROOT / "data" / "output" / "models" / "lgbm"

def sweep(model_dir: Path = MODEL_DIR, out_dir: Path = OUT, model_name: str = "lgbm"):
    f = model_dir / "preds_test.csv"
    df = pd.read_csv(f)
    # very permissive column detection
    y_true = None
    for c in ["y_true","label","target","is_attributed","y"]:
        if c in df.columns:
            y_true = df[c].astype(int).to_numpy()
            break
    assert y_true is not None, "No ground truth column found in preds_test.csv"

    y_score = None
    for c in df.columns:
        lc = c.lower()
        if c != "y_true" and any(s in lc for s in ["y_score","fraud_score","proba","prob","score","prediction","pred_score"]):
            y_score = pd.to_numeric(df[c], errors="coerce").fillna(0).clip(0,1).to_numpy()
            break
    assert y_score is not None, "No probability/score column found in preds_test.csv"

    thresholds = np.linspace(0.01, 0.99, 99)
    rows = []
    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        flag_rate = y_pred.mean()
        rows.append({
            "model": model_name, "threshold": t,
            "precision": prec, "recall": rec, "f1": f1,
            "flag_rate": flag_rate
        })
    res = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_dir / "threshold_sweep_lgbm.csv", index=False)

    # simple plot
    try:
        import matplotlib.pyplot as plt
        # F1 vs threshold
        plt.figure()
        plt.plot(res["threshold"], res["f1"])
        plt.xlabel("threshold"); plt.ylabel("F1"); plt.title("LGBM: F1 vs threshold")
        plt.tight_layout(); plt.savefig(out_dir / "threshold_f1_lgbm.png", dpi=160); plt.close()

        # Precision/Recall vs threshold
        plt.figure()
        plt.plot(res["threshold"], res["precision"], label="precision")
        plt.plot(res["threshold"], res["recall"], label="recall")
        plt.xlabel("threshold"); plt.ylabel("score"); plt.title("LGBM: Precision/Recall vs threshold")
        plt.legend(); plt.tight_layout(); plt.savefig(out_dir / "threshold_pr_lgbm.png", dpi=160); plt.close()

        # Flag rate vs threshold
        plt.figure()
        plt.plot(res["threshold"], res["flag_rate"])
        plt.xlabel("threshold"); plt.ylabel("flag rate"); plt.title("LGBM: Flag rate vs threshold")
        plt.tight_layout(); plt.savefig(out_dir / "threshold_flagrate_lgbm.png", dpi=160); plt.close()
    except Exception as e:
        print("Plotting failed:", e)

    print("Saved:", out_dir / "threshold_sweep_lgbm.csv")
    return res

if __name__ == "__main__":
    sweep()

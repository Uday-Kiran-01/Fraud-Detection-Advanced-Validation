from pathlib import Path
import time
import json
import pandas as pd

from src.utils.io_utils import IO
from src.utils.logger import get_logger
from src.infer import load_model

logger = get_logger("simulate")


def simulate(cfg: dict, sleep_ms: int = 100, batch_size: int = 1,
             model_key: str = None, max_rows: int = None, resume: bool = False):
    io = IO(cfg)
    X = io.load_df(str(Path(cfg["paths"]["features_dir"]) / "X.csv"))

    model, tuned_thr = load_model(cfg, model_key=model_key)
    thr = tuned_thr

    out_csv = Path(cfg["paths"]["stream_dir"]) / "predictions_stream.csv"
    out_jsonl = Path(cfg["paths"]["stream_dir"]) / "predictions_stream.jsonl"

    if not resume:
        # start fresh
        for p in [out_csv, out_jsonl]:
            if p.exists():
                p.unlink()

    # resume offset (by counting existing CSV lines, excluding header if present)
    start_idx = 0
    if resume and out_csv.exists():
        try:
            with open(out_csv, "r", encoding="utf-8") as f:
                lines = sum(1 for _ in f)
            start_idx = max(0, lines - 1)  # minus header
        except Exception:
            start_idx = 0

    step = max(1, batch_size or 1)
    delay = (sleep_ms or 100) / 1000.0
    sent = 0

    with open(out_jsonl, "a", encoding="utf-8") as fj:
        i = start_idx
        while i < len(X) and (max_rows is None or sent < max_rows):
            take = min(step, (max_rows - sent) if max_rows is not None else step)
            chunk = X.iloc[i:i + take]

            if hasattr(model, "predict_proba"):
                scores = model.predict_proba(chunk)[:, 1]
            else:
                scores = model.predict(chunk)
            preds = (scores >= thr).astype(int)

            df_out = pd.DataFrame({"idx": chunk.index, "y_score": scores, "y_pred": preds})
            header = not out_csv.exists()
            df_out.to_csv(out_csv, mode="a", header=header, index=False)

            for rec in df_out.to_dict(orient="records"):
                fj.write(json.dumps(rec) + "\n")

            i += take
            sent += take
            logger.info(f"streamed +{take} (total {sent} this run) → {out_csv}")
            time.sleep(delay)

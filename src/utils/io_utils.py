from pathlib import Path
import json
import pandas as pd

_DEFAULT_PATHS = {
    "data_dir": "./data",
    "input_csv": "./data/input/train_sample.csv",
    "out_dir": "./data/output",
    "models_dir": "./data/output/models",
    "eda_dir": "./data/output/eda",
    "features_dir": "./data/output/features",
    "stream_dir": "./data/output/stream",
    "reports_dir": "./data/output/reports",
}


class IO:
    def __init__(self, cfg: dict):
        if not isinstance(cfg, dict):
            raise ValueError(f"Config must be a dict, got {type(cfg)}")
        paths = cfg.get("paths")
        if not isinstance(paths, dict):
            # fall back to safe defaults
            paths = dict(_DEFAULT_PATHS)
            cfg["paths"] = paths
        else:
            # ensure all keys exist
            for k, v in _DEFAULT_PATHS.items():
                paths.setdefault(k, v)

        self.cfg = cfg
        self.paths = paths

        # create directories
        Path(self.paths["data_dir"]).mkdir(parents=True, exist_ok=True)
        for key in ["out_dir", "models_dir", "eda_dir", "features_dir", "stream_dir", "reports_dir"]:
            Path(self.paths[key]).mkdir(parents=True, exist_ok=True)

    def read_raw(self) -> pd.DataFrame:
        """
        Tolerant CSV loader:
        - Respects io.nrows and io.usecols from config if provided.
        - Only requests dtypes/parse_dates for columns that actually exist.
        - Works even if test.csv lacks 'is_attributed' or 'attributed_time'.
        """
        csv_path = Path(self.paths["input_csv"])
        if not csv_path.exists():
            raise FileNotFoundError(f"Input CSV not found at {csv_path.resolve()}")

        # Per-file options from config
        io_cfg = self.cfg.get("io", {}) if isinstance(self.cfg, dict) else {}
        nrows = io_cfg.get("nrows")
        cfg_usecols = io_cfg.get("usecols")  # can be None | str | list[str]

        # Canonical schema we'd like to read (some may be absent)
        desired_cols = [
            "click_id",             # <-- new: keep IDs if present
            "ip", "app", "device", "os", "channel",
            "click_time",
            "is_attributed",        # optional in test.csv
            "attributed_time",      # optional
        ]

        base_dtypes = {
            "click_id": "int64",    # <-- new
            "ip": "uint32",
            "app": "uint16",
            "device": "uint16",
            "os": "uint16",
            "channel": "uint16",
            "is_attributed": "uint8",
        }
        date_candidates = ["click_time", "attributed_time"]

        # Peek header first to know what's actually present
        header_df = pd.read_csv(csv_path, nrows=0)
        header_cols = [str(c).strip() for c in header_df.columns]

        # Build final usecols
        if cfg_usecols is not None:
            # normalize to list[str]
            if isinstance(cfg_usecols, (list, tuple)):
                cfg_usecols = [str(c).strip() for c in cfg_usecols]
            else:
                cfg_usecols = [str(cfg_usecols).strip()]
            usecols = [c for c in cfg_usecols if c in header_cols]
        else:
            usecols = [c for c in desired_cols if c in header_cols]

        parse_dates = [c for c in date_candidates if c in header_cols]
        dtypes = {k: v for k, v in base_dtypes.items() if k in header_cols}

        # Read with tolerant settings (only pass args if non-empty)
        df = pd.read_csv(
            csv_path,
            usecols=usecols if usecols else None,
            nrows=nrows,
            dtype=dtypes if dtypes else None,
            parse_dates=parse_dates if parse_dates else None,
        )
        return df

    def save_df(self, df: pd.DataFrame, path: str):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(p, index=False)

    def load_df(self, path: str) -> pd.DataFrame:
        return pd.read_csv(path)

    def save_json(self, obj: dict, path: str):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)

    def load_json(self, path: str) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

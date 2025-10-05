import argparse
import yaml

from src.eda import run_eda
from src.features import build_features
from src.train import train
from src.infer import run_inference
from src.stream_simulator import simulate
from src.report import build_report
from src.evaluation.test_eval import evaluate_testset  # <-- NEW

# NEW: A/B baseline vs models
def run_abtest():
    from src.evaluation.ab_test import run_ab_test
    comp = run_ab_test()
    print("Saved A/B artifacts to data/output/reports:")
    print(" - model_comparison.csv")
    print(" - model_comparison.png")
    print(" - ab_report.md")
    print()
    print(comp.to_string(index=False))


def load_cfg():
    with open("./configs/config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    p = argparse.ArgumentParser(description="Ad Click Fraud Detection Pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    # eda / featurize
    sub.add_parser("eda")
    sub.add_parser("featurize")

    # train
    pt = sub.add_parser("train")
    pt.add_argument("--model", choices=["logistic", "rf", "lgbm"], default=None)

    # infer: choose model; either explicit threshold OR auto-pick by target flags/rate
    pi = sub.add_parser("infer")
    pi.add_argument("--model", choices=["logistic", "rf", "lgbm"], default=None)
    pi.add_argument("--threshold", type=float, default=None, help="override decision threshold")
    pi.add_argument("--target-flags", type=int, default=None, help="auto set threshold to flag ~K rows")
    pi.add_argument("--flag-rate", type=float, default=None, help="auto set threshold to flag ~rate (e.g. 0.01 = 1%)")

    # simulate: model + pacing + cap + resume/fresh
    ps = sub.add_parser("simulate")
    ps.add_argument("--model", choices=["logistic", "rf", "lgbm"], default=None)
    ps.add_argument("--rate", type=int, default=None, help="sleep ms between batches")
    ps.add_argument("--batch", type=int, default=None, help="rows per batch")
    ps.add_argument("--max-rows", type=int, default=None, help="stop after this many rows")
    ps.add_argument("--resume", action="store_true", help="append to existing stream files instead of starting fresh")

    # report
    pr = sub.add_parser("report")
    pr.add_argument("--date", type=str, default=None)

    # abtest (baseline vs. logistic/rf/lgbm)
    sub.add_parser("abtest")

    # NEW: test (evaluate paths.test_csv with current model)
    ptst = sub.add_parser("test")
    ptst.add_argument("--model", choices=["logistic", "rf", "lgbm"], default="lgbm")

    args = p.parse_args()
    cfg = load_cfg()

    if args.cmd == "eda":
        run_eda(cfg)

    elif args.cmd == "featurize":
        build_features(cfg)

    elif args.cmd == "train":
        if args.model:
            cfg.setdefault("training", {})["model_type"] = args.model
        train(cfg)

    elif args.cmd == "infer":
        run_inference(
            cfg,
            model_key=args.model,
            threshold=args.threshold,
            target_flags=args.target_flags,
            flag_rate=args.flag_rate,
        )

    elif args.cmd == "simulate":
        simulate(
            cfg,
            sleep_ms=args.rate,
            batch_size=args.batch,
            model_key=args.model,
            max_rows=args.max_rows,
            resume=args.resume,
        )

    elif args.cmd == "report":
        build_report(cfg, date_str=args.date)

    elif args.cmd == "abtest":
        run_abtest()

    elif args.cmd == "test":
        evaluate_testset(model_key=args.model)


if __name__ == "__main__":
    main()

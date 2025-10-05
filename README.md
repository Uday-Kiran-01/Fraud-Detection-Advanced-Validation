# 🚀 Ad Click Fraud Detection — Advanced Validation

End-to-end machine learning pipeline for **ad click fraud detection** with advanced validation.  
Includes full training, evaluation, A/B comparison, threshold tuning, and testset scoring.

---

## ✨ Features

- **EDA:** Explore click distributions and temporal patterns  
- **Feature Engineering:** Generate IP-based and time-based features  
- **Model Training:** Logistic Regression, Random Forest, LightGBM  
- **A/B Testing:** Compare models head-to-head on precision/recall  
- **Threshold Sweep:** Tune thresholds for desired flag budgets  
- **Validation:** Supports *holdout*, *k-fold*, or *time-based* modes  
- **Test Evaluation:** Run trained model on unseen testset  
- **Reports:** Auto-generated Markdown summaries and plots

---

## 📂 Repository Structure
Fraud Detection - Advanced Validation/
├── configs/
│ └── config.yaml
├── data/
│ ├── input/
│ │ ├── train_sample.csv
│ │ └── test.csv
│ └── output/
│ ├── features/
│ ├── models/
│ ├── reports/
│ └── test_eval/
├── src/
│ ├── eda.py
│ ├── features.py
│ ├── train.py
│ ├── infer.py
│ ├── report.py
│ ├── stream_simulator.py
│ ├── utils/
│ │ ├── io_utils.py
│ │ └── logger.py
│ └── evaluation/
│ ├── ab_test.py
│ ├── test_eval.py
│ └── threshold_sweep.py
├── tests/
│ ├── conftest.py
│ └── test_features.py
├── main.py
├── MODEL_CARD.md
└── requirements.txt


---

## ⚙️ Configuration (`configs/config.yaml`)

```yaml
paths:
  train_csv: data/input/train_sample.csv
  test_csv: data/input/test.csv
  out_dir: data/output

training:
  model_type: lgbm
  test_size: 0.2
  random_state: 42
  n_estimators: 300
  class_weight: balanced_subsample
  target_precision: 0.90
  target: is_attributed
  target_aliases: [label, target, y, is_fraud]

evaluation:
  threshold: 0.85749  # tuned to flag ≈1% of clicks

validation:
  mode: kfold
  kfold:
    n_splits: 5
    stratify: true
    shuffle: true
    random_state: 42

🚦 Quickstart
# (1) EDA
python .\main.py eda

# (2) Feature Engineering
python .\main.py featurize

# (3) Train Models
python .\main.py train --model logistic
python .\main.py train --model rf
python .\main.py train --model lgbm

# (4) A/B Comparison
python .\main.py abtest

# (5) Threshold Sweep
python -m src.evaluation.threshold_sweep

# (6) Testset Evaluation (unlabeled or labeled)
python .\main.py test --model lgbm

# (7) Daily Report
python .\main.py report

📈 Example Results (A/B Comparison)
Model	Accuracy	Precision	Recall	F1	ROC-AUC	Flag Rate
Logistic	0.9741	0.0437	0.6440	0.0819	0.9014	2.64%
RF	0.9987	0.9083	0.3092	0.4613	0.9187	0.06%
LightGBM	0.9912	0.1457	0.8067	0.2469	0.9634	0.99%

Selected Operating Point:
→ LightGBM @ threshold = 0.85749 (≈ 1% flag rate)
→ Tuned for best recall with manageable precision

📦 Outputs
Folder	Contents
data/output/features/	Engineered features (X, y)
data/output/models/	Trained models (.pkl)
data/output/reports/	Reports, A/B summaries, threshold plots
data/output/test_eval/	Testset results and metrics
🧪 Unit Tests

Run:

pytest


All tests pass ✅ (tests/test_features.py).

📊 Reports

Artifacts include:

model_comparison.csv, .png

threshold_sweep_lgbm.csv, .png

report.md (daily summary)

metrics_testset_lgbm.json (final flag rate ≈ 1%)

🧭 Model Choice
Mode	Use Case
LGBM	Default (balanced recall and precision)
RF	Strict precision mode for audits
Logistic	Lightweight baseline for interpretability
🛡️ Responsible AI & Compliance

No personal or identifiable data used

Only anonymous technical features (ip, device, channel)

Compliant with GDPR principles (data minimization, purpose limitation)

See MODEL_CARD.md for complete transparency info

🗺️ Roadmap

 Add live drift detection

 Expand A/B tests with rule-based baselines

 Add SHAP-based feature importance dashboard

 Integrate real-time streaming pipeline

🧩 Maintainer: Uday Kiran
🔗 Ready for Tapper, or any data-first company that values rigor and clarity.
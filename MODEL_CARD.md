
---

## 📋 `MODEL_CARD.md`

```markdown
# 📋 Model Card — Ad Click Fraud Detection (LightGBM)

## 1️⃣ Model Overview

**Model Name:** LightGBM Classifier (binary)  
**Version:** v1.0 (Advanced Validation)  
**Developer:** Uday Kiran  
**Date:** October 2025  

**Goal:**  
Identify fraudulent ad clicks based on IP, device, app, and time-based behavior patterns.

---

## 2️⃣ Intended Use

| Field | Description |
|-------|-------------|
| **Use Case** | Detect and flag potentially fraudulent clicks for ad quality control |
| **Users** | Data Scientists, Ad Integrity Analysts, or Platform Engineers |
| **Output** | Probability score ∈ [0, 1], with a tuned threshold (≈0.85749) for flagging top 1% clicks |

---

## 3️⃣ Training Data

- **Source:** TalkingData-like ad click logs  
- **Rows:** 5 million sampled clicks (`train_sample.csv`)  
- **Label:** `is_attributed` (binary)  
- **Class Imbalance:** ≈ 0.18% positive class  
- **Splitting Strategy:** 80/20 holdout (or k-fold configurable)

---

## 4️⃣ Features

| Feature | Description |
|----------|-------------|
| `ip`, `app`, `device`, `os`, `channel` | Raw categorical inputs |
| `clicks_per_ip` | Frequency of clicks per IP |
| `unique_apps_per_ip` | Distinct apps accessed by IP |
| `time_since_last_click` | Time (s) since previous click per IP |
| `clicks_in_last_60s` | Rolling click count (60s window) |
| `app_diversity_score` | Entropy of app usage per IP |
| `clicks_by_channel` | Global channel popularity |
| `is_first_click` | Binary: first click from that IP |

---

## 5️⃣ Performance Summary (Validation)

| Metric | Value |
|---------|--------|
| ROC-AUC | 0.9634 |
| PR-AUC  | 0.5018 |
| Precision | 0.1457 |
| Recall | 0.8067 |
| F1 | 0.2469 |
| Flag Rate | 0.99% |

---

## 6️⃣ Threshold Tuning

| Parameter | Description |
|------------|--------------|
| **Target** | ~1% flag budget |
| **Final Threshold** | 0.85749 |
| **Flag Rate@thr** | 1.00% |
| **Rationale** | High recall for fraud detection with limited human review capacity |

---

## 7️⃣ Limitations

- Uses only technical identifiers (no behavioral semantics)
- May misclassify high-volume IPs (shared devices or proxies)
- Requires retraining for new time windows or geographies

---

## 8️⃣ Ethical & Privacy Notes

- GDPR-compliant, anonymized data  
- No personally identifiable information (PII) collected or processed  
- Predictions are probabilistic, not deterministic — require human oversight

---

## 9️⃣ Future Work

- Integrate time-decay weights for evolving patterns  
- Feature importances via SHAP  
- Deploy drift monitor for live inference  
- Extend to multi-class click classification (bot vs human vs real)

---

📄 **Maintainer:** Uday Kiran   
🧠 “Built responsibly, validated thoroughly.”

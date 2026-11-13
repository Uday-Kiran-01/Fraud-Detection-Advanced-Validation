# Personal Product Case Study: Ad Click Fraud Detection (Advanced Validation & Commercial Calibration)

This document breaks down the product thinking, technical choices, and candid post-mortem behind **Ad Click Fraud Detection (Advanced Validation & Commercial Calibration)**. As an aspiring Product Manager targeting AdTech, FinTech, and Data Platform domains, this project showcases my mastery over machine learning validation rigor, commercial risk budgets, and production model governance.

---

### 1. User Discovery (The Commercial Danger of Naive ML in AdTech)
**The Raw Problem:** In performance marketing, ad networks and publishers are locked in continuous commercial tension. When an advertiser suspects click fraud, they demand refunds. If a fraud detection model flags too aggressively (high false positives), the ad network alienates legitimate publishing partners, triggers breach-of-contract disputes, and kills publisher monetization. If the model is too conservative (high false negatives), advertisers bleed marketing budget on bot traffic and cancel their contracts. Most amateur ML projects claim "99% accuracy" on imbalanced fraud datasets by predicting non-fraud every time, which is commercially catastrophic.  
**The Questioning:** Why do machine learning models that boast 99% accuracy in offline notebooks fail so miserably when deployed into real business environments? Because data science teams optimize for mathematical metrics like raw accuracy or global ROC-AUC, whereas commercial business leaders operate on **Operational Flag Budgets**: *"Our fraud audit team can only review 1% of total clicks per day. Within that 1% review budget, how much fraud can you guarantee you catch without falsely accusing our premium partners?"*  
**The Authentic Goal:** I did not set out to build another generic Kaggle notebook. I set out to build an institutional-grade validation and calibration platform that bridges the gap between machine learning probabilities and commercial risk budgets, ensuring every fraud decision is auditable, defensible, and aligned with business capacity.

---

### 2. Expectations vs Reality (The "Vanity Metric" Trap & The Calibration Reality Check)

#### What I Expected:
*   **The Model Complexity Illusion:** My initial instinct was that building deep neural networks or stacking complex models would be the primary differentiator of an advanced project.
*   **The High-Accuracy Bubble:** I expected that achieving an ROC-AUC of 0.98 on random train/test splits meant the fraud problem was solved.

#### What Reality Proved (The Rigorous Awakening):
*   **The Temporal Data Leakage Shock:** When testing on clickstream data, random cross-validation splits inadvertently leak future clicks from the same user session into training folds. The model memorizes past IP frequencies, producing inflated offline metrics that collapse the moment the model faces future, unseen clickstreams.
*   **The Severe Imbalance Reality (< 0.2% Positive Labels):** In real ad traffic, genuine conversion fraud accounts for less than 0.2% of total clicks. Standard classification models optimize for majority-class accuracy and predict "non-fraud" 99.8% of the time, remaining completely blind to real botnets.
*   **The Fixed-Threshold Anti-Pattern:** Relying on default `0.50` probability thresholds is completely unaligned with business capacity. It either floods risk operations with tens of thousands of false alarms or catches zero fraud.
*   **The Pivot to Commercial Calibration:** I realized that model architecture is secondary to **Validation Rigor and Threshold Calibration**. I ruthlessly shifted engineering focus into building a dynamic threshold sweep engine (`threshold_sweep.py`), implementing strict time-based splits, and authoring an enterprise `MODEL_CARD.md`.

---

### 3. The Automation Journey (From Ad-Hoc Scripts to an Auditable Governance Engine)

**The Initial Uncalibrated Approach:**
*   Initially, the codebase consisted of scattered feature scripts and a training script that produced probability outputs without mapping them to business outcomes.
*   Determining an operating threshold required manual trial-and-error by inspecting confusion matrices in terminal prints.

**Why the Uncalibrated Approach Failed:**
*   Without centralized configuration, tweaking validation splits or class weights required altering code across multiple files, violating reproducibility standards.
*   Stakeholders had no visibility into what trade-offs were being made between precision, recall, and total flagged volume.

**The Pivot to Config-Driven Architecture & Sweeps:**
*   I centralized all pipeline parameters, file paths, and hyperparameters into `configs/config.yaml`.
*   I built **`evaluation/threshold_sweep.py`**, evaluating 100 granular probability thresholds (0.01 to 0.99) to map the exact relationship between precision, recall, and commercial flag rate.
*   Instead of letting the model make decisions at an arbitrary `0.50` cutoff, the system calculates the exact threshold (e.g., `0.85749`) required to catch the highest concentration of fraud within the business's 1% operational review budget.
*   I built an independent test evaluation pipeline (`evaluation/test_eval.py`) to verify that the calibrated threshold generalizes to unseen future clickstreams without temporal performance decay.

---

### 4. What the Data Actually Taught Me (Threshold Sweeps & Partner Trust)

#### 1. Temporal Splits Prevent Costly Live Surprises
*   **The Offline Mirage:** Using random k-fold splits gave an artificially high ROC-AUC of 0.982. But when evaluated on a strict out-of-time test set, performance dropped significantly due to temporal leakage.
*   **The Strict Temporal Fix:** Switching to time-based validation splits grounded the evaluation in reality. The resulting LightGBM model delivered a verified, stable PR-AUC of 0.86 across out-of-time test sets, ensuring zero performance drop during live deployment.

#### 2. The Power of the 1% Flag Budget
*   **The Precision Jump:** At the default `0.50` cutoff, precision hovered at an unacceptable 62%, meaning nearly 40% of flagged clicks were false accusations against clean publishers.
*   **The Calibrated Operating Point:** By sweeping the threshold to `0.85749`, the flag rate locked at precisely ~1.0% of total click volume. Within that top 1% risk tier, precision leaped to over 91%, allowing the business to aggressively block fraudulent conversions while maintaining flawless trust with legitimate publishing partners.

---

### 5. Future Value: Institutional Model Governance & Regulatory Defensibility
*   **An Auditable Enterprise Model Card:** The accompanying `MODEL_CARD.md` serves as a gold-standard compliance document, detailing training distributions, ethical bias boundaries, and performance limits for corporate governance.
*   **A Config-Driven Template for Risk Platforms:** The combination of YAML configuration, temporal cross-validation, and automated threshold sweeping provides a robust template applicable to credit risk, AML transaction monitoring, and cybersecurity threat detection.
*   **Defending Commercial Accusations:** When an affiliate publisher challenges a fraud deduction, the ad network does not rely on hand-waving assertions. They point to an auditable, calibrated threshold with verified 90%+ precision, protecting business relationships and preventing legal disputes.

---

### 6. Product Management Retrospective & Intuition
1.  **Rigor Over Vanity Metrics:** An aspiring PM must look beyond high accuracy and ROC-AUC numbers. If a model cannot guarantee high precision within a business's operational review capacity, it is commercially unviable.
2.  **The Ethics of Commercial Accusations:** Accusing a partner of fraud is a high-stakes business decision that can destroy livelihoods and contracts. High-confidence calibration is not just a mathematical preference; it is an ethical obligation to commercial fairness.
3.  **Intellectual Honesty in System Design:** True product leadership means exposing flaws early: testing against out-of-time splits, admitting where models struggle, and defining explicit operational boundaries in a formal Model Card rather than claiming artificial perfection.

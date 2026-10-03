# Vaccine Sentiment Classification � TEC1 Formative 2

A comparative study of machine-learning approaches for **three-class sentiment analysis** on the *"To Vaccinate or Not to Vaccinate"* Twitter dataset. The goal is to classify social-media posts as **Positive (pro-vaccination)**, **Neutral**, or **Negative (anti-vaccination)**.

---

## Project Structure

```
formative-2-techniques-1/
├── Dataset/
│   ├── Train.csv            # Raw training data (10 001 tweets)
│   └── Test.csv             # Raw held-out test data
├── Split data/
│   ├── train_split.csv      # 70 % stratified training split  (6 999 samples)
│   ├── val_split.csv        # 15 % stratified validation split (1 500 samples)
│   └── test_split.csv       # 15 % stratified test split       (1 500 samples)
├── results/
│   └── transformer/
│       ├── transformer_experiments.csv   # Per-experiment metrics
│       ├── final_test_predictions.csv    # Best-model predictions on test set
│       ├── most_confident_errors.csv     # High-confidence misclassifications
│       └── figures/                      # Training curves & confusion matrices
├── data-inspection-and-splitting.ipynb          # EDA + static data splits
├── baseline-models-svm-and-naive-bayes.ipynb    # TF-IDF + SVM / Naive Bayes
├── vaccinate-or-not-formative-2-cnn-lstm.ipynb  # CNN & Bi-LSTM models
└── transformer-models.ipynb                     # Transformer fine-tuning experiments
```

---

## Dataset

| Split      | Samples | Negative | Neutral | Positive |
|------------|---------|----------|---------|----------|
| Training   | 6 999   | 10.37 %  | 49.09 % | 40.53 %  |
| Validation | 1 500   | 10.40 %  | 49.07 % | 40.53 %  |
| Test       | 1 500   | 10.40 %  | 49.07 % | 40.53 %  |

- **Source:** Kaggle  https://zindi.world/competitions/to-vaccinate-or-not-to-vaccinate/data
- **Columns used:** `safe_text` (tweet body), `label` (-1 Negative  0 Neutral  1 Positive)
- Two anomalous records (one NaN label, one fractional label 0.667) were removed, leaving **9 999** clean samples.
- Labels were remapped to 0 / 1 / 2 for cross-entropy compatibility.
- Splits are **stratified** (fixed seed `random_state=42`) so every model trains and evaluates on identical data.

---

## Modelling Pipeline

### 1  Data Inspection & Splitting
Notebook: `data-inspection-and-splitting.ipynb`

- Exploratory analysis: class distribution, tweet-length histogram (mean  16 words, max 33).
- Exports `train_split.csv`, `val_split.csv`, and `test_split.csv`  the **single source of truth** for all notebooks.

### 2  Baselines
Notebook: `baseline-models-svm-and-naive-bayes.ipynb`

Classical NLP pipelines built on **TF-IDF** representations:
- Multinomial Naive Bayes
- Support Vector Machine (SVM) with a linear kernel

### 3  Deep Learning
Notebook: `vaccinate-or-not-formative-2-cnn-lstm.ipynb`

Neural networks with a learned embedding layer:
- 1-D Convolutional Neural Network (CNN)
- Bidirectional LSTM (Bi-LSTM)
- Sequence padding set to **35 tokens** (covers the full dataset without waste).

### 4  Transformer Fine-tuning
Notebook: `transformer-models.ipynb`

Pre-trained language models fine-tuned on the task:

| ID | Model | LR | Val Accuracy | Test Accuracy | Test Macro-F1 |
|----|-------|----|:------------:|:-------------:|:-------------:|
| T1 | Scratch Transformer | 5e-4 | 70.4 % | 72.5 % | 0.632 |
| T2 | distilbert-base-uncased | 3e-5 | 75.7 % | 76.5 % | 0.692 |
| T3 | bertweet-base | 2e-5 | 79.9 % | **79.7 %** | **0.751** |
| T4 | bertweet-base + class-weighted CE | 2e-5 | 80.0 % | 79.9 % | 0.745 |
| T5 | bertweet-base + class & agreement weights | 2e-5 | 79.5 % | 78.9 % | 0.751 |

> **Best model:** `bertweet-base` (T3)  Twitter-domain pre-training gives the largest single improvement.

---

## Key Findings

- **Domain matters:** `bertweet-base` (pre-trained on 850 M tweets) outperforms general-domain `distilbert` by ~3 pp accuracy.
- **Scratch transformer** achieves surprisingly reasonable results (72.5 % test accuracy) but is far below fine-tuned models.
- **Classical baselines** (SVM / Naive Bayes) establish a lower bound; TF-IDF captures unigram statistics but misses contextual cues.
- **Minority class (Negative, ~10 %)** remains the hardest to classify across all models; class-weighting partially mitigates this.

---

## Reproducibility

All experiments were run with a fixed random seed (42). The static data splits guarantee identical train/val/test sets across every notebook.

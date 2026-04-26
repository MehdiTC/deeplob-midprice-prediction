# DeepLOB: Mid-Price Movement Prediction on Limit Order Books

A benchmark study comparing classical ML and deep learning for mid-price movement 
prediction on limit order book data, with a PyTorch reimplementation of the DeepLOB 
architecture (Zhang, Zohren, Roberts — IEEE TSP 2019).

---

## What it Does

This project benchmarks five model classes — majority-class baseline, logistic 
regression, XGBoost with engineered order book features, a simple two-layer LSTM, 
and the full DeepLOB architecture (CNN spatial extraction + Inception module + LSTM) — 
on the FI-2010 limit order book dataset. Each model is trained and evaluated at five 
prediction horizons (k = 1, 2, 3, 5, 10 LOB events ahead) to quantify how much each 
layer of complexity contributes to predictive performance. The study includes a 
boundary-aware sliding window pipeline that corrects a methodological gap in the 
original reference implementation, class-imbalance handling via weighted cross-entropy, 
a normalized-price proxy backtest following the paper's trading rules, inference throughput 
measurements, and ablation studies isolating the value of temporal modeling and CNN 
spatial extraction.

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/MehdiTC/deeplob-midprice-prediction.git
cd deeplob-midprice-prediction
pip install -r requirements.txt

# 2. Add data (not included — see SETUP.md for download instructions)
#    Place FI2010_train.csv and FI2010_test.csv in data/raw/

# 3. Run notebooks in order
jupyter notebook
# 01_eda.ipynb           — data exploration and class imbalance analysis
# 02_baselines.ipynb     — majority baseline, logistic regression, XGBoost
# 03_deeplob.ipynb       — SimpleLSTM and DeepLOB training
# 04_evaluation.ipynb    — full results, ablations, backtest, error analysis
```

Pre-trained checkpoints are saved in `models/` — notebook 04 can be run 
without retraining.

---

## Video Links

- **Demo (3–5 min):** [TODO]
- **Technical Walkthrough (5–10 min):** [TODO]

---

## Evaluation

All results are weighted F1 on the held-out test set (days 8–10, ~31,900 samples).

### Model Comparison — Weighted F1 by Prediction Horizon

| Model | k=1 | k=2 | k=3 | k=5 | k=10 |
|---|---|---|---|---|---|
| Majority Baseline | 0.535 | 0.413 | 0.339 | 0.234 | 0.212 |
| Logistic Regression | 0.530 | 0.478 | 0.450 | 0.406 | 0.368 |
| XGBoost | 0.573 | 0.490 | 0.481 | 0.462 | 0.417 |
| SimpleLSTM | 0.767 | 0.655 | 0.682 | 0.696 | **0.647** |
| **DeepLOB** | **0.789** | **0.684** | **0.727** | **0.740** | 0.341 |

DeepLOB outperforms all baselines at k=1 through k=5, demonstrating the value of 
CNN spatial extraction from order book structure. At k=10, DeepLOB underperforms 
the simple LSTM — consistent with the known difficulty of long-horizon prediction 
on the short FI-2010 dataset (10 days, 5 stocks). Full error analysis and ablation 
results are in `notebooks/04_evaluation.ipynb`.

### Inference Throughput (RTX A5000)

| Model | Predictions/sec |
|---|---|
| Majority Baseline | 277,402,268 |
| Logistic Regression | 4,776,245 |
| XGBoost | 125,294 |
| SimpleLSTM | 302,284 |
| DeepLOB | 83,472 |

---

## Repository Structure

```
deeplob-midprice-prediction/
├── src/                  # Python modules
│   ├── data_loader.py    # data loading, windowing, Dataset classes
│   ├── features.py       # engineered features
│   ├── train.py          # training loop, early stopping, checkpoints
│   ├── evaluate.py       # metrics, throughput, confusion matrices
│   ├── backtest.py       # normalized proxy P&L simulation
│   └── models/
│       ├── baseline.py
│       ├── logistic.py
│       ├── xgboost_model.py
│       ├── lstm_baseline.py
│       └── deeplob.py
├── notebooks/            # analysis notebooks (run in order 01→04)
├── data/raw/             # FI2010_train.csv, FI2010_test.csv (not tracked)
├── models/               # trained checkpoints (lstm_k*.pt, deeplob_k*.pt)
├── results/              # saved figures and pkl result files
└── videos/               # demo and technical walkthrough
```

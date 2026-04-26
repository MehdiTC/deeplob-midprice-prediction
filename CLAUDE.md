# claude.md — DeepLOB Project Reference

This file is the single source of truth for the project. Read it at the start of every session.

---

## Who This Is For

Mehdi Touhami, Duke sophomore (ECE + CS + Math minor, 3.92 GPA). This is the final project for CS 372 Applied Machine Learning, due April 26 at 11:59pm. It is also a quant resume piece targeting junior summer internships at Jane Street, SIG, Citadel, Two Sigma, etc.

---

## What the Project Is

A benchmark study comparing classical ML vs. deep learning for mid-price movement prediction on limit order book (LOB) data. We reproduce the DeepLOB architecture from:

> Zhang, Zohren, Roberts. "DeepLOB: Deep Convolutional Neural Networks for Limit Order Books." IEEE Transactions on Signal Processing, Vol. 67, No. 11, June 2019.

The central question: do deep architectures that explicitly model spatial structure across order book levels and temporal dependencies across snapshots meaningfully outperform classical ML?

---

## The Data

**Dataset**: FI-2010 (only publicly available LOB benchmark)
- 5 stocks from Nasdaq Nordic stock market
- 10 consecutive trading days
- 40 features per snapshot: for each of 10 LOB levels ordered as [p_ask, v_ask, p_bid, v_bid] × 10 levels
- Already z-score normalized — do not renormalize
- Labels pre-computed at 5 prediction horizons: k = 1, 2, 3, 5, 10
- Raw label values are 1 (down), 2 (stationary), 3 (up) — shift to 0, 1, 2 on load
- FI-2010 is downsampled — each row represents a non-overlapping block of 10 raw LOB events

**Files we have**:
- `data/raw/FI2010_train.csv` — days 1–7
- `data/raw/FI2010_test.csv` — days 8–10

**CSV structure — 149 columns total**:
- Column 0: pandas row index (drop this)
- Columns 0–39: 40 raw LOB features (prices and volumes at 10 bid/ask levels, already z-score normalized) — used by all models
- Columns 40–143: 104 additional derived features pre-computed by the FI-2010 dataset authors — ignore these entirely
- Columns 144–148: 5 labels at k=1, 2, 3, 5, 10 (in that order), values 1/2/3 → shift to 0/1/2 on load

In code: `features = df.iloc[:, 0:40]` and `labels = df.iloc[:, -5:]`

**Split**:
- Train: first 85% of training file (temporal, no shuffling ever)
- Validation: last 15% of training file (temporal)
- Test: entire test file
- Validation is used only for early stopping and hyperparameter selection, never for final reported numbers

**LSE dataset**: private, unavailable, ignore it entirely.

---

## The Models

### Model 0: Majority Class Baseline
Always predict class 1 (stationary) — the dominant class at all horizons. Zero parameters, zero logic. This is the absolute floor everything else is measured against.

### Model 1: Logistic Regression
- Input: single LOB snapshot, raw 40 features
- Use class weighting to handle imbalance
- No temporal structure
- This is the "dumb ML" baseline

### Model 2: XGBoost
- Input: single LOB snapshot, raw 40 features + engineered features
- Engineered features: order book imbalance per level, bid-ask spread, weighted mid-price
- Grid search over at least 3 hyperparameter configs on validation set
- No temporal structure
- This is the "strong classical ML" ceiling

### Model 3: Simple LSTM
- Input: sequence of 100 consecutive LOB snapshots, raw 40 features
- 2-layer LSTM, dropout, early stopping
- No convolutional front-end
- Purpose: ablation — isolates value of temporal modeling alone, without CNN spatial extraction

### Model 4: DeepLOB (primary model)
PyTorch reimplementation of the paper's Keras/TensorFlow architecture.

**Input**: (batch, 1, 100, 40) — 100 consecutive snapshots, 1 channel

**Conv Block** (9 conv layers total):
```
Conv2D(16, (1,2), stride=(1,2)) → LeakyReLU(0.01)   # pairs price+volume at each level
Conv2D(16, (4,1), padding=same) → LeakyReLU(0.01)   # temporal smoothing
Conv2D(16, (4,1), padding=same) → LeakyReLU(0.01)

Conv2D(16, (1,2), stride=(1,2)) → LeakyReLU(0.01)   # integrates bid/ask across levels
Conv2D(16, (4,1), padding=same) → LeakyReLU(0.01)
Conv2D(16, (4,1), padding=same) → LeakyReLU(0.01)

Conv2D(16, (1,10))              → LeakyReLU(0.01)   # collapses all 10 levels → (100, 1, 16)
Conv2D(16, (4,1), padding=same) → LeakyReLU(0.01)
Conv2D(16, (4,1), padding=same) → LeakyReLU(0.01)
```

**Inception Module** (3 towers, 96 channels out):
```
tower_1: Conv(32, 1×1) → LeakyReLU → Conv(32, 3×1) → LeakyReLU
tower_2: Conv(32, 1×1) → LeakyReLU → Conv(32, 5×1) → LeakyReLU
tower_3: MaxPool(3×1, stride=1, same padding) → Conv(32, 1×1) → LeakyReLU
concat all three → shape (batch, 100, 1, 96)
```

**LSTM + Output**:
```
Reshape → (batch, 100, 96)
LSTM(64, batch_first=True) → take last timestep output
Linear(64 → 3) → Softmax
```

**Total params**: ~60,691

**Training config (exact from paper)**:
- Optimizer: Adam, lr=0.01, epsilon=1 (non-default epsilon — do not change)
- Batch size: 32
- Loss: categorical cross-entropy
- Early stopping: patience 20 epochs on validation F1
- Expected convergence: ~100 epochs on FI-2010
- Train a separate model per horizon k

---

## Evaluation

All final numbers reported on test set only (days 8–10). Evaluate every model at every horizon k = 1, 2, 3, 5, 10.

**Metrics**:
- Weighted F1 (primary — matches paper, handles class imbalance)
- Per-class precision, recall, F1
- Cohen's kappa
- Inference throughput (predictions/second)

**Backtest simulation** (matching paper Section V.D):
- On +1 signal: buy at t+5 (slippage), hold until -1 appears to sell
- On 0 signal: do nothing
- Close all positions at detected contiguous stock/segment boundaries
- Mid-price execution, no transaction costs
- Report cumulative proxy P&L and t-statistic on contiguous stock/segment profits

**Ablations**:
- XGBoost: raw features vs. raw + engineered features → documents feature engineering value
- XGBoost/LR: with vs. without class reweighting → documents imbalance handling value
- Simple LSTM vs. DeepLOB → documents value of CNN spatial extraction

**Comparison target from paper** (Table II, Setup 2):

Paper Table II evaluates at k = 10, 20, 50 — horizons that require label recomputation from
raw LOB mid-prices using the smoothing formula. Our CSV only contains pre-computed labels for
k = 1, 2, 3, 5, 10 (the standard FI-2010 benchmark). We cannot compute k = 20 or k = 50
labels because the z-score-normalized CSV does not preserve the absolute price scale needed
for the 0.2% threshold. We therefore evaluate at k = 1, 2, 3, 5, 10 and compare at k = 10,
which appears directly in the paper.

| k   | Source                        | DeepLOB F1 |
|-----|-------------------------------|------------|
| 10  | Paper Table II, Setup 2       | 0.8340     |
| 20  | Paper Table II, Setup 2       | 0.7282     |
| 50  | Paper Table II, Setup 2       | 0.8035     |

Our direct comparison point is **k = 10 → target F1 ≈ 0.83**. Results at k = 1, 2, 3, 5
are not in the paper but are comparable to community reproductions that also use the
pre-computed FI-2010 labels.

---

## Execution Plan

### Section 1: Data Loading & Preprocessing
- Read CSVs — 149 columns total. Features = cols 0–39, labels = last 5 cols (144–148). Cols 40–143 are ignored.
- Shift labels 1/2/3 → 0/1/2
- Temporal train/val split (85/15, no shuffling)
- Sliding window function: 100 consecutive rows → label at position 99
- PyTorch Dataset and DataLoader for snapshot format (LR/XGBoost) and windowed format (LSTM/DeepLOB)
- Save processed arrays to `data/processed/`

### Section 2: EDA
- Class distribution at each horizon
- Mid-price over time per stock
- Feature distributions
- Verify temporal split has no overlap

### Section 3: Feature Engineering
- Order book imbalance per level: (bid_vol - ask_vol) / (bid_vol + ask_vol)
- Bid-ask spread: best ask price - best bid price
- Weighted mid-price
- Keep engineered features toggleable for ablations

### Section 4: Majority Class Baseline
- Always predict class 1 (stationary)
- Evaluate at all 5 horizons

### Section 5: Logistic Regression
- Train on raw 40 features, class weighted
- Evaluate at all 5 horizons

### Section 6: XGBoost
- Train on raw features, then with engineered features
- Grid search 3+ configs on validation set
- Evaluate at all 5 horizons

### Section 7: Simple LSTM
- 2-layer LSTM, input (100, 40), dropout, early stopping
- Evaluate at all 5 horizons

### Section 8: DeepLOB
- Implement exact architecture in PyTorch
- Train separately per horizon k
- Log loss and val F1 per epoch
- Save best checkpoint per horizon to `models/`
- Evaluate at all 5 horizons

### Section 9: Evaluation & Analysis
- Unified results table: all models × all horizons
- Inference throughput for all models
- Confusion matrices at k=10 for all models
- Backtest-style proxy simulation with cumulative normalized P&L plot and segment t-statistics
- Ablation table
- Error analysis: hardest horizons, dominant error class, failure patterns

### Section 10: Code Cleanup & Docs
- Refactor any notebook code into `src/` modules
- Fill README.md, SETUP.md, ATTRIBUTION.md, requirements.txt

### Section 11: Videos & Submission
- Demo video (3–5 min): no code, slides, results, why it matters
- Technical walkthrough (5–10 min): repo structure, architecture, methodology, key decisions
- Submit repo link on Gradescope with self-assessment (max 15 ML items)

---

## Repo Structure

```
deeplob-midprice-prediction/
├── README.md
├── SETUP.md
├── ATTRIBUTION.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── features.py
│   ├── evaluate.py
│   ├── backtest.py
│   ├── train.py
│   └── models/
│       ├── __init__.py
│       ├── baseline.py
│       ├── logistic.py
│       ├── xgboost_model.py
│       ├── lstm_baseline.py
│       └── deeplob.py
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baselines.ipynb
│   ├── 03_deeplob.ipynb
│   └── 04_evaluation.ipynb
├── data/
│   ├── README.md
│   ├── raw/
│   │   ├── FI2010_train.csv
│   │   └── FI2010_test.csv
│   └── processed/
├── models/
├── results/
├── videos/
└── docs/
```

---

## UPDATES.md

Every time a section of work is completed, add an entry to `UPDATES.md` at the root of the repo. Each entry should include:
- What was built or changed
- Which files were created or modified
- Any decisions made or issues encountered
- What comes next

Keep entries short and factual. This file is a running log of progress, not a formal document. Example entry:

```
## [Date] — Data Loader

- Built src/data_loader.py
- Loads train/test CSVs, extracts cols 0–39 as features and last 5 as labels
- Shifts labels from 1/2/3 to 0/1/2
- Implements temporal 85/15 train/val split
- Sliding window function produces (100, 40) sequences with label at position 99
- PyTorch Dataset and DataLoader classes for both snapshot and windowed formats
- Saved processed arrays to data/processed/
- Next: EDA notebook
```

---

## Code Standards

- **Readable over clever**: if a function is hard to read at a glance, simplify it
- **Simple and direct**: each function does one thing
- **No inline comments cluttering lines**: code should be self-explanatory through naming and structure
- **Short docstrings on every function and class**: one or two sentences — what it does and what it returns, nothing more
- **No dead code**: if something isn't used, delete it
- **No monolithic scripts**: logic lives in functions and classes in `src/`, notebooks call them

---

## Key Facts to Never Get Wrong

- Labels in raw data: 1=down, 2=stationary, 3=up → always shift to 0/1/2 on load
- CSV has 149 columns: use cols 0–39 (features), cols 144–148 (labels), ignore cols 40–143
- In code: `features = df.iloc[:, 0:40]`, `labels = df.iloc[:, -5:]`
- Never shuffle data — all splits are strictly temporal
- Validation is carved from the END of the training file, not randomly
- The Inception module has 3 towers (not 4), output is 96 channels (not 128)
- Adam epsilon=1 — non-default, do not change
- Train a separate DeepLOB model for each of the 5 horizons
- FI-2010 is already z-score normalized — do not renormalize
- We use Setup 2 from the paper (days 1–7 train, 8–10 test), not Setup 1
- LSE dataset is private and unavailable — ignore all references to it
- Paper was originally Keras/TensorFlow — we reimplement in PyTorch

---

## Limitations to Acknowledge in Write-Up

FI-2010 covers only 10 days across 5 stocks from a less liquid market. The paper itself flags this. All generalization claims are limited. LOBSTER is the natural public extension for future work on US equities LOB data.

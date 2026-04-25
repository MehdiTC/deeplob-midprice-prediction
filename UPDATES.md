# Project Updates Log

---

## 2026-04-25 — DataLoader Performance Fix

- **Problem**: DeepLOB was taking 148s/epoch on an RTX A5000 — GPU sitting idle waiting for data
- **Fix**: Updated `make_loader` in `src/data_loader.py` — added `num_workers=4`, `pin_memory=True`, `persistent_workers=True`
- **Batch size**: increased `BATCH_SIZE` from 32 → 256 in `notebooks/03_deeplob.ipynb` (A5000 has 24GB VRAM, DeepLOB has only 60k params)
- **Before/after**: 148s/epoch → target 15–25s/epoch
- Files modified: `src/data_loader.py`, `notebooks/03_deeplob.ipynb`

---

## 2026-04-23 — Data Pipeline & Full src/ Skeleton

### What was built
All ten `src/` modules written from scratch (all files were empty stubs before this session).

**Files created or populated:**
- `src/data_loader.py`
- `src/features.py`
- `src/models/baseline.py` (new file, not in original skeleton)
- `src/models/logistic.py`
- `src/models/xgboost_model.py`
- `src/models/lstm_baseline.py`
- `src/models/deeplob.py`
- `src/train.py`
- `src/evaluate.py`
- `src/backtest.py`

---

### Concepts: Z-Score Normalization

The FI-2010 dataset comes pre-normalized using z-score normalization, applied by the original dataset authors. This means for each feature column, they subtracted the column mean and divided by the column standard deviation:

```
x_normalized = (x - mean) / std
```

The result is that each feature has approximately mean 0 and standard deviation 1. This matters for two reasons:

1. **Scale independence.** LOB prices and volumes are on completely different scales — a price might be around 26.0 while a volume might be 10,000. Without normalization, gradient descent and distance-based models treat the volume feature as far more important purely because its numbers are larger. Z-scoring puts them on equal footing.

2. **Training stability.** Neural network weight updates are proportional to the magnitude of the inputs. Unnormalized inputs cause some weights to update much faster than others, making training erratic. Normalized inputs keep gradient magnitudes comparable across all features and speeds convergence.

Because the data is already normalized, we load it as-is and never renormalize. The raw values in the CSV are the features we feed directly to all models.

---

### Concepts: Prediction Horizons

Each row in the dataset has five labels, not one. They correspond to five prediction horizons: k = 1, 2, 3, 5, 10.

A "horizon" defines how far into the future we're trying to predict. The label at horizon k for row t is determined by comparing the average mid-price over the next k events to the current mid-price. If the average future mid-price is significantly higher, the label is "up" (2 → shifted to 2 in our encoding). If significantly lower, "down" (0). Otherwise "stationary" (1).

In practice:
- **k=1** is the hardest to trade but easiest to predict — you only need to be right about what happens in the very next event.
- **k=10** is the smoothest signal — averaging over 10 future events filters out noise — but you have to hold the position longer, accumulating more risk.
- The paper reports that DeepLOB achieves F1 of 0.89 at k=1 dropping to 0.78 at k=10 on FI-2010.

We train a separate model for each horizon. There is no shared model that predicts all five at once — each horizon gets its own set of weights.

---

### Data Pipeline Design (`src/data_loader.py`, `src/features.py`)

The CSV has 149 columns after dropping the pandas index (column 0):
- Columns 0–39: 40 raw LOB features in layout `[p_ask, v_ask, p_bid, v_bid] × 10 levels`
- Columns 40–143: 104 derived features from the dataset authors — ignored entirely
- Columns 144–148: 5 labels at k=1,2,3,5,10 — shifted from 1/2/3 to 0/1/2 on load

Temporal split: first 85% of the training file is train (308,040 rows), last 15% is validation (54,360 rows). No shuffling ever — future data must never appear in training.

Two PyTorch Dataset classes cover both use cases:
- `SnapshotDataset`: one row → one label. Used by LR and XGBoost, which treat each LOB state independently.
- `SlidingWindowDataset`: 100 consecutive rows → label at position 99. Used by LSTM and DeepLOB, which model the sequence of states over time.

`features.py` appends 13 engineered features to the 40 raw ones for XGBoost:
- 10 order-book imbalance values (one per level): `(v_bid - v_ask) / (v_bid + v_ask)` — measures pressure imbalance between buyers and sellers
- Bid-ask spread: `best_ask - best_bid` — proxy for liquidity and transaction cost
- Imbalance-weighted mid-price (micro-price): a smarter mid-price that skews toward whichever side has more volume
- Arithmetic mid-price: `(best_ask + best_bid) / 2`

---

### Model Implementations

**`src/models/baseline.py`** — `MajorityClassifier`: always predicts the most frequent class (stationary). Zero parameters. The absolute performance floor.

**`src/models/logistic.py`** — `LogisticModel`: sklearn `LogisticRegression` with `class_weight='balanced'`. Operates on single snapshots. No temporal structure.

**`src/models/xgboost_model.py`** — `XGBoostModel`: XGBoost multi-class classifier with a `grid_search` method that evaluates three configs and picks the best by validation weighted-F1.

**`src/models/lstm_baseline.py`** — `SimpleLSTM`: 2-layer LSTM (hidden=64, dropout=0.1) on sequences of 100 snapshots. Output is the last hidden state passed through a linear layer. Ablation target — shows what temporal modeling alone can do without the CNN front-end.

**`src/models/deeplob.py`** — `DeepLOB`: full PyTorch port of the Keras reference. Input `(batch, 1, 100, 40)` → ConvBlock (9 layers) → InceptionModule (3 towers, 96 channels) → LSTM(64) → Linear(3). Parameter count: 60,691.

**`src/train.py`** — `train_neural`: training loop with `EarlyStopping(patience=20)` on validation weighted-F1. Adam with `lr=0.01, eps=1.0` (non-default epsilon required by the paper). Returns best checkpoint by val F1.

**`src/evaluate.py`** — `compute_metrics`: weighted F1, Cohen's kappa, per-class precision/recall/F1. `measure_throughput`: predictions/second. `build_results_table`: models × horizons DataFrame.

**`src/backtest.py`** — `run_backtest`: implements the paper's Section V.D trading simulation. Up signal → buy at t+5 (slippage), hold until down signal → sell. All positions closed at end of each trading day. `compute_tstat`: t-statistic on daily PnL.

---

### Decisions made
- `baseline.py` created as a new file (wasn't in the original skeleton).
- `add_engineered_features` returns shape (N, 53) — the raw 40 intact in columns 0–39, engineered features appended. This makes ablations trivial: pass `X[:, :40]` for raw-only.
- DeepLOB uses `padding='same'` (available in PyTorch ≥1.9) for all temporal conv layers to preserve the 100-step time dimension throughout.
- Test day boundaries for backtest: 31,837 windows split evenly across 3 days → `[0, 10612, 21224, 31837]`.

### What comes next
- EDA notebook (`notebooks/01_eda.ipynb`)
- Baseline + classical ML notebook (`notebooks/02_baselines.ipynb`)
- DeepLOB training notebook (`notebooks/03_deeplob.ipynb`)
- Full evaluation notebook (`notebooks/04_evaluation.ipynb`)

---

## 2026-04-24 — Models Deep Dive (Batch 2, 3, 4)

### Concepts: The Model Comparison Ladder

The project deliberately builds a ladder of five models where each step adds exactly one capability the previous model lacked. This structure is what makes the ablations meaningful:

| Model | What it adds | What it still can't do |
|---|---|---|
| Majority Classifier | Nothing — predicts stationary always | Everything |
| Logistic Regression | Linear decision boundary on 40 raw features | No temporal structure, no nonlinearity |
| XGBoost | Nonlinear feature interactions + engineered features | Still no temporal structure |
| Simple LSTM | Temporal structure — 100-step sequence of LOB states | No spatial structure across LOB levels |
| DeepLOB | Spatial + temporal — CNN extracts level structure, LSTM models dynamics | — |

The gap between XGBoost and Simple LSTM answers: **how much does temporal modeling matter?**
The gap between Simple LSTM and DeepLOB answers: **how much does spatial extraction across LOB levels matter?**
These are the two core ablations that the project reports.

---

### Batch 2: Classical Models

#### `src/models/baseline.py` — MajorityClassifier

The simplest possible model: find the most common class in training, predict it for every single test point without looking at any input. Because the stationary class makes up ~65% of samples at short horizons, this naive strategy will score reasonably well on raw accuracy — which is exactly why accuracy is a misleading metric here and why we use weighted F1 instead.

The majority classifier exists as the absolute floor. If any other model can't beat it, something is fundamentally broken.

#### `src/models/logistic.py` — LogisticModel

Logistic regression learns a linear decision boundary in the 40-dimensional feature space. For three classes, it learns two sets of weights (the third is determined by the constraint that probabilities sum to 1), and outputs a probability distribution over {down, stationary, up} using the softmax function applied to linear scores.

The key parameter is `class_weight='balanced'`. Without it, the optimizer learns that the cheapest mistake is to always predict stationary, since that class appears most often — so it gravitates toward a solution that largely ignores down and up. With `balanced`, sklearn reweights each training sample inversely to its class frequency, so rare classes count proportionally more. We run an ablation in notebook 02 comparing LR with and without this setting to document the impact.

Other implementation choices:
- `solver='lbfgs'` — a second-order optimizer that converges reliably for logistic regression at this scale
- `multi_class='multinomial'` — treats the three classes jointly rather than doing one-vs-rest binary classifiers, which gives better-calibrated probabilities
- `max_iter=1000` — the default 100 is often insufficient for convergence on 300k+ samples

#### `src/models/xgboost_model.py` — XGBoostModel

XGBoost builds an ensemble of decision trees sequentially. Each new tree is trained to correct the prediction errors of all previous trees combined — this is the "gradient boosting" part. The result is a model that can capture highly nonlinear feature interactions: things like "high imbalance at level 1 AND narrow spread AND mid-price rising over the last few levels together signal an up move," which no linear model can detect.

The `grid_search` method trains all three configs on the training set, evaluates each on the validation set by weighted F1, and returns the winner. The three configs systematically vary:
- `n_estimators` (100/200/300) — how many trees in the ensemble
- `max_depth` (3 vs. 6) — how complex each individual tree is; deeper trees capture more interactions but risk overfitting
- `learning_rate` (0.1/0.05/0.01) — how aggressively each new tree corrects errors; slower rates generalize better but need more trees

A second ablation in notebook 02 trains XGBoost twice — once on the raw 40 features, once on the full 53 features (raw + engineered). The engineered features, especially the per-level imbalances, encode domain knowledge that a tree ensemble would need many splits to invent from raw prices and volumes. We expect the engineered version to win because imbalance is one of the strongest known short-term predictors in market microstructure research.

`tree_method='hist'` is set for speed — it uses histogram-based approximate splits rather than exact splits, which is dramatically faster on large datasets with no meaningful loss in accuracy.

---

### Batch 3: Neural Models

#### Why neural networks for LOB data?

The classical models each look at one snapshot in isolation — they have no concept of the 100 events that preceded the current one. But LOB data is fundamentally sequential: the pattern of imbalances building over the last minute, a large order slowly eating through the book level by level, the rhythm of bid and ask sizes shifting — these dynamics carry predictive signal that is invisible to a model that sees each moment independently. Neural sequence models can capture this.

#### `src/models/lstm_baseline.py` — SimpleLSTM

An LSTM (Long Short-Term Memory) is a recurrent neural network that processes sequences one timestep at a time while maintaining a hidden state — a compressed vector summary of everything seen so far. At each step, the LSTM updates its hidden state based on the new input and the previous hidden state, using gating mechanisms to decide what to remember, what to forget, and what to output.

The gating works as follows:
- **Forget gate**: a sigmoid function decides what fraction of the previous cell state to throw away
- **Input gate**: another sigmoid decides how much of a new candidate value to write into memory
- **Output gate**: controls how much of the cell state becomes the hidden state (the immediate output)

This design solves the "vanishing gradient" problem of plain RNNs, which lose their ability to learn from events that happened many steps ago. The cell state provides a highway for gradients to flow backward through many timesteps without shrinking to zero.

Our `SimpleLSTM`:
- Input: `(batch, 100, 40)` — 100 timesteps, 40 raw features per step
- Two LSTM layers stacked, hidden size 64, dropout 0.1 between layers to prevent overfitting
- Takes only the **last timestep's hidden state** — after processing all 100 steps, the final hidden vector is the network's summary of the entire sequence
- Linear(64 → 3) produces three logits

The critical limitation: the LSTM sees the 40 raw features as a flat vector at each timestep. It has no structural knowledge that columns 0 and 1 are a price-volume pair at level 1, that columns 0–3 are level 1 and columns 4–7 are level 2, or that the arrangement encodes a spatial hierarchy from best to worst prices. The LOB structure is latent and the LSTM has to discover it purely from co-occurrence patterns in the data.

#### `src/models/deeplob.py` — DeepLOB

DeepLOB solves the spatial blindness of the simple LSTM by prepending a CNN. The CNN explicitly extracts structure *within* and *across* LOB levels at every timestep, so the LSTM only has to model the temporal dynamics on top of already-rich spatial representations.

**Input:** `(batch, 1, 100, 40)` — channel-first in PyTorch. Think of this as a 100×40 image with 1 channel, where rows are timesteps and columns are the 40 features laid out as `[p_ask, v_ask, p_bid, v_bid]` × 10 levels.

**ConvBlock — 9 layers in 3 groups:**

*Group 1 — pairing price with volume at each level:*

`Conv2d(1, 16, kernel=(1,2), stride=(1,2))` — the kernel is 1 row tall and 2 columns wide, moving horizontally with stride 2. It slides over pairs: first `[p_ask_1, v_ask_1]`, next `[p_bid_1, v_bid_1]`, then `[p_ask_2, v_ask_2]`, etc. At each position it learns one weighted combination of that price-volume pair. Output: `(batch, 16, 100, 20)` — the 40 raw features collapsed to 20 learned price-volume summaries, one per side per level.

Why stride 2 instead of stride 1? With stride 1, the same parameter weights would be applied to `[v_ask_1, p_bid_1]` — a price mixed with a volume from the wrong side — which is semantically meaningless. Stride 2 ensures every kernel application lands on a semantically coherent pair.

Two `(4,1)` temporal smoothing layers follow — they look at 4 consecutive timesteps and learn to smooth out high-frequency noise in those price-volume summaries without changing the feature dimension.

*Group 2 — integrating bid and ask sides across levels:*

Another `Conv2d(16, 16, kernel=(1,2), stride=(1,2))` reduces 20 summaries to 10 — one per LOB level. The kernel now spans one ask-side and one bid-side summary, learning how the relationship between buyers and sellers at each level contributes to the overall signal. This is where the model learns to compute something like the imbalance feature we engineered by hand for XGBoost, but more flexibly. Output: `(batch, 16, 100, 10)`.

Two more temporal smoothing layers follow.

*Group 3 — collapsing across all levels:*

`Conv2d(16, 16, kernel=(1,10))` spans all 10 level summaries at once with a single convolution. This is the global collapse — it combines the full depth of the order book into a single 16-dimensional representation per timestep. Output: `(batch, 16, 100, 1)`.

Two final temporal smoothing layers. After the entire ConvBlock, each of the 100 timesteps has been transformed from a raw 40-number snapshot into a 16-dimensional learned representation that already encodes price-volume structure, bid/ask balance, and cross-level aggregation.

**InceptionModule — 3 towers, 96 channels total:**

The Inception module addresses a fundamental question: at what temporal scale are the most important patterns? A 3-timestep window captures very short-term momentum. A 5-timestep window captures slightly longer-range patterns. The right answer likely varies by market regime, by horizon, and even by which stock you're looking at.

Rather than picking one, the Inception module uses all three simultaneously:

- **Tower 1**: `1×1` conv (channel mixing) → `3×1` conv → 32 channels. Captures short-range temporal patterns over 3 steps.
- **Tower 2**: `1×1` conv (channel mixing) → `5×1` conv → 32 channels. Captures medium-range temporal patterns over 5 steps.
- **Tower 3**: `MaxPool(3×1, stride=1)` → `1×1` conv → 32 channels. A pooling branch that produces smoothed, max-response features.

The `1×1` convolutions before each branch are "channel mixers" — they're cheap 1-timestep convolutions that let the network recombine the 16 incoming channels in learned ways before passing them to the larger filter. This adds nonlinearity and expressiveness without computational cost.

The three tower outputs are concatenated along the channel dimension: `32 + 32 + 32 = 96` channels. Output: `(batch, 96, 100, 1)`.

The intuition from technical analysis: practitioners often plot moving averages with multiple window sizes simultaneously to capture both short-term momentum and longer-term trend. The Inception module learns its own "decay rates" for those windows from data rather than having them hand-specified.

**LSTM + Output:**

After squeezing the trailing spatial dimension and permuting to `(batch, 100, 96)`, the LSTM receives a sequence of 100 timesteps where each step is a 96-dimensional vector — already rich with extracted spatial features. The LSTM only needs to model the temporal dynamics of how these representations evolve, rather than having to simultaneously discover both spatial structure and temporal dynamics.

`LSTM(input_size=96, hidden_size=64, batch_first=True)` → take the last timestep output → `Linear(64, 3)` → 3 logits.

Total parameters: **60,691** — exactly matching the original Keras implementation.

---

### Batch 4: Evaluation, Backtest & Notebooks

#### `src/evaluate.py` — Measuring what matters

**`compute_metrics`** returns a full dictionary of results per model per horizon:

- **Weighted F1** — the primary metric. F1 is the harmonic mean of precision and recall: `2 × (precision × recall) / (precision + recall)`. "Weighted" means each class's F1 score is weighted by its sample count before averaging. This penalizes a model that ignores rare classes proportionally to how often those classes appear.

- **Per-class precision and recall** — the diagnostic layer. Precision answers: "when the model says 'up', how often is it actually up?" Recall answers: "of all the real 'up' events, what fraction did the model catch?" A model can have high precision but low recall (very conservative — only predicts up when extremely confident, misses most events) or the reverse (fires on everything, catches most events but with many false alarms). These numbers reveal *how* a model is failing, not just that it is.

- **Cohen's kappa** — chance-corrected agreement. Kappa normalizes for the baseline probability of agreeing by chance. A kappa of 0 means the model is no better than a random classifier that knows the class frequencies. A kappa of 1 is perfect. Unlike accuracy, kappa handles class imbalance correctly by construction.

**`measure_throughput`** times predictions-per-second by running 3 warm-up passes (so any just-in-time compilation or caching doesn't skew results) followed by 10 timed passes, averaging across the timed passes. In a live trading context, the model must produce a prediction before the next LOB update arrives — throughput determines whether a model is practically deployable at a given event frequency.

**`build_results_table`** assembles the final comparison table: a DataFrame with model names as rows and horizons `k=1,2,3,5,10` as columns, each cell containing the weighted F1 for that model-horizon pair. This is the single most important output of the entire project — the quantitative answer to "does deep learning outperform classical ML on LOB prediction?"

#### `src/backtest.py` — Turning predictions into a trading simulation

The backtest validates that the model's predictions are practically useful, not just statistically above baseline. The setup matches Section V.D of the paper exactly:

**Signal mapping:**
- Signal 2 (predicted up) → go long: buy 1 share, but execute at the price 5 timesteps later (`slippage_steps=5`) to account for the fact that you can't execute at the exact prediction moment — the market has moved by the time your order hits the book
- Signal 0 (predicted down) → go short: sell 1 share at t+5
- Signal 1 (stationary) → do nothing, stay in current position

**Holding rule:** hold the long position until a signal 0 appears; hold the short until a signal 2 appears. This means each trade duration is variable — determined by when the opposite signal next fires.

**Day-close rule:** at the end of each trading day boundary, force-close all open positions at the prevailing mid-price regardless of signal. This prevents overnight position accumulation and matches the paper's simulation.

**Day boundary estimation:** the test set has 31,837 windows covering 3 trading days. Since the raw CSV has no day-marker column, we divide evenly: boundaries at windows 0, 10,612, 21,224, 31,837.

**`compute_tstat`** runs a one-sample t-test on the 3 daily P&L values against zero. The t-statistic is essentially a Sharpe ratio normalized for sample size — it answers whether profits are consistently positive rather than the result of one lucky day. The paper reports t-statistics above 5 for DeepLOB on LSE; on FI-2010 with only 3 test days any positive t-stat is meaningful.

#### How the 4 notebooks connect end-to-end

The notebooks are not self-contained — they are the execution layer that calls `src/` functions, records results, and produces figures. The dependency chain:

1. **`01_eda.ipynb`** — no models, no training. Loads data via `load_raw` and `extract_features_labels`, produces 4 figures saved to `results/`. Purpose: establish that class imbalance is real, temporal structure exists, normalization is correct.

2. **`02_baselines.ipynb`** — trains and evaluates MajorityClassifier, LogisticModel, and XGBoostModel (with grid search) at all 5 horizons. Produces Ablation A (XGBoost raw vs. engineered features) and Ablation B (LR with/without class weighting). Saves metric dictionaries for use in notebook 04's unified table.

3. **`03_deeplob.ipynb`** — the long-running notebook. Trains SimpleLSTM and DeepLOB for all 5 horizons using `train_neural`, plots loss and val F1 curves per epoch, saves 10 checkpoints to `models/`. This is the notebook to kick off and leave running on MPS — each DeepLOB horizon takes ~1–2 hours on Apple Silicon.

4. **`04_evaluation.ipynb`** — depends on all of the above being complete. Loads all checkpoints, runs inference on the held-out test set, calls `build_results_table` to produce the unified 5×5 comparison, plots confusion matrices at k=10, runs `run_backtest` on DeepLOB test predictions, plots cumulative P&L, measures throughput for all models, and produces the final ablation table (Simple LSTM vs. DeepLOB).

**Critical dependency:** nothing in notebook 04 can run until notebook 03 has finished training and saved all checkpoints. Starting DeepLOB training as early as possible is the single most time-sensitive task remaining.

---

## 2026-04-23 — Boundary-aware windowing (`src/data_loader.py`)

- Identified that FI-2010 concatenates 5 stocks with no boundary marker column
- Sliding windows that straddle stock boundaries are invalid time series inputs
- Implemented `find_stock_boundaries()` using mid-price jump detection (threshold = mean + 10*std)
- Implemented `make_windows()` that skips any window crossing a stock boundary
- Affected windows: ~396 out of 300,000+ in training (~0.1% of data)
- NOTE: The original DeepLOB paper and reference notebook do not handle this —
  they slide naively across the full concatenated file. This is a deliberate
  methodological improvement over the reference implementation.
- Next: build PyTorch Dataset and DataLoader classes around `make_windows()`

# Setup

## Requirements

- Python 3.11+
- pip

GPU strongly recommended for retraining (CUDA or Apple Silicon MPS). Notebook 04 evaluation runs on CPU in a few minutes using the pre-trained checkpoints.

## Installation

```bash
git clone https://github.com/MehdiTC/deeplob-midprice-prediction.git
cd deeplob-midprice-prediction
pip install -r requirements.txt
```

## Data

The FI-2010 dataset is not included in this repository. To obtain it:

1. Download the FI-2010 benchmark from the original authors or a public mirror (e.g. https://www.kaggle.com/datasets/freemanone/fi2010?resource=download)
2. Place the files as:
   - `data/raw/FI2010_train.csv` — days 1–7 (362,400 rows × 149 columns)
   - `data/raw/FI2010_test.csv` — days 8–10 (31,937 rows × 149 columns)

The CSVs are already z-score normalised. Do not renormalise them.

## Running the Project

Run notebooks in order:

```bash
jupyter notebook
```

| Notebook | What it does | Runtime |
|---|---|---|
| `01_eda.ipynb` | Data exploration and class distribution | ~1 min |
| `02_baselines.ipynb` | Majority, Logistic, XGBoost training and evaluation | ~5 min |
| `03_deeplob.ipynb` | SimpleLSTM and DeepLOB training | ~8–12 hours on GPU |
| `04_evaluation.ipynb` | Full results, backtest, ablations, error analysis | ~5 min |

## Skipping Training

Pre-trained checkpoints are committed to `models/`. To skip straight to evaluation:

1. Place the FI-2010 data files as above
2. Run notebooks 01 and 02 to generate `results/classical_results.pkl`
3. Run the checkpoint rebuild cell in notebook 03 to generate `results/neural_results.pkl`
4. Run notebook 04

## Hardware Notes

- Training was performed on a Duke CS cluster node (NVIDIA RTX A5000, CUDA 11.8)
- The code automatically detects CUDA → MPS → CPU in that priority order
- On Apple Silicon (MPS), expect ~10–15× slower training than CUDA
- Notebook 04 inference runs comfortably on CPU

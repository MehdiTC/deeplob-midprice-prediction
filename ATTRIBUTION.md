# Attribution

## Paper Being Reproduced

Zhang, Z., Zohren, S., & Roberts, S. (2019). DeepLOB: Deep Convolutional Neural Networks for Limit Order Books. *IEEE Transactions on Signal Processing*, 67(11), 3001–3012. https://doi.org/10.1109/TSP.2019.2907260

The original Keras/TensorFlow reference implementation is included at `data/raw/DeepLOB.ipynb` and was used as the authoritative specification for the PyTorch reimplementation in `src/models/deeplob.py`.

## Dataset

Ntakaris, A., Magris, M., Kanniainen, J., Gabbouj, M., & Iosifidis, A. (2018). Benchmark Dataset for Mid-Price Forecasting of Limit Order Book Data with Machine Learning Methods. *Journal of Forecasting*, 37(8), 852–866.

FI-2010 dataset: 10 days of limit order book data for 5 stocks from the Nasdaq Nordic market, with pre-computed labels at 5 prediction horizons. Accessed via the publicly available benchmark release.

## AI-Generated Code

This project was developed with substantial assistance from Claude (Anthropic), an AI assistant, in an agentic coding mode. AI assistance was used throughout the codebase. The following files contain significant AI-generated content:

| File | AI contribution |
|---|---|
| `src/data_loader.py` | Entire file |
| `src/features.py` | Entire file |
| `src/train.py` | Little to none |
| `src/evaluate.py` | Entire file |
| `src/backtest.py` | Substantial portions |
| `src/boundary_eval_backtest.py` | Substantial portions |
| `src/models/baseline.py` | None |
| `src/models/logistic.py` | None |
| `src/models/xgboost_model.py` | Substantial portions |
| `src/models/lstm_baseline.py` | None |
| `src/models/deeplob.py` | Substantial portions |
| `notebooks/01_eda.ipynb` | Substantial portions |
| `notebooks/02_baselines.ipynb` | Substantial portions |
| `notebooks/03_deeplob.ipynb` | Substantial portions |
| `notebooks/04_evaluation.ipynb` | Substantial portions |

All AI-generated code was reviewed, debugged, and validated by me, Mehdi Touhami Chahdi. Key architectural decisions (weight initialisation fix for LeakyReLU, Adam epsilon=1 from paper, patience tuning for longer horizons, class weighting for imbalanced horizons) were identified and reasoned through collaboratively. I'm fully responsible for the correctness and integrity of all submitted work.

## External Libraries

- **PyTorch** (torch): neural network implementation and GPU training
- **scikit-learn**: logistic regression, evaluation metrics, class weight computation
- **XGBoost**: gradient boosted trees
- **NumPy / Pandas**: data manipulation
- **Matplotlib / Seaborn**: visualisation
- **SciPy**: t-test for backtest significance
- **tqdm**: progress bars
- **Jupyter**: notebook environment

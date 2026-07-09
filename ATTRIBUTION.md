# Attribution

## Paper Being Reproduced

Zhang, Z., Zohren, S., & Roberts, S. (2019). DeepLOB: Deep Convolutional Neural Networks for Limit Order Books. *IEEE Transactions on Signal Processing*, 67(11), 3001–3012. https://doi.org/10.1109/TSP.2019.2907260

The original Keras/TensorFlow reference implementation is included at `data/raw/DeepLOB.ipynb` and was used as the authoritative specification for the PyTorch reimplementation in `src/models/deeplob.py`.

## Dataset

Ntakaris, A., Magris, M., Kanniainen, J., Gabbouj, M., & Iosifidis, A. (2018). Benchmark Dataset for Mid-Price Forecasting of Limit Order Book Data with Machine Learning Methods. *Journal of Forecasting*, 37(8), 852–866.

FI-2010 dataset: 10 days of limit order book data for 5 stocks from the Nasdaq Nordic market, with pre-computed labels at 5 prediction horizons. Accessed via the publicly available benchmark release.

## External Libraries

- **PyTorch** (torch): neural network implementation and GPU training
- **scikit-learn**: logistic regression, evaluation metrics, class weight computation
- **XGBoost**: gradient boosted trees
- **NumPy / Pandas**: data manipulation
- **Matplotlib / Seaborn**: visualisation
- **SciPy**: t-test for backtest significance
- **tqdm**: progress bars
- **Jupyter**: notebook environment

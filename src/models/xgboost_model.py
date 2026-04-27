"""XGBoost classifier with grid-search over three hyperparameter configs.

AI attribution: this file contains AI-assisted implementation reviewed, debugged,
and validated by the author; see ATTRIBUTION.md.
"""

import numpy as np
from sklearn.metrics import f1_score
from xgboost import XGBClassifier


CONFIGS = [
    dict(n_estimators=100, max_depth=3, learning_rate=0.1),
    dict(n_estimators=200, max_depth=6, learning_rate=0.05),
    dict(n_estimators=300, max_depth=6, learning_rate=0.01),
]


class XGBoostModel:
    """XGBoost multi-class classifier for LOB snapshot prediction.

    Supports grid search over CONFIGS using validation weighted-F1.
    """

    def __init__(self):
        self._model: XGBClassifier | None = None
        self.best_config: dict | None = None

    def grid_search(
        self,
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        configs: list | None = None,
        verbose: bool = True,
    ) -> dict:
        """Train each config on (X_tr, y_tr) and pick the best by val weighted-F1.

        Returns the winning config dict and stores it in self.best_config.
        """
        configs = configs or CONFIGS
        best_f1, best_cfg = -1.0, None

        for cfg in configs:
            m = XGBClassifier(
                **cfg,
                eval_metric="mlogloss",
                tree_method="hist",
                nthread=1,
                verbosity=0,
            )
            m.fit(X_tr, y_tr)
            preds = m.predict(X_val)
            f1 = f1_score(y_val, preds, average="weighted")
            if verbose:
                print(f"  config={cfg}  val_F1={f1:.4f}")
            if f1 > best_f1:
                best_f1, best_cfg = f1, cfg

        self.best_config = best_cfg
        if verbose:
            print(f"  → best config: {best_cfg}  (val_F1={best_f1:.4f})")
        return best_cfg

    def fit(self, X: np.ndarray, y: np.ndarray, config: dict | None = None):
        """Train on (X, y) using the given config (or best_config if omitted)."""
        cfg = config or self.best_config or CONFIGS[0]
        self._model = XGBClassifier(
            **cfg,
            eval_metric="mlogloss",
            tree_method="hist",
            nthread=1,
            verbosity=0,
        )
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class indices."""
        return self._model.predict(X)

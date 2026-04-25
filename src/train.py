"""Training loop, early stopping, and checkpoint utilities for neural models."""

import copy
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


class EarlyStopping:
    """Stops training when validation weighted-F1 stops improving.

    Tracks the best F1 seen so far and counts epochs without improvement.
    """

    def __init__(self, patience: int = 20):
        self.patience = patience
        self.best_f1 = -1.0
        self.best_state: Optional[dict] = None
        self.counter = 0

    def step(self, val_f1: float, model_state: dict) -> bool:
        """Return True if training should stop."""
        if val_f1 > self.best_f1:
            self.best_f1 = val_f1
            self.best_state = copy.deepcopy(model_state)
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Run one training epoch and return average cross-entropy loss."""
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y_batch)
    return total_loss / len(loader.dataset)


def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Run one evaluation pass and return (avg_loss, weighted_f1)."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item() * len(y_batch)

            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(y_batch.cpu().numpy())

    preds_all = np.concatenate(all_preds)
    labels_all = np.concatenate(all_labels)
    avg_loss = total_loss / len(loader.dataset)
    weighted_f1 = f1_score(labels_all, preds_all, average="weighted", zero_division=0)
    return avg_loss, weighted_f1


def train_neural(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    max_epochs: int = 200,
    patience: int = 20,
    lr: float = 0.01,
    eps: float = 1.0,
    class_weights: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> tuple[dict, dict]:
    """Train a neural model with early stopping on validation weighted-F1.

    Returns:
        best_state_dict: state dict of the checkpoint with best val F1
        history: dict with lists train_loss, val_loss, val_f1 (one value per epoch)
    """
    if device is None:
        device = DEVICE
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=eps)
    weight = class_weights.to(device) if class_weights is not None else None
    criterion = nn.CrossEntropyLoss(weight=weight)
    stopper = EarlyStopping(patience=patience)
    history: dict[str, list] = {"train_loss": [], "val_loss": [], "val_f1": []}

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        tr_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_f1 = eval_epoch(model, val_loader, criterion, device)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)

        if verbose and (epoch % 5 == 0 or epoch == 1):
            elapsed = time.time() - t0
            print(
                f"  epoch {epoch:3d}  "
                f"tr_loss={tr_loss:.4f}  val_loss={val_loss:.4f}  "
                f"val_F1={val_f1:.4f}  ({elapsed:.1f}s)"
            )

        if stopper.step(val_f1, model.state_dict()):
            if verbose:
                print(f"  → early stop at epoch {epoch}  best_F1={stopper.best_f1:.4f}")
            break

    return stopper.best_state, history


def save_checkpoint(state_dict: dict, path: str):
    """Save a model state dict to disk."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, path)


def load_checkpoint(model: nn.Module, path: str) -> nn.Module:
    """Load a state dict from disk into model (in-place) and return the model."""
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    return model

"""Optional PyTorch DNN backend.

Only needed when the config asks for real dropout, which scikit-learn's
MLPClassifier does not support. Everything else in the project runs
without torch installed; this module is imported lazily.

    pip install torch --index-url https://download.pytorch.org/whl/cpu
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np


def build_torch_dnn(params: Dict[str, Any], seed: int):
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "models.dnn.backend='torch' requires PyTorch. Either install the "
            "CPU wheel or set backend='sklearn' (which uses L2 instead of "
            "dropout)."
        ) from exc
    return TorchMLPClassifier(params, seed)


class TorchMLPClassifier:
    """Minimal sklearn-style wrapper around a small CPU MLP with dropout."""

    def __init__(self, params: Dict[str, Any], seed: int = 0):
        self.params = dict(params)
        self.seed = int(seed)
        self.model = None
        self.classes_ = np.array([0, 1])

    def _make(self, n_features: int):
        import torch
        import torch.nn as nn

        torch.manual_seed(self.seed)
        torch.set_num_threads(int(self.params.get("n_threads", 2)))

        n_layers = int(self.params.get("n_hidden_layers", 2))
        width = int(self.params.get("neurons_per_layer", 64))
        taper = float(self.params.get("taper", 1.0))
        dropout = float(self.params.get("dropout", 0.0))

        layers, fan_in, current = [], n_features, width
        for _ in range(n_layers):
            size = max(4, int(round(current)))
            layers += [nn.Linear(fan_in, size), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            fan_in, current = size, current * taper
        layers.append(nn.Linear(fan_in, 2))
        return nn.Sequential(*layers)

    def parameters(self):
        return self.model.parameters()

    def fit(self, X, y):
        import torch
        import torch.nn as nn

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        self.classes_ = np.unique(y)
        self.model = self._make(X.shape[1])

        optimiser = torch.optim.Adam(
            self.model.parameters(),
            lr=float(self.params.get("learning_rate", 1e-3)),
            weight_decay=float(self.params.get("alpha", 0.0)),
        )
        loss_fn = nn.CrossEntropyLoss()
        batch_size = int(self.params.get("batch_size", 256))
        epochs = int(self.params.get("epochs", 40))

        tensor_x = torch.from_numpy(X)
        tensor_y = torch.from_numpy(y)
        n = len(X)

        self.model.train()
        for _ in range(epochs):
            # Sequential batching, not shuffled: windows are temporal.
            for start in range(0, n, batch_size):
                xb = tensor_x[start:start + batch_size]
                yb = tensor_y[start:start + batch_size]
                optimiser.zero_grad()
                loss = loss_fn(self.model(xb), yb)
                loss.backward()
                optimiser.step()
        return self

    def predict_proba(self, X):
        import torch
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.from_numpy(np.asarray(X, dtype=np.float32)))
            return torch.softmax(logits, dim=1).numpy()

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1).astype(np.int64)

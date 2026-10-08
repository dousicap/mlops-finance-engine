"""Réseau de neurones PyTorch pour données tabulaires, servi en ONNX.

Entraînement : PyTorch (extra `pip install .[torch]`), perceptron multicouche, arrêt anticipé
sur un jeu de validation interne. Service : le réseau est exporté en ONNX et exécuté par
onnxruntime. L'image de production n'embarque donc pas PyTorch (plusieurs centaines de Mo).
"""

from __future__ import annotations

import io

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


def _dense(X) -> np.ndarray:
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    return X.astype(np.float32)


class TorchMLPClassifier(ClassifierMixin, BaseEstimator):
    """Classifieur binaire compatible scikit-learn (fit, predict_proba)."""

    def __init__(self, hidden=(64, 32), dropout=0.2, lr=1e-3, weight_decay=1e-4,
                 epochs=60, batch_size=256, patience=8, seed=42):
        self.hidden = hidden
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.seed = seed

    def fit(self, X, y):
        import torch
        from torch import nn

        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        X = _dense(X)
        y = np.asarray(y).astype(np.float32)
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1]

        idx = rng.permutation(len(X))
        n_val = max(int(0.15 * len(X)), 1)
        val, tr = idx[:n_val], idx[n_val:]
        Xt, yt = torch.from_numpy(X[tr]), torch.from_numpy(y[tr])
        Xv, yv = torch.from_numpy(X[val]), torch.from_numpy(y[val])

        layers, width = [], X.shape[1]
        for h in self.hidden:
            layers += [nn.Linear(width, h), nn.ReLU(), nn.Dropout(self.dropout)]
            width = h
        layers.append(nn.Linear(width, 1))
        net = nn.Sequential(*layers)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.BCEWithLogitsLoss()

        best, best_state, wait = float("inf"), None, 0
        for epoch in range(self.epochs):
            net.train()
            order = torch.randperm(len(Xt))
            for i in range(0, len(Xt), self.batch_size):
                b = order[i:i + self.batch_size]
                opt.zero_grad()
                loss = loss_fn(net(Xt[b]).squeeze(1), yt[b])   # 1. passe avant, 2. perte
                loss.backward()                                 # 3. gradients
                opt.step()                                      # 4. mise à jour des poids
            net.eval()
            with torch.no_grad():
                val_loss = loss_fn(net(Xv).squeeze(1), yv).item()
            if val_loss < best - 1e-4:
                best, wait = val_loss, 0
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
            else:
                wait += 1
                if wait >= self.patience:
                    break
        net.load_state_dict(best_state)
        net.eval()
        self.epochs_run_ = epoch + 1
        self.val_loss_ = best

        buf = io.BytesIO()
        torch.onnx.export(
            net, torch.zeros(1, X.shape[1]), buf, input_names=["x"], output_names=["logit"],
            dynamic_axes={"x": {0: "n"}, "logit": {0: "n"}}, dynamo=False,
        )
        self.onnx_model_ = buf.getvalue()
        self._session = None
        return self

    def _sess(self):
        if getattr(self, "_session", None) is None:
            import onnxruntime as ort

            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1
            self._session = ort.InferenceSession(self.onnx_model_, opts, providers=["CPUExecutionProvider"])
        return self._session

    def predict_proba(self, X):
        logit = self._sess().run(None, {"x": _dense(X)})[0].reshape(-1)
        p = 1.0 / (1.0 + np.exp(-logit))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def __getstate__(self):
        # La session onnxruntime n'est pas sérialisable : elle est recréée au premier appel.
        state = self.__dict__.copy()
        state["_session"] = None
        return state

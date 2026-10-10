"""PyTorch MLP with a learned Type embedding, early-stopped on val PR-AUC."""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

from src.models.deep_utils import EarlyStopping
from src.utils.metrics import pr_auc

EMBED_DIM = 2
N_TYPES = 3


class TypeMLP(nn.Module):
    def __init__(self, n_numeric: int, hidden: tuple[int, ...] = (64, 32)) -> None:
        super().__init__()
        self.embed = nn.Embedding(N_TYPES, EMBED_DIM)
        layers: list[nn.Module] = []
        width = n_numeric + EMBED_DIM
        for h in hidden:
            layers += [nn.Linear(width, h), nn.ReLU(), nn.Dropout(0.2)]
            width = h
        layers.append(nn.Linear(width, 1))  # logit; sigmoid applied at scoring time
        self.net = nn.Sequential(*layers)

    def forward(self, x_num: torch.Tensor, x_type: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x_num, self.embed(x_type)], dim=1)).squeeze(1)


def fit_torch_mlp(
    X_tr: np.ndarray,
    t_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    t_val: np.ndarray,
    y_val: np.ndarray,
    seed: int = 42,
    max_epochs: int = 200,
    patience: int = 15,
) -> tuple[TypeMLP, np.ndarray]:
    """Return the best-val model (by PR-AUC) and its val scores."""
    torch.manual_seed(seed)
    model = TypeMLP(X_tr.shape[1])
    pos_weight = torch.tensor(
        [(y_tr == 0).sum() / (y_tr == 1).sum()], dtype=torch.float32
    )
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=5
    )
    x_tr, t_t = torch.from_numpy(X_tr), torch.from_numpy(t_tr).long()
    y_t = torch.from_numpy(y_tr).float()
    x_v, t_v = torch.from_numpy(X_val), torch.from_numpy(t_val).long()
    stopper, best_state = EarlyStopping(patience), None
    gen = torch.Generator().manual_seed(seed)
    for epoch in range(max_epochs):
        model.train()
        for idx in torch.randperm(len(y_t), generator=gen).split(256):
            opt.zero_grad()
            loss_fn(model(x_tr[idx], t_t[idx]), y_t[idx]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_score = torch.sigmoid(model(x_v, t_v)).numpy()
        score = pr_auc(y_val, val_score)
        sched.step(score)
        if score > stopper.best + stopper.min_delta:
            best_state = copy.deepcopy(model.state_dict())
            best_val = val_score
        if stopper.step(score, epoch):
            break
    model.load_state_dict(best_state)
    return model, best_val


def predict_torch_mlp(model: TypeMLP, X: np.ndarray, t: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(
            model(torch.from_numpy(X), torch.from_numpy(t).long())
        ).numpy()

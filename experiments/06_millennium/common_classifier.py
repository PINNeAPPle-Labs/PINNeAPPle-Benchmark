"""Validation-selected PINNeAPPle classifier used by the Millennium learning probes.

Candidates (all from the PINNeAPPle model registry): GenericMLP (``bench_mlp``, GELU), GenericResMLP
(``bench_res_mlp``) and ModifiedMLP with a low Fourier bandwidth. Each candidate is trained with
early stopping on the validation split; the configuration with the best validation accuracy is the
only one evaluated on the test split. Inputs are standardised with training statistics.
"""
from __future__ import annotations

import numpy as np
import torch


def _candidates(d_in, n_cls):
    from pinneapple_neural import build_model
    from pinneapple_neural.architectures.modified_mlp import ModifiedMLP
    return {
        "bench_mlp(w128,d3,gelu)": lambda: build_model("bench_mlp", in_dim=d_in, out_dim=n_cls, width=128, depth=3, act="gelu"),
        "bench_mlp(w256,d4,gelu)": lambda: build_model("bench_mlp", in_dim=d_in, out_dim=n_cls, width=256, depth=4, act="gelu"),
        "bench_res_mlp(w128,d4)": lambda: build_model("bench_res_mlp", in_dim=d_in, out_dim=n_cls, width=128, depth=4),
        "modified_mlp(sigma0.1)": lambda: ModifiedMLP(in_dim=d_in, out_dim=n_cls, hidden_dim=128, n_layers=3, n_fourier=32, sigma=0.1),
    }


def fit_select(X, y, tr, va, te, n_cls, epochs=300, weight=None, seed=0, lr=1e-3, wd=1e-4, dropout_input=0.0):
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xn = torch.tensor((X - mu) / sd, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    w = None if weight is None else torch.tensor(weight, dtype=torch.float32)
    report = {}
    best_name, best_acc, best_model = None, -1.0, None
    for name, mk in _candidates(X.shape[1], n_cls).items():
        torch.manual_seed(seed)
        m = mk()
        opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
        b_acc, b_state, bad = -1.0, None, 0
        tr_t = torch.tensor(tr)
        for ep in range(epochs):
            m.train()
            for idx in tr_t[torch.randperm(len(tr_t))].split(256):
                loss = torch.nn.functional.cross_entropy(m(Xn[idx]).y, yt[idx], weight=w)
                opt.zero_grad(); loss.backward(); opt.step()
            m.eval()
            with torch.no_grad():
                acc = float((m(Xn[va]).y.argmax(1) == yt[va]).float().mean())
            if acc > b_acc:
                b_acc, b_state, bad = acc, {k: v.clone() for k, v in m.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= 30:
                    break
        m.load_state_dict(b_state)
        report[name] = {"val_accuracy": b_acc, "epochs": ep + 1}
        if b_acc > best_acc:
            best_name, best_acc, best_model = name, b_acc, m
    with torch.no_grad():
        p = best_model(Xn[te]).y.argmax(1).numpy()
    return {"selected_on_validation": best_name, "candidates": report,
            "test_accuracy": float((p == y[te]).mean()),
            "confusion": [[int(((y[te] == i) & (p == j)).sum()) for j in range(n_cls)] for i in range(n_cls)]}, p

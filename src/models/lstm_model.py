"""
LSTM model for day-ahead wind generation, used by train_models.py (--lstm).

How the sequences are built (no data leakage)
---------------------------------------------
To predict hour T, the LSTM reads the feature rows of the L hours ending at T
(T-L+1, ..., T), using exactly the same feature columns as the other models.
Every one of those rows is already available at forecast time t = T-24:
  * generation lags at hour h <= T are generation from h-24 <= t or earlier;
  * the weather forecast for each hour h was issued about one day before h;
  * the calendar is known in advance.
So the LSTM sees how the forecast and the lags evolve over the day leading to
the target hour, without seeing anything a forecaster would not have.

Tuning
------
Hyper-parameters are chosen on the last 20 % of the training period (with a
48-hour gap), using early stopping on that validation block. The model is then
refitted on the whole training period for the best number of epochs. This is
simpler than the 4-fold CV used for the other models (an LSTM is too slow to
train many times on a laptop); state this in the report.
"""
import copy
import importlib.util
import time

import numpy as np
import pandas as pd

# PyTorch is imported only when the LSTM trains (see _load_torch). Loading it at
# the start makes PyTorch and XGBoost compete for CPU threads, which slows XGBoost
# down many times and distorts the timing comparison.
HAS_TORCH = importlib.util.find_spec("torch") is not None
torch = nn = LSTMRegressor = None


def _load_torch():
    global torch, nn, LSTMRegressor
    if torch is not None:
        return
    import os
    import torch as _torch
    from torch import nn as _nn
    torch, nn = _torch, _nn
    torch.set_num_threads(int(os.environ.get("N_THREADS", str(os.cpu_count() or 1))))

    class _LSTMRegressor(nn.Module):
        def __init__(self, n_features, hidden):
            super().__init__()
            self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
            self.head = nn.Linear(hidden, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :]).squeeze(-1)

    LSTMRegressor = _LSTMRegressor

SEQ_LEN = 24            # hours of history in each sequence
BATCH = 256
LR = 1e-3
SEED = 42
DEVICE = "cpu"          # "cuda" to train on an NVIDIA GPU (set by tune_fit_predict)


# ------------------------------------------------------------- sequences
def build_grid(grid, feats, cap, train_mask_fn):
    """One float32 array with every province's hourly feature rows stacked,
    separated by SEQ_LEN rows of NaN so no sequence crosses two provinces.

    grid: hourly feature table on a complete grid (NaN where data is missing)
    cap:  capacity proxy per province (for normalising the generation lags)
    train_mask_fn: function(datetime_series) -> boolean mask of training rows
    Returns (array, lookup) where lookup maps (province, datetime) -> row index.
    """
    blocks, keys, offset = [], [], 0
    hist = [c for c in feats if c.startswith("gen_")]
    for prov, g in grid.groupby("provincia", sort=True):
        g = g.sort_values("datetime")
        x = g[feats].astype("float64").copy()
        x[hist] = x[hist].div(cap[prov])
        blocks.append(x.values)
        keys.append(pd.DataFrame({"provincia": prov, "datetime": g["datetime"].values,
                                  "row": np.arange(offset, offset + len(g))}))
        offset += len(g)
        blocks.append(np.full((SEQ_LEN, len(feats)), np.nan))
        offset += SEQ_LEN
    arr = np.vstack(blocks)

    # Standardise with training-period statistics only
    lookup = pd.concat(keys, ignore_index=True)
    tr_rows = lookup.loc[train_mask_fn(pd.to_datetime(lookup["datetime"], utc=True)), "row"]
    mu = np.nanmean(arr[tr_rows.values], axis=0)
    sd = np.nanstd(arr[tr_rows.values], axis=0)
    sd[sd == 0] = 1.0
    arr = ((arr - mu) / sd).astype(np.float32)
    lookup["datetime"] = pd.to_datetime(lookup["datetime"], utc=True)
    return arr, lookup.set_index(["provincia", "datetime"])["row"]


def row_index(rows, lookup):
    """Grid row index of each (province, datetime) in a DataFrame."""
    keys = pd.MultiIndex.from_frame(rows[["provincia", "datetime"]])
    return lookup.reindex(keys).values.astype(np.int64)


def valid_windows(arr, idx):
    """True where the SEQ_LEN-hour window ending at idx has no missing values."""
    ok = idx >= SEQ_LEN - 1
    out = np.zeros(len(idx), dtype=bool)
    cum = np.concatenate([[0], np.cumsum(np.isnan(arr).any(axis=1))])
    i = idx[ok]
    out[ok] = (cum[i + 1] - cum[i + 1 - SEQ_LEN]) == 0
    return out


def gather(view, idx):
    """Batch of sequences, shape (batch, SEQ_LEN, n_features)."""
    return np.ascontiguousarray(view[idx - SEQ_LEN + 1].transpose(0, 2, 1))


# ------------------------------------------------------------------ model
def _predict(model, view, idx):
    model.eval()
    preds = []
    with torch.no_grad():
        for s in range(0, len(idx), 4096):
            xb = torch.from_numpy(gather(view, idx[s:s + 4096])).to(DEVICE)
            preds.append(model(xb).cpu().numpy())
    return np.concatenate(preds) if preds else np.array([])


def _fit(view, idx_tr, y_tr, hidden, epochs, idx_va=None, y_va=None, patience=3):
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    model = LSTMRegressor(view.shape[1], hidden).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    best_rmse, best_state, best_epoch, bad = np.inf, None, epochs, 0
    for epoch in range(epochs):
        model.train()
        perm = rng.permutation(len(idx_tr))
        for s in range(0, len(perm), BATCH):
            b = perm[s:s + BATCH]
            xb = torch.from_numpy(gather(view, idx_tr[b])).to(DEVICE)
            yb = torch.from_numpy(y_tr[b].astype(np.float32)).to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        if idx_va is not None:
            val = float(np.sqrt(np.mean((_predict(model, view, idx_va) - y_va) ** 2)))
            print(f"      epoch {epoch + 1}: validation nRMSE {100 * val:.2f} %", flush=True)
            if val < best_rmse - 1e-5:
                best_rmse, best_epoch, bad = val, epoch + 1, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                bad += 1
                if bad >= patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_rmse, best_epoch


def tune_fit_predict(arr, idx_tr, y_tr, t_tr, idx_te, quick=False, device="cpu"):
    """Tune on the last 20 % of training, refit on all training, predict test.
    Returns (predictions, info dict)."""
    if not HAS_TORCH:
        raise ImportError("PyTorch is not installed: pip install torch "
                          "--index-url https://download.pytorch.org/whl/cpu")
    global DEVICE
    _load_torch()
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda, but PyTorch cannot see a GPU. Install the CUDA "
                           "version of PyTorch (see pytorch.org) and check 'nvidia-smi'.")
    DEVICE = device
    if device == "cuda":
        print(f"    LSTM on GPU: {torch.cuda.get_device_name(0)}", flush=True)
    view = np.lib.stride_tricks.sliding_window_view(arr, SEQ_LEN, axis=0)

    cut = t_tr.quantile(0.8)
    in_fit = (t_tr < cut - pd.Timedelta(hours=48)).values
    in_val = (t_tr >= cut).values
    grid = [32] if quick else [32, 64]
    epochs = 3 if quick else 20

    t0 = time.perf_counter()
    results = []
    for hidden in grid:
        print(f"    LSTM hidden={hidden}: tuning", flush=True)
        _, rmse, ep = _fit(view, idx_tr[in_fit], y_tr[in_fit], hidden, epochs,
                           idx_tr[in_val], y_tr[in_val], patience=1 if quick else 3)
        results.append((rmse, hidden, ep))
    best_rmse, best_hidden, best_epoch = min(results)

    print(f"    LSTM refit on all training data (hidden={best_hidden}, "
          f"{best_epoch} epochs)", flush=True)
    t1 = time.perf_counter()
    model, _, _ = _fit(view, idx_tr, y_tr, best_hidden, best_epoch)
    fit_s = time.perf_counter() - t1
    tune_s = time.perf_counter() - t0

    t2 = time.perf_counter()
    pred = _predict(model, view, idx_te)
    pred_ms = 1000 * (time.perf_counter() - t2) / max(1, len(idx_te)) * 1000
    info = {"Tune_s": tune_s, "Fit_s": fit_s, "Pred_ms_per_1k": pred_ms,
            "CV_nRMSE_%": 100 * best_rmse,
            "Best_params": {"hidden": best_hidden, "epochs": best_epoch}}
    return pred, info

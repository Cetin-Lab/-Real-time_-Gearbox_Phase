# -*- coding: utf-8 -*-
"""
Unified training entry point.

Usage
-----
# TPNWHT  (train real and imaginary branches separately)
python train.py --model tpnwht --part real --epochs 2000
python train.py --model tpnwht --part imag --epochs 500

# TPNWHT2 (with learnable decimation layers)
python train.py --model tpnwht2 --part real --epochs 2000
python train.py --model tpnwht2 --part imag --epochs 500

# JWHT    (joint real+imaginary, single model)
python train.py --model jwht --epochs 500

Data
----
Expected .npy files in the working directory:
  x_train.npy   shape: (N_train, 1024)
  y_train.npy   shape: (N_train, 32)   — first 16 = real, last 16 = imag
  x_test.npy    shape: (N_test,  1024)
  y_test.npy    shape: (N_test,  32)
"""

import argparse
import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models import TPNWHT, TPNWHT2, JWHT

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train WHT-based phase estimators")
    p.add_argument("--model",  choices=["tpnwht", "tpnwht2", "jwht"],
                   required=True, help="Model variant to train")
    p.add_argument("--part",   choices=["real", "imag", "joint"],
                   default="joint",
                   help="Which output to train (real/imag for TPNWHT/TPNWHT2; "
                        "joint for JWHT)")
    p.add_argument("--data_dir", default=".",
                   help="Directory containing x_train.npy / y_train.npy etc.")
    p.add_argument("--epochs",   type=int, default=500)
    p.add_argument("--batch",    type=int, default=64)
    p.add_argument("--lr",       type=float, default=1e-3)
    p.add_argument("--out_dir",  default="checkpoints",
                   help="Directory to save model checkpoints")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Data helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_data(data_dir: str, part: str, batch: int):
    """Return (train_loader, test_loader, tag) for the requested part."""
    x_train = np.load(os.path.join(data_dir, "x_train.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    x_test  = np.load(os.path.join(data_dir, "x_test.npy"))
    y_test  = np.load(os.path.join(data_dir, "y_test.npy"))

    model_prefix = os.environ.get("_TRAIN_MODEL_PREFIX", "")
    if part == "real":
        y_tr, y_te, tag = y_train[:, :16], y_test[:, :16], f"{model_prefix}real_parts"
    elif part == "imag":
        y_tr, y_te, tag = y_train[:, 16:], y_test[:, 16:], f"{model_prefix}imag_parts"
    else:  # joint
        y_tr, y_te, tag = y_train, y_test, f"{model_prefix}joint"

    def make_loader(x, y, shuffle):
        ds = TensorDataset(
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )
        return DataLoader(ds, batch_size=batch, shuffle=shuffle)

    return make_loader(x_train, y_tr, True), make_loader(x_test, y_te, False), tag


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def train(model, optimizer, train_loader, test_loader,
          epochs, device, tag, out_dir):
    criterion = nn.MSELoss()
    os.makedirs(out_dir, exist_ok=True)

    best_train_loss = best_test_loss = float("inf")
    best_train_state = best_test_state = None
    train_log, test_log = [], []

    for epoch in range(1, epochs + 1):
        # ── train ──────────────────────────────────────────────────────────
        model.train()
        total = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
        avg_train = total / len(train_loader)
        train_log.append(avg_train)

        # ── validate ────────────────────────────────────────────────────────
        model.eval()
        total = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                total += criterion(model(x), y).item()
        avg_test = total / len(test_loader)
        test_log.append(avg_test)

        print(f"[{epoch:4d}/{epochs}]  train={avg_train:.6f}  test={avg_test:.6f}")

        # ── checkpoints ─────────────────────────────────────────────────────
        if avg_train < best_train_loss:
            best_train_loss  = avg_train
            best_train_state = copy.deepcopy(model.state_dict())
            torch.save(best_train_state,
                       os.path.join(out_dir, f"best_train_model_{tag}.pth"))

        if avg_test < best_test_loss:
            best_test_loss  = avg_test
            best_test_state = copy.deepcopy(model.state_dict())
            torch.save(best_test_state,
                       os.path.join(out_dir, f"best_test_model_{tag}.pth"))

    # ── save final best ──────────────────────────────────────────────────────
    torch.save(best_train_state, os.path.join(out_dir, f"final_best_train_model_{tag}.pth"))
    torch.save(best_test_state,  os.path.join(out_dir, f"final_best_test_model_{tag}.pth"))
    print(f"\nBest train loss: {best_train_loss:.6f}")
    print(f"Best test  loss: {best_test_loss:.6f}")

    # ── loss curve ───────────────────────────────────────────────────────────
    plt.figure(figsize=(10, 4))
    plt.plot(train_log, label="Train Loss")
    plt.plot(test_log,  label="Test Loss")
    plt.xlabel("Epoch"); plt.ylabel("MSE Loss")
    plt.title(f"Training curve — {tag}")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"loss_curve_{tag}.png"))
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── resolve part / model compatibility ──────────────────────────────────
    if args.model == "jwht":
        part = "joint"
    else:
        part = args.part if args.part != "joint" else "real"
        print(f"Training {args.model.upper()} — {part} branch")

    # ── build model ──────────────────────────────────────────────────────────
    # Inject model name prefix into tag via env var (picked up in load_data)
    prefix = args.model.upper() + "_"
    os.environ["_TRAIN_MODEL_PREFIX"] = prefix
    train_loader, test_loader, tag = load_data(args.data_dir, part, args.batch)

    model_map = {"tpnwht": TPNWHT, "tpnwht2": TPNWHT2, "jwht": JWHT}
    model = model_map[args.model]().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model.upper()}  |  Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # ── optional warm-start ──────────────────────────────────────────────────
    ckpt = os.path.join(args.out_dir, f"final_best_train_model_{tag}.pth")
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device))
        print(f"Loaded checkpoint: {ckpt}")

    train(model, optimizer, train_loader, test_loader,
          args.epochs, device, tag, args.out_dir)


if __name__ == "__main__":
    main()

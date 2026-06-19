"""kairos finetune — fetch OHLCV via price_cache and fine-tune a Kronos model.

Usage:
    uv run finetune --model kronos-base --symbol AAPL --output-model ./aapl-model
    uv run finetune --model kronos-small --symbol MSFT --device cuda:0 --epochs 10
    uv run finetune --model kronos-base  --symbol TSLA --start 2020-01-01 --end 2024-06-14
"""
from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="finetune",
        description="Fetch OHLCV history via price_cache and fine-tune a Kronos model.",
    )
    p.add_argument("--model", required=True,
                   help="Base model name (kronos-mini | kronos-small | kronos-base) or HF repo ID.")
    p.add_argument("--symbol", required=True,
                   help="Ticker symbol to fetch training data for, e.g. AAPL.")
    p.add_argument("--interval", default="1d",
                   help="Bar interval (default: 1d).")
    p.add_argument("--start", default=None,
                   help="History start date YYYY-MM-DD (default: 5 years ago).")
    p.add_argument("--end", default=None,
                   help="History end date YYYY-MM-DD (default: today).")
    p.add_argument("--lookback-window", type=int, default=90,
                   help="Training lookback window in bars (default: 90).")
    p.add_argument("--predict-window", type=int, default=10,
                   help="Training prediction window in bars (default: 10).")
    p.add_argument("--epochs", type=int, default=10,
                   help="Number of training epochs (default: 10).")
    p.add_argument("--batch-size", type=int, default=32,
                   help="Training batch size (default: 32).")
    p.add_argument("--lr", type=float, default=4e-5,
                   help="Learning rate (default: 4e-5).")
    p.add_argument("--device", default="cpu",
                   help="Torch device: cpu | cuda | cuda:0 (default: cpu).")
    p.add_argument("--output-model", required=True,
                   help="Directory to save the fine-tuned model.")
    p.add_argument("--train-ratio", type=float, default=0.7)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0,
                   help="DataLoader worker count (default: 0).")
    p.add_argument("--remote", action="store_true",
                   help="Use remote PostgreSQL price_cache instead of local SQLite.")
    p.add_argument("--log-interval", type=int, default=10,
                   help="Print training loss every N steps (default: 10).")
    return p


# ---------------------------------------------------------------------------
# In-memory Dataset (mirrors CustomKlineDataset without CSV dependency)
# ---------------------------------------------------------------------------

import torch
from torch.utils.data import DataLoader, Dataset


class _OHLCVDataset(Dataset):
    """Sliding-window dataset built from a price_cache DataFrame."""

    FEATURES = ["open", "high", "low", "close", "volume", "amount"]
    TIME_FEATURES = ["minute", "hour", "weekday", "day", "month"]

    def __init__(
        self,
        df: pd.DataFrame,
        split: str = "train",
        lookback_window: int = 90,
        predict_window: int = 10,
        clip: float = 5.0,
        seed: int = 42,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
    ):
        self.window = lookback_window + predict_window + 1
        self.clip = clip
        self.seed = seed
        self.split = split
        self.py_rng = random.Random(seed)

        df = df.copy()
        ts = df.index.to_series().reset_index(drop=True)
        df = df.reset_index(drop=True)

        # Derive time features from the DatetimeIndex
        df["minute"] = ts.dt.minute
        df["hour"] = ts.dt.hour
        df["weekday"] = ts.dt.weekday
        df["day"] = ts.dt.day
        df["month"] = ts.dt.month

        # Ensure amount column exists
        if "amount" not in df.columns:
            df["amount"] = df["close"] * df["volume"]

        data = df[self.FEATURES + self.TIME_FEATURES].fillna(method="ffill")

        n = len(data)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        if split == "train":
            self.data = data.iloc[:train_end].reset_index(drop=True)
        elif split == "val":
            self.data = data.iloc[train_end:val_end].reset_index(drop=True)
        else:
            self.data = data.iloc[val_end:].reset_index(drop=True)

        self.n_samples = max(0, len(self.data) - self.window + 1)
        print(f"[{split.upper()}] rows={len(self.data)}, samples={self.n_samples}")

    def set_epoch_seed(self, epoch: int) -> None:
        self.py_rng.seed(self.seed + epoch)
        self._epoch = epoch

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        max_start = len(self.data) - self.window
        if max_start <= 0:
            raise ValueError("Dataset too small for a single window")

        if self.split == "train":
            epoch = getattr(self, "_epoch", 0)
            start_idx = (idx * 9973 + (epoch + 1) * 104729) % (max_start + 1)
        else:
            start_idx = idx % (max_start + 1)

        window = self.data.iloc[start_idx: start_idx + self.window]
        x = window[self.FEATURES].values.astype(np.float32)
        x_stamp = window[self.TIME_FEATURES].values.astype(np.float32)

        mu, sigma = x.mean(0), x.std(0)
        x = np.clip((x - mu) / (sigma + 1e-5), -self.clip, self.clip)

        return torch.from_numpy(x), torch.from_numpy(x_stamp)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _train(model, tokenizer, device, train_loader, val_loader,
           epochs: int, lr: float, output_dir: str, log_interval: int,
           train_dataset, val_dataset) -> None:

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  betas=(0.9, 0.95), weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        steps_per_epoch=len(train_loader),
        epochs=epochs,
        pct_start=0.03,
        div_factor=10,
    )

    best_val_loss = float("inf")
    step = 0

    for epoch in range(epochs):
        model.train()
        train_dataset.set_epoch_seed(epoch * 10_000)
        epoch_loss = 0.0
        n_batches = 0

        for batch_x, batch_x_stamp in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_x_stamp = batch_x_stamp.to(device, non_blocking=True)

            with torch.no_grad():
                tok0, tok1 = tokenizer.encode(batch_x, half=True)

            logits = model(tok0[:, :-1], tok1[:, :-1], batch_x_stamp[:, :-1, :])
            loss, _, _ = model.head.compute_loss(
                logits[0], logits[1], tok0[:, 1:], tok1[:, 1:]
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1
            step += 1

            if step % log_interval == 0:
                lr_now = optimizer.param_groups[0]["lr"]
                print(f"  epoch {epoch+1}/{epochs}  step {step}  "
                      f"loss {loss.item():.4f}  lr {lr_now:.2e}")

        # Validation
        model.eval()
        val_dataset.set_epoch_seed(0)
        val_loss = 0.0
        val_n = 0
        with torch.no_grad():
            for batch_x, batch_x_stamp in val_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_x_stamp = batch_x_stamp.to(device, non_blocking=True)
                tok0, tok1 = tokenizer.encode(batch_x, half=True)
                logits = model(tok0[:, :-1], tok1[:, :-1], batch_x_stamp[:, :-1, :])
                loss, _, _ = model.head.compute_loss(
                    logits[0], logits[1], tok0[:, 1:], tok1[:, 1:]
                )
                val_loss += loss.item()
                val_n += 1

        avg_train = epoch_loss / n_batches if n_batches else 0
        avg_val = val_loss / val_n if val_n else 0
        print(f"Epoch {epoch+1}/{epochs}  train_loss={avg_train:.4f}  "
              f"val_loss={avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            save_path = os.path.join(output_dir, "best_model")
            os.makedirs(save_path, exist_ok=True)
            model.save_pretrained(save_path)
            print(f"  → saved best model (val_loss={best_val_loss:.4f}) to {save_path}")

    # Always save a final checkpoint too
    final_path = os.path.join(output_dir, "final_model")
    os.makedirs(final_path, exist_ok=True)
    model.save_pretrained(final_path)
    print(f"Final model saved to {final_path}")
    print(f"Training complete. Best val loss: {best_val_loss:.4f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device(args.device)

    # ----------------------------------------------------------- fetch history
    import kairos
    import price_cache as pc

    kairos.configure(remote=args.remote)

    tz = "America/New_York"
    end_date = pd.Timestamp(args.end, tz=tz) if args.end else pd.Timestamp.now(tz=tz)
    start_date = (
        pd.Timestamp(args.start, tz=tz)
        if args.start
        else end_date - timedelta(days=5 * 365)
    )

    print(f"Fetching {args.symbol} ({args.interval})  "
          f"{start_date.date()} → {end_date.date()} …")

    raw = pc.get_price_data(
        args.symbol,
        start_date.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d"),
        interval=args.interval,
        db_path=pc.DB_PATH,
    )

    if raw is None or raw.empty:
        print(f"ERROR: no data returned for {args.symbol!r}. "
              "Check the symbol and date range.", file=sys.stderr)
        sys.exit(1)

    # Rename price_cache columns → lowercase
    raw = raw.rename(columns={
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume",
    })
    print(f"Fetched {len(raw)} bars  "
          f"({raw.index[0].date()} – {raw.index[-1].date()})")

    min_required = args.lookback_window + args.predict_window + 1
    if len(raw) < min_required:
        print(
            f"ERROR: need at least {min_required} bars "
            f"(lookback {args.lookback_window} + predict {args.predict_window} + 1) "
            f"but only got {len(raw)}.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ------------------------------------------------------- build datasets
    train_ds = _OHLCVDataset(raw, "train", args.lookback_window, args.predict_window,
                              seed=args.seed, train_ratio=args.train_ratio,
                              val_ratio=args.val_ratio)
    val_ds = _OHLCVDataset(raw, "val", args.lookback_window, args.predict_window,
                            seed=args.seed + 1, train_ratio=args.train_ratio,
                            val_ratio=args.val_ratio)

    if train_ds.n_samples == 0 or val_ds.n_samples == 0:
        print("ERROR: not enough data to build training/validation splits. "
              "Fetch more history or reduce --lookback-window.", file=sys.stderr)
        sys.exit(1)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=device.type != "cpu", drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers,
                            pin_memory=device.type != "cpu")

    # ------------------------------------------------------- load model
    from kairos.cli._models import load_predictor, resolve

    cfg = resolve(args.model)
    # Load tokenizer and model separately (we need them on device for training)
    repo_root = Path(__file__).parent.parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from model import Kronos, KronosTokenizer  # noqa: E402

    print(f"Loading tokenizer  {cfg['tokenizer_id']} …")
    tokenizer = KronosTokenizer.from_pretrained(cfg["tokenizer_id"]).to(device)
    tokenizer.eval()

    print(f"Loading model      {cfg['model_id']} …")
    kronos = Kronos.from_pretrained(cfg["model_id"]).to(device)

    n_params = sum(p.numel() for p in kronos.parameters())
    print(f"Model parameters: {n_params:,}")

    # ------------------------------------------------------- train
    output_dir = args.output_model
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nFine-tuning for {args.epochs} epoch(s) on {device} …")
    _train(kronos, tokenizer, device,
           train_loader, val_loader,
           epochs=args.epochs,
           lr=args.lr,
           output_dir=output_dir,
           log_interval=args.log_interval,
           train_dataset=train_ds,
           val_dataset=val_ds)


if __name__ == "__main__":
    main()

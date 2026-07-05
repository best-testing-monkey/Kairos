"""
train_large_model.py

Trains a Kronos-large (~499M param) predictor.  Supports three training modes:

  distill_only         -- Phase 1 only: train on teacher-predicted tokens
  groundtruth_only     -- single phase on ground-truth tokens (from-scratch or finetune)
  distill_then_finetune -- Phase 1 on distilled tokens, Phase 2 on ground-truth tokens

The distillation token cache must be pre-generated with generate_distilled_tokens.py
before running this script (except for groundtruth_only, which can also accept a
gt-only cache or re-use the same cache file with mode='groundtruth').

Usage
-----
python train_large_model.py --config configs/config_kronos_large.yaml
python train_large_model.py --config configs/config_kronos_large.yaml --skip-phase1
"""

import os
import sys
import time
import signal
import random
import argparse
import logging
from logging.handlers import RotatingFileHandler

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# ---------------------------------------------------------------------------
# Graceful interrupt support
# ---------------------------------------------------------------------------

_stop_requested = False

def _sigint_handler(signum, frame):
    global _stop_requested
    print("\n[SIGINT] Interrupt received - finishing current batch then saving checkpoint...")
    _stop_requested = True

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model import Kronos

from distilled_dataset import DistilledTokenDataset
from config_loader import LargeModelConfig


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(exp_name: str, log_dir: str, rank: int = 0) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(f"large_model_training_rank_{rank}")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger

    file_handler = RotatingFileHandler(
        os.path.join(log_dir, f'large_model_training_rank_{rank}.log'),
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding='utf-8',
    )
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if rank == 0:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(formatter)
        logger.addHandler(console)

    return logger


# ---------------------------------------------------------------------------
# Core training loop
# ---------------------------------------------------------------------------

def _save_checkpoint(path, raw_model, optimizer, scheduler, epoch, best_val_loss):
    torch.save({
        'epoch':           epoch,
        'model_state':     raw_model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'best_val_loss':   best_val_loss,
    }, path)


def train_phase(model, train_loader, val_loader, epochs, optimizer, scheduler,
                save_dir, phase_name, grad_clip, accumulation_steps,
                log_interval, device, logger, rank=0, use_ddp=False,
                checkpoint_interval=500):
    """
    Train for `epochs` on pre-tokenised (s1_ids, s2_ids, stamps) batches.

    Checkpointing:
      - After every completed epoch  → latest_checkpoint.pt
      - Every `checkpoint_interval` optimizer steps → latest_checkpoint.pt
      - On Ctrl-C (SIGINT)           → latest_checkpoint.pt, then clean exit

    Resume: re-run with the same command; the checkpoint is auto-detected.
    """
    global _stop_requested
    _stop_requested = False

    prev_handler = signal.signal(signal.SIGINT, _sigint_handler)

    raw_model     = model.module if use_ddp else model
    best_val_loss = float('inf')
    start_epoch   = 0
    global_step   = 0

    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, 'latest_checkpoint.pt')

    if os.path.exists(ckpt_path):
        if rank == 0:
            print(f"[{phase_name}] Resuming from checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        raw_model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        scheduler.load_state_dict(ckpt['scheduler_state'])
        start_epoch   = ckpt['epoch']
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        steps_per_epoch = max(len(train_loader) // accumulation_steps, 1)
        global_step   = start_epoch * steps_per_epoch
        if rank == 0:
            print(f"[{phase_name}] Resumed at epoch {start_epoch + 1}/{epochs}  "
                  f"(best val so far: {best_val_loss:.4f})")

    if start_epoch >= epochs:
        if rank == 0:
            print(f"[{phase_name}] Already completed ({epochs} epochs). Skipping.")
        signal.signal(signal.SIGINT, prev_handler)
        return best_val_loss

    interrupted = False

    for epoch in range(start_epoch, epochs):
        epoch_start      = time.time()
        model.train()
        epoch_train_loss = 0.0
        train_batches    = 0

        optimizer.zero_grad()

        for batch_idx, (s1_ids, s2_ids, stamps) in enumerate(train_loader):
            s1_ids = s1_ids.to(device, non_blocking=True)
            s2_ids = s2_ids.to(device, non_blocking=True)
            stamps = stamps.to(device, non_blocking=True)

            s1_in,  s2_in  = s1_ids[:, :-1], s2_ids[:, :-1]
            s1_out, s2_out = s1_ids[:, 1:],  s2_ids[:, 1:]
            stamps_in      = stamps[:, :-1, :]

            s1_logits, s2_logits = raw_model(s1_in, s2_in, stamps_in)
            loss, s1_loss, s2_loss = raw_model.head.compute_loss(
                s1_logits, s2_logits, s1_out, s2_out)

            (loss / accumulation_steps).backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_norm=grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % log_interval == 0:
                    lr  = optimizer.param_groups[0]['lr']
                    msg = (f"[{phase_name}] Epoch {epoch+1}/{epochs}, "
                           f"Step {batch_idx+1}/{len(train_loader)}, "
                           f"LR: {lr:.2e}, Loss: {loss.item():.4f} "
                           f"(s1: {s1_loss.item():.4f}, s2: {s2_loss.item():.4f})")
                    logger.info(msg)
                    if rank == 0:
                        print(msg)

                # Periodic mid-epoch checkpoint
                if rank == 0 and checkpoint_interval > 0 and global_step % checkpoint_interval == 0:
                    _save_checkpoint(ckpt_path, raw_model, optimizer, scheduler,
                                     epoch, best_val_loss)

                # Graceful interrupt
                if _stop_requested:
                    if rank == 0:
                        print(f"[{phase_name}] Saving interrupt checkpoint at step {global_step}...")
                        _save_checkpoint(ckpt_path, raw_model, optimizer, scheduler,
                                         epoch, best_val_loss)
                        print(f"[{phase_name}] Checkpoint saved → {ckpt_path}")
                        print(f"[{phase_name}] Resume by re-running the same command.")
                    interrupted = True
                    break

            epoch_train_loss += loss.item()
            train_batches    += 1

        if interrupted:
            break

        # Flush remaining gradient accumulation
        if len(train_loader) % accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_norm=grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # Validation
        model.eval()
        val_loss    = 0.0
        val_batches = 0
        with torch.no_grad():
            for s1_ids, s2_ids, stamps in val_loader:
                s1_ids = s1_ids.to(device, non_blocking=True)
                s2_ids = s2_ids.to(device, non_blocking=True)
                stamps = stamps.to(device, non_blocking=True)
                s1_logits, s2_logits = raw_model(s1_ids[:, :-1], s2_ids[:, :-1],
                                                  stamps[:, :-1, :])
                v_loss, _, _ = raw_model.head.compute_loss(
                    s1_logits, s2_logits, s1_ids[:, 1:], s2_ids[:, 1:])
                val_loss    += v_loss.item()
                val_batches += 1

        if use_ddp:
            t = torch.tensor([epoch_train_loss, train_batches, val_loss, val_batches],
                              dtype=torch.float64, device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            avg_train = t[0].item() / max(int(t[1].item()), 1)
            avg_val   = t[2].item() / max(int(t[3].item()), 1)
        else:
            avg_train = epoch_train_loss / max(train_batches, 1)
            avg_val   = val_loss / max(val_batches, 1)

        epoch_time = time.time() - epoch_start
        summary = (f"\n--- [{phase_name}] Epoch {epoch+1}/{epochs} ---\n"
                   f"Train Loss: {avg_train:.4f}  Val Loss: {avg_val:.4f}  "
                   f"Time: {epoch_time:.1f}s\n")
        logger.info(summary)
        if rank == 0:
            print(summary)

        if avg_val < best_val_loss and rank == 0:
            best_val_loss = avg_val
            best_path     = os.path.join(save_dir, 'best_model')
            os.makedirs(best_path, exist_ok=True)
            raw_model.save_pretrained(best_path)
            msg = (f"[{phase_name}] Best model saved "
                   f"(val loss: {best_val_loss:.4f}) → {best_path}")
            logger.info(msg)
            print(msg)

        # End-of-epoch checkpoint (next epoch to run = epoch+1)
        if rank == 0:
            _save_checkpoint(ckpt_path, raw_model, optimizer, scheduler,
                             epoch + 1, best_val_loss)

    signal.signal(signal.SIGINT, prev_handler)

    if interrupted:
        sys.exit(0)

    return best_val_loss


# ---------------------------------------------------------------------------
# LargeModelTrainer
# ---------------------------------------------------------------------------

class LargeModelTrainer:

    def __init__(self, config_path: str):
        self.config = LargeModelConfig(config_path)
        self.rank = int(os.environ.get('RANK', '0'))
        self.world_size = int(os.environ.get('WORLD_SIZE', '1'))
        self.local_rank = int(os.environ.get('LOCAL_RANK',
                                              str(self.config.device_id)))
        self.device = self._setup_device()

    def _setup_device(self):
        if self.config.use_cuda and torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            return torch.device(f'cuda:{self.local_rank}')
        return torch.device('cpu')

    def _setup_distributed(self):
        if self.world_size > 1 and torch.cuda.is_available():
            if not dist.is_initialized():
                dist.init_process_group(backend='nccl')

    def _set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _build_model(self) -> Kronos:
        cfg = self.config
        model = Kronos(
            s1_bits=cfg.s1_bits,
            s2_bits=cfg.s2_bits,
            n_layers=cfg.n_layers,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            ff_dim=cfg.ff_dim,
            ffn_dropout_p=cfg.ffn_dropout_p,
            attn_dropout_p=cfg.attn_dropout_p,
            resid_dropout_p=cfg.resid_dropout_p,
            token_dropout_p=cfg.token_dropout_p,
            learn_te=cfg.learn_te,
        )
        return model.to(self.device)

    def _build_dataloaders(self, split: str, mode: str, batch_size: int):
        """Load DistilledTokenDataset for a given split and mode."""
        cache_path = os.path.join(self.config.distill_cache_dir,
                                  f'{split}_distilled_tokens.pt')
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Token cache not found: {cache_path}\n"
                "Run generate_distilled_tokens.py first."
            )
        dataset = DistilledTokenDataset(cache_path, mode=mode)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=(split == 'train'),
        )
        return loader

    def run(self, skip_phase1: bool = False, skip_phase2: bool = False):
        self._setup_distributed()
        self._set_seed(self.config.seed)

        use_ddp = dist.is_available() and dist.is_initialized()

        log_dir = os.path.join(self.config.base_save_path, 'logs')
        logger = setup_logging(self.config.exp_name, log_dir, self.rank)

        model = self._build_model()
        param_count = sum(p.numel() for p in model.parameters())
        if self.rank == 0:
            print(f"Kronos-large parameters: {param_count:,}")
        logger.info(f"Kronos-large parameters: {param_count:,}")

        mode = self.config.training_mode

        # ------------------------------------------------------------------ #
        # Phase 1: distillation warm-start                                   #
        # ------------------------------------------------------------------ #
        if mode in ('distill_only', 'distill_then_finetune') and not skip_phase1:
            if self.rank == 0:
                print('\n' + '=' * 60)
                print('Phase 1: Distillation warm-start')
                print('=' * 60)

            p1_save = os.path.join(self.config.large_model_save_path, 'phase1_best')
            os.makedirs(p1_save, exist_ok=True)

            train_loader = self._build_dataloaders('train', 'distill', self.config.phase1_batch_size)
            val_loader = self._build_dataloaders('val', 'distill', self.config.phase1_batch_size)

            if use_ddp:
                model = DDP(model, device_ids=[self.local_rank],
                            output_device=self.local_rank, find_unused_parameters=False)

            optimizer = torch.optim.AdamW(
                (model.module if use_ddp else model).parameters(),
                lr=self.config.phase1_lr,
                betas=(self.config.adam_beta1, self.config.adam_beta2),
                weight_decay=self.config.adam_weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.config.phase1_lr,
                steps_per_epoch=max(len(train_loader) // self.config.accumulation_steps, 1),
                epochs=self.config.phase1_epochs,
                pct_start=0.03,
                div_factor=10,
            )

            best_p1 = train_phase(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=self.config.phase1_epochs,
                optimizer=optimizer,
                scheduler=scheduler,
                save_dir=p1_save,
                phase_name='Phase1-Distill',
                grad_clip=self.config.phase1_grad_clip,
                accumulation_steps=self.config.accumulation_steps,
                log_interval=self.config.log_interval,
                device=self.device,
                logger=logger,
                rank=self.rank,
                use_ddp=use_ddp,
            )
            logger.info(f"Phase 1 best val loss: {best_p1:.4f}")

            # Unwrap DDP so Phase 2 can re-wrap if needed
            if use_ddp:
                model = model.module

        # ------------------------------------------------------------------ #
        # Phase 2: ground-truth finetune                                     #
        # ------------------------------------------------------------------ #
        if mode in ('groundtruth_only', 'distill_then_finetune') and not skip_phase2:
            if self.rank == 0:
                print('\n' + '=' * 60)
                print('Phase 2: Ground-truth finetune')
                print('=' * 60)

            p2_save = os.path.join(self.config.large_model_save_path, 'phase2_best')
            os.makedirs(p2_save, exist_ok=True)

            train_loader = self._build_dataloaders('train', 'groundtruth', self.config.phase2_batch_size)
            val_loader = self._build_dataloaders('val', 'groundtruth', self.config.phase2_batch_size)

            if use_ddp and not isinstance(model, DDP):
                model = DDP(model, device_ids=[self.local_rank],
                            output_device=self.local_rank, find_unused_parameters=False)

            optimizer = torch.optim.AdamW(
                (model.module if use_ddp else model).parameters(),
                lr=self.config.phase2_lr,
                betas=(self.config.adam_beta1, self.config.adam_beta2),
                weight_decay=self.config.adam_weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.config.phase2_lr,
                steps_per_epoch=max(len(train_loader) // self.config.accumulation_steps, 1),
                epochs=self.config.phase2_epochs,
                pct_start=0.03,
                div_factor=10,
            )

            best_p2 = train_phase(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=self.config.phase2_epochs,
                optimizer=optimizer,
                scheduler=scheduler,
                save_dir=p2_save,
                phase_name='Phase2-Groundtruth',
                grad_clip=self.config.phase2_grad_clip,
                accumulation_steps=self.config.accumulation_steps,
                log_interval=self.config.log_interval,
                device=self.device,
                logger=logger,
                rank=self.rank,
                use_ddp=use_ddp,
            )
            logger.info(f"Phase 2 best val loss: {best_p2:.4f}")

        if self.rank == 0:
            print('\nTraining complete.')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Train Kronos-large with optional distillation warm-start')
    parser.add_argument('--config', type=str, required=True, help='Path to config_kronos_large.yaml')
    parser.add_argument('--skip-phase1', action='store_true', help='Skip Phase 1 (distillation)')
    parser.add_argument('--skip-phase2', action='store_true', help='Skip Phase 2 (ground-truth finetune)')
    args = parser.parse_args()

    trainer = LargeModelTrainer(args.config)
    trainer.run(skip_phase1=args.skip_phase1, skip_phase2=args.skip_phase2)

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()

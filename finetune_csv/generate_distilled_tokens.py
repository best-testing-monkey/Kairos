"""
generate_distilled_tokens.py

Runs a teacher Kronos predictor over training/validation data and saves both the
teacher-predicted token sequences and the ground-truth token sequences to a .pt
cache file.  The cache is consumed by DistilledTokenDataset during large model
training, so the teacher never needs to be loaded at training time.

Usage
-----
python generate_distilled_tokens.py --config configs/config_kronos_large.yaml
python generate_distilled_tokens.py --config configs/config_kronos_large.yaml --split val
python generate_distilled_tokens.py --config configs/config_kronos_large.yaml --split train,val --sample
"""

import os
import sys
import argparse

import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model import Kronos, KronosTokenizer

from finetune_base_model import CustomKlineDataset
from config_loader import LargeModelConfig


def generate_split(teacher: Kronos, tokenizer: KronosTokenizer, dataset: CustomKlineDataset,
                   batch_size: int, num_workers: int, device: torch.device,
                   sample_mode: str = 'argmax', temperature: float = 1.0, top_p: float = 1.0):
    """Return dict with keys s1, s2, gt_s1, gt_s2, stamps (all CPU tensors)."""

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    all_pred_s1, all_pred_s2 = [], []
    all_gt_s1, all_gt_s2 = [], []
    all_stamps = []

    teacher.eval()
    tokenizer.eval()

    with torch.no_grad():
        for batch_idx, (batch_x, batch_x_stamp) in enumerate(loader):
            batch_x = batch_x.to(device, non_blocking=True)
            batch_x_stamp = batch_x_stamp.to(device, non_blocking=True)

            # Ground-truth tokens from the tokenizer
            gt_s1, gt_s2 = tokenizer.encode(batch_x, half=True)

            # Teacher predictions: run the teacher in teacher-forcing mode
            token_in_s1 = gt_s1[:, :-1]
            token_in_s2 = gt_s2[:, :-1]
            stamps_in = batch_x_stamp[:, :-1, :]

            s1_logits, s2_logits = teacher(token_in_s1, token_in_s2, stamps_in)

            if sample_mode == 'argmax':
                pred_s1 = s1_logits.argmax(dim=-1)
                pred_s2 = s2_logits.argmax(dim=-1)
            else:
                # Nucleus sampling
                pred_s1 = _sample_tokens(s1_logits, temperature=temperature, top_p=top_p)
                pred_s2 = _sample_tokens(s2_logits, temperature=temperature, top_p=top_p)

            all_pred_s1.append(pred_s1.cpu())
            all_pred_s2.append(pred_s2.cpu())
            all_gt_s1.append(gt_s1[:, 1:].cpu())
            all_gt_s2.append(gt_s2[:, 1:].cpu())
            all_stamps.append(stamps_in.cpu())

            if (batch_idx + 1) % 50 == 0:
                print(f"  Processed {batch_idx + 1}/{len(loader)} batches")

    return {
        's1': torch.cat(all_pred_s1, dim=0),
        's2': torch.cat(all_pred_s2, dim=0),
        'gt_s1': torch.cat(all_gt_s1, dim=0),
        'gt_s2': torch.cat(all_gt_s2, dim=0),
        'stamps': torch.cat(all_stamps, dim=0),
    }


def _sample_tokens(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    """Sample from logits with temperature and nucleus (top-p) filtering."""
    B, T, V = logits.shape
    logits = logits.reshape(B * T, V)

    if temperature != 1.0:
        logits = logits / temperature

    probs = torch.softmax(logits, dim=-1)

    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
        cumulative = sorted_probs.cumsum(dim=-1)
        remove_mask = cumulative - sorted_probs > top_p
        sorted_probs[remove_mask] = 0.0
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
        samples = torch.multinomial(sorted_probs, num_samples=1)
        tokens = sorted_indices.gather(dim=-1, index=samples).squeeze(-1)
    else:
        tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)

    return tokens.reshape(B, T)


def main():
    parser = argparse.ArgumentParser(description='Generate distillation token cache from a teacher Kronos model')
    parser.add_argument('--config', type=str, required=True, help='Path to config_kronos_large.yaml')
    parser.add_argument('--split', type=str, default='train,val',
                        help='Comma-separated splits to generate: train,val,test (default: train,val)')
    parser.add_argument('--sample', action='store_true',
                        help='Use multinomial sampling instead of argmax for teacher predictions')
    args = parser.parse_args()

    config = LargeModelConfig(args.config)

    device = torch.device(f'cuda:{config.device_id}' if config.use_cuda and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading finetuned tokenizer...")
    tokenizer = KronosTokenizer.from_pretrained(config.finetuned_tokenizer_path)
    tokenizer = tokenizer.to(device)

    print(f"Loading teacher predictor from: {config.teacher_predictor_path}")
    teacher = Kronos.from_pretrained(config.teacher_predictor_path)
    teacher = teacher.to(device)

    teacher_params = sum(p.numel() for p in teacher.parameters())
    print(f"Teacher parameters: {teacher_params:,}")

    os.makedirs(config.distill_cache_dir, exist_ok=True)

    sample_mode = 'sample' if args.sample else config.sample_mode
    print(f"Token generation mode: {sample_mode}")

    splits = [s.strip() for s in args.split.split(',')]
    for split in splits:
        print(f"\n--- Generating '{split}' split ---")
        dataset = CustomKlineDataset(
            data_path=config.data_path,
            data_type=split,
            lookback_window=config.lookback_window,
            predict_window=config.predict_window,
            clip=config.clip,
            seed=config.seed,
            train_ratio=config.train_ratio,
            val_ratio=config.val_ratio,
            test_ratio=config.test_ratio,
        )
        print(f"Dataset size: {len(dataset)} samples")

        cache = generate_split(
            teacher=teacher,
            tokenizer=tokenizer,
            dataset=dataset,
            batch_size=config.distill_batch_size,
            num_workers=config.num_workers,
            device=device,
            sample_mode=sample_mode,
            temperature=config.sampling_temperature,
            top_p=config.top_p,
        )

        out_path = os.path.join(config.distill_cache_dir, f'{split}_distilled_tokens.pt')
        torch.save(cache, out_path)
        print(f"Saved {split} cache: {out_path}")
        print(f"  s1 shape:    {cache['s1'].shape}")
        print(f"  s2 shape:    {cache['s2'].shape}")
        print(f"  gt_s1 shape: {cache['gt_s1'].shape}")
        print(f"  stamps shape:{cache['stamps'].shape}")

    print("\nDistillation cache generation complete.")


if __name__ == '__main__':
    main()

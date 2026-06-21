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
import signal
import argparse
import hashlib
import itertools

import torch
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Graceful interrupt support
# ---------------------------------------------------------------------------

_stop_requested = False

def _sigint_handler(signum, frame):
    global _stop_requested
    print("\n[SIGINT] Finishing current batch then saving checkpoint...")
    _stop_requested = True

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model import Kronos, KronosTokenizer

from finetune_base_model import CustomKlineDataset
from config_loader import LargeModelConfig


# ---------------------------------------------------------------------------
# Dataset fingerprinting
# ---------------------------------------------------------------------------

def _dataset_hash(data_path: str) -> str:
    """Lightweight fingerprint: sorted (filename, size, mtime_ns) for each CSV."""
    h = hashlib.sha256()
    if os.path.isdir(data_path):
        for fname in sorted(os.listdir(data_path)):
            if not fname.endswith('.csv'):
                continue
            st = os.stat(os.path.join(data_path, fname))
            h.update(f"{fname}:{st.st_size}:{st.st_mtime_ns}\n".encode())
    else:
        st = os.stat(data_path)
        h.update(f"{os.path.basename(data_path)}:{st.st_size}:{st.st_mtime_ns}\n".encode())
    return h.hexdigest()[:24]


def _hash_file(cache_dir: str) -> str:
    return os.path.join(cache_dir, 'dataset_hash.txt')


def _cache_is_valid(cache_dir: str, data_path: str, splits: list) -> bool:
    """True when all split .pt files exist and the stored hash matches current data."""
    hf = _hash_file(cache_dir)
    if not os.path.exists(hf):
        return False
    with open(hf) as f:
        stored = f.read().strip()
    if stored != _dataset_hash(data_path):
        return False
    return all(
        os.path.exists(os.path.join(cache_dir, f'{s}_distilled_tokens.pt'))
        for s in splits
    )


def _write_hash(cache_dir: str, data_path: str):
    with open(_hash_file(cache_dir), 'w') as f:
        f.write(_dataset_hash(data_path))


# ---------------------------------------------------------------------------
# Partial checkpoint helpers
# ---------------------------------------------------------------------------

def _partial_ckpt_path(cache_dir: str, split: str) -> str:
    return os.path.join(cache_dir, f'{split}_partial_checkpoint.pt')


def _save_partial(cache_dir: str, split: str, lists: dict,
                  batches_done: int, total_batches: int):
    final = _partial_ckpt_path(cache_dir, split)
    tmp   = final + '.tmp'
    torch.save({
        'batches_done':  batches_done,
        'total_batches': total_batches,
        's1':     torch.cat(lists['s1'],     dim=0),
        's2':     torch.cat(lists['s2'],     dim=0),
        'gt_s1':  torch.cat(lists['gt_s1'],  dim=0),
        'gt_s2':  torch.cat(lists['gt_s2'],  dim=0),
        'stamps': torch.cat(lists['stamps'], dim=0),
    }, tmp)
    os.replace(tmp, final)  # atomic on Linux — never leaves a partial file


def _load_partial(cache_dir: str, split: str):
    """Return (accumulated_lists, start_batch).  Lists contain one pre-cat tensor each."""
    path = _partial_ckpt_path(cache_dir, split)
    if not os.path.exists(path):
        return {'s1': [], 's2': [], 'gt_s1': [], 'gt_s2': [], 'stamps': []}, 0
    ckpt = torch.load(path, weights_only=False)
    lists = {k: [ckpt[k]] for k in ('s1', 's2', 'gt_s1', 'gt_s2', 'stamps')}
    print(f"  Resuming from batch {ckpt['batches_done']}/{ckpt['total_batches']}")
    return lists, int(ckpt['batches_done'])


def generate_split(teacher: Kronos, tokenizer: KronosTokenizer, dataset: CustomKlineDataset,
                   batch_size: int, num_workers: int, device: torch.device,
                   cache_dir: str, split: str,
                   sample_mode: str = 'argmax', temperature: float = 1.0, top_p: float = 1.0,
                   checkpoint_interval: int = 500):
    """
    Generate distilled token cache for one split.

    Checkpointing:
      - Every `checkpoint_interval` batches → {split}_partial_checkpoint.pt
      - On Ctrl-C (SIGINT)                 → same file, then clean exit
    Resume: re-run the same command; the partial checkpoint is auto-detected.
    """
    global _stop_requested
    _stop_requested = False
    prev_handler = signal.signal(signal.SIGINT, _sigint_handler)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    total_batches = len(loader)

    # Load any existing partial checkpoint
    lists, start_batch = _load_partial(cache_dir, split)

    teacher.eval()
    tokenizer.eval()

    with torch.no_grad():
        for local_idx, (batch_x, batch_x_stamp) in enumerate(
                itertools.islice(loader, start_batch, None)):
            abs_idx = start_batch + local_idx

            batch_x       = batch_x.to(device, non_blocking=True)
            batch_x_stamp = batch_x_stamp.to(device, non_blocking=True)

            gt_s1, gt_s2 = tokenizer.encode(batch_x, half=True)

            token_in_s1 = gt_s1[:, :-1]
            token_in_s2 = gt_s2[:, :-1]
            stamps_in   = batch_x_stamp[:, :-1, :]

            s1_logits, s2_logits = teacher(token_in_s1, token_in_s2, stamps_in)

            if sample_mode == 'argmax':
                pred_s1 = s1_logits.argmax(dim=-1)
                pred_s2 = s2_logits.argmax(dim=-1)
            else:
                pred_s1 = _sample_tokens(s1_logits, temperature=temperature, top_p=top_p)
                pred_s2 = _sample_tokens(s2_logits, temperature=temperature, top_p=top_p)

            lists['s1'].append(pred_s1.cpu())
            lists['s2'].append(pred_s2.cpu())
            lists['gt_s1'].append(gt_s1[:, 1:].cpu())
            lists['gt_s2'].append(gt_s2[:, 1:].cpu())
            lists['stamps'].append(stamps_in.cpu())

            batches_done = abs_idx + 1

            if batches_done % 50 == 0:
                print(f"  Processed {batches_done}/{total_batches} batches")

            # Periodic checkpoint
            if checkpoint_interval > 0 and batches_done % checkpoint_interval == 0:
                _save_partial(cache_dir, split, lists, batches_done, total_batches)

            # Graceful interrupt
            if _stop_requested:
                _save_partial(cache_dir, split, lists, batches_done, total_batches)
                print(f"  Checkpoint saved at batch {batches_done}/{total_batches}")
                print("  Re-run the same command to resume.")
                signal.signal(signal.SIGINT, prev_handler)
                sys.exit(0)

    signal.signal(signal.SIGINT, prev_handler)

    result = {
        's1':     torch.cat(lists['s1'],     dim=0),
        's2':     torch.cat(lists['s2'],     dim=0),
        'gt_s1':  torch.cat(lists['gt_s1'],  dim=0),
        'gt_s2':  torch.cat(lists['gt_s2'],  dim=0),
        'stamps': torch.cat(lists['stamps'], dim=0),
    }

    # Clean up partial checkpoint now that we have the full result
    partial = _partial_ckpt_path(cache_dir, split)
    if os.path.exists(partial):
        os.remove(partial)

    return result


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
    parser.add_argument('--force', action='store_true',
                        help='Regenerate cache even if dataset hash matches')
    args = parser.parse_args()

    config = LargeModelConfig(args.config)

    splits = [s.strip() for s in args.split.split(',')]

    if not args.force and _cache_is_valid(config.distill_cache_dir, config.data_path, splits):
        print("Distillation cache is up to date (dataset hash matches). Nothing to regenerate.")
        print(f"  Cache dir : {config.distill_cache_dir}")
        print("  Use --force to regenerate anyway.")
        return

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

    if args.force:
        for split in splits:
            p = _partial_ckpt_path(config.distill_cache_dir, split)
            if os.path.exists(p):
                os.remove(p)
                print(f"  Removed partial checkpoint for '{split}' (--force)")

    sample_mode = 'sample' if args.sample else config.sample_mode
    print(f"Token generation mode: {sample_mode}")

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
            cache_dir=config.distill_cache_dir,
            split=split,
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

    _write_hash(config.distill_cache_dir, config.data_path)
    print(f"\nDistillation cache generation complete.")
    print(f"  Dataset hash written → {_hash_file(config.distill_cache_dir)}")


if __name__ == '__main__':
    main()

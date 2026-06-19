import torch
from torch.utils.data import Dataset


class DistilledTokenDataset(Dataset):
    """
    Dataset backed by a pre-generated .pt cache from generate_distilled_tokens.py.

    mode='distill'      -> returns teacher-predicted (s1, s2) tokens as targets
    mode='groundtruth'  -> returns ground-truth (s1, s2) tokens from the tokenizer
    """

    def __init__(self, cache_path: str, mode: str = 'distill'):
        if mode not in ('distill', 'groundtruth'):
            raise ValueError(f"mode must be 'distill' or 'groundtruth', got '{mode}'")

        data = torch.load(cache_path, map_location='cpu')

        if mode == 'distill':
            self.s1 = data['s1'].long()
            self.s2 = data['s2'].long()
        else:
            self.s1 = data['gt_s1'].long()
            self.s2 = data['gt_s2'].long()

        stamps = data.get('stamps')
        self.stamps = stamps if stamps is not None else torch.zeros(len(self.s1), self.s1.shape[1], 5)

    def __len__(self):
        return len(self.s1)

    def __getitem__(self, idx):
        return self.s1[idx], self.s2[idx], self.stamps[idx]

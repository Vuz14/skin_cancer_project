import random
import numpy as np
import torch
import math
from typing import List, Optional

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def compute_class_weights(labels: List[int]):
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(labels)
    weights = compute_class_weight('balanced', classes=classes, y=labels)
    w_map = {int(c): float(w) for c, w in zip(classes, weights)}
    sample_weights = np.array([w_map[int(l)] for l in labels], dtype=np.float32)
    return sample_weights, w_map

def get_warmup_cosine_scheduler(optimizer, warmup_epochs, max_epochs, last_epoch=-1):
    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            return float(current_epoch) / float(max(1, warmup_epochs))
        progress = float(current_epoch - warmup_epochs) / float(max(1, max_epochs - warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)

def set_finetune_mode(model: torch.nn.Module, mode: str = 'partial_unfreeze', substrings: Optional[List[str]] = None):
    substrings = substrings or []
    if mode == 'freeze_backbones':
        for name, p in model.named_parameters():
            if name.startswith('backbone'): p.requires_grad = False
            else: p.requires_grad = True
    elif mode == 'partial_unfreeze':
        for _, p in model.named_parameters(): p.requires_grad = False
        for name, p in model.named_parameters():
            if name.startswith('classifier') or 'metadata_mlp' in name or 'emb_layers' in name:
                p.requires_grad = True
        if substrings:
            for name, p in model.named_parameters():
                for s in substrings:
                    if s in name:
                        p.requires_grad = True; break
    elif mode == 'full_unfreeze':
        for _, p in model.named_parameters(): p.requires_grad = True
    else: raise ValueError("Unknown FINE_TUNE_MODE")
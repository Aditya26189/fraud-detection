import random
import numpy as np
import torch
from torch_geometric.datasets import EllipticBitcoinDataset

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_elliptic(root="graphge/data/Elliptic", val_ratio=0.10, seed=0):
    set_seed(seed)
    dataset = EllipticBitcoinDataset(root=root)
    data = dataset[0]

    train_mask = data.train_mask.clone()
    test_mask = data.test_mask.clone()

    assert train_mask.sum() > 0, "Empty train mask"
    assert test_mask.sum() > 0, "Empty test mask"
    assert not (train_mask & test_mask).any(), "Mask overlap"

    train_idx = train_mask.nonzero(as_tuple=False).view(-1)
    perm = train_idx[torch.randperm(train_idx.numel(), generator=torch.Generator().manual_seed(seed))]

    val_size = max(1, int(val_ratio * perm.numel()))
    val_idx = perm[:val_size]
    new_train_idx = perm[val_size:]

    val_mask = torch.zeros_like(train_mask)
    val_mask[val_idx] = True
    train_mask[:] = False
    train_mask[new_train_idx] = True

    y_train = data.y[train_mask]
    counts = torch.bincount(y_train, minlength=2).float().clamp(min=1.0)
    weights = counts.sum() / counts / counts.mean()

    return data, train_mask, val_mask, test_mask, weights
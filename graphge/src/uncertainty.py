import torch
import numpy as np

@torch.no_grad()
def mc_dropout_predict(model, data, mask, T=30):
    model.eval()
    probs_list = []
    for _ in range(T):
        logits = model(data.x, data.edge_index, force_dropout=True)
        probs = torch.exp(logits[mask])
        probs_list.append(probs.cpu())
    probs_T = torch.stack(probs_list, dim=0)
    mean_probs = probs_T.mean(dim=0)
    eps = 1e-12
    entropy = -(mean_probs * torch.log(mean_probs.clamp(min=eps))).sum(dim=1)
    return mean_probs.numpy(), entropy.numpy()
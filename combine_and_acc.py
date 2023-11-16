import os
import numpy as np
import torch
preds = []
targets = []

def top_k_accuracy(pred, targets, k=1):
    # Get the top k indices of the predictions
    top_k_preds = torch.topk(pred, k, dim=1)[1]  # [n, k]

    # Expand targets to [n, k] for comparison
    targets = targets.view(-1, 1).expand_as(top_k_preds)

    # Check if targets are in top k predictions
    correct = torch.any(top_k_preds == targets, dim=1).float().sum()

    return (correct / len(targets)).cpu().item()

dataroot = '/projectnb/ivc-ml/sunxm/code/open_flamingo/snapshots/GPTScore/sun397'
num_splits = 10
for split in range(num_splits):
    pred_path = os.path.join(dataroot, 'pred_%d.npy' % split)
    target_path = os.path.join(dataroot, 'target_%d.npy' % split)
    pred = np.load(pred_path)
    target = np.load(target_path)
    preds.append(pred)
    targets.append(target)

preds = np.concatenate(preds, axis=0)
targets = np.concatenate(targets, axis=0)
preds = torch.from_numpy(preds)
targets = torch.from_numpy(targets)
acc = top_k_accuracy(preds.cpu(), targets, k=1)
print('Top-1 is %0.2f' % (acc*100))
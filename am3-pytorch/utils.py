"""Utility functions and classes for AM3
"""

import os
import shutil

import torch
import torch.nn.functional as F

import wandb

def get_preds(prototypes, embeddings, targets):
    """Compute the accuracy of the prototypical network on the test/query points.
    Params:
    - prototypes (torch.FloatTensor): prototypes for each class (b, N, emb_dim).
    - embeddings (torch.FloatTensor): embeddings of the query points (b, N*K, emb_dim).
    - targets (torch.LongTensor): targets of the query points (b, N*K).
    Returns:
    - preds (torch.LongTensor): predicted classes of the query points (b, N*K)
    - accuracy (torch.FloatTensor): Mean accuracy on the query points.
    """
    # TODO sing a KD tree would be better?
    sq_distances = torch.sum((prototypes.unsqueeze(1)
        - embeddings.unsqueeze(2)) ** 2, dim=-1)
    _, preds = torch.min(sq_distances, dim=-1)
    return preds.detach().cpu().numpy(), torch.mean(preds.eq(targets).float())


def prototypical_loss(prototypes, embeddings, targets, **kwargs):
    """Compute the loss (i.e. negative log-likelihood) for the prototypical 
    network, on the test/query points.
    Parameters:
    - prototypes (torch.FloatTensor): prototypes for each class (b, N, emb_dim).
    - embeddings (torch.FloatTensor): embeddings of the query points (b, N*K, emb_dim).
    - targets (torch.LongTensor): targets of the query points (b, N*K).
    Returns:
    - loss (torch.FloatTensor): The negative log-likelihood on the query points.
    """
    squared_distances = torch.sum((prototypes.unsqueeze(2)
        - embeddings.unsqueeze(1)) ** 2, dim=-1)
    return F.cross_entropy(-squared_distances, targets, **kwargs)


class AverageMeter(object):
    """Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# save/load checkpoints
def save_checkpoint(checkpoint_dict: dict, is_best: bool):
    """Saves a model checkpoint to file. Keeps most recent and best model.
    Params:
    - checkpoint_dict (dict): dict containing all model state info, to pickle
    - is_best (bool): whether this checkpoint is the best seen so far.
    """
    checkpoint_file = os.path.join(wandb.run.dir, "ckpt.pth.tar")
    best_file = os.path.join(wandb.run.dir, "best.pth.tar")
    torch.save(checkpoint_dict, checkpoint_file)
    
    if is_best:
        shutil.copyfile(checkpoint_file, best_file)


def load_checkpoint(model, optimizer, device, checkpoint_file: str):
    """Loads a model checkpoint.
    Params:
    - model (nn.Module): instantised model to load weights into
    - optimizer (nn.optim): instantised optimizer to load state into
    - device (torch.device): device the model is on.
    - checkpoint_file (str): path to checkpoint to load.
    Returns:
    - model with loaded state dict
    - optimizer with loaded state dict
    """
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    print(f"Loaded {checkpoint_file}, "
          f"trained to epoch {checkpoint['batch_idx']} with best loss {checkpoint['best_loss']}")

    return model, optimizer

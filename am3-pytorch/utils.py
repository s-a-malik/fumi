"""Utility functions and classes for AM3
"""

import os
import shutil

import torch
import wandb

def get_preds(prototypes, embeddings, targets):
    """CREDIT: https://github.com/tristandeleu/pytorch-meta/blob/master/examples/protonet
    Compute the accuracy of the prototypical network on the test/query points.
    Parameters
    ----------
    prototypes : `torch.FloatTensor` instance
        A tensor containing the prototypes for each class. This tensor has shape 
        `(meta_batch_size, num_classes, embedding_size)`.
    embeddings : `torch.FloatTensor` instance
        A tensor containing the embeddings of the query points. This tensor has 
        shape `(meta_batch_size, num_examples, embedding_size)`.
    targets : `torch.LongTensor` instance
        A tensor containing the targets of the query points. This tensor has 
        shape `(meta_batch_size, num_examples)`.
    Returns
    -------
    accuracy : `torch.FloatTensor` instance
        Mean accuracy on the query points.
    """
    # using a KD tree would be better

    sq_distances = torch.sum((prototypes.unsqueeze(1)
        - embeddings.unsqueeze(2)) ** 2, dim=-1)
    _, preds = torch.min(sq_distances, dim=-1)
    return preds.detach().cpu().numpy(), torch.mean(preds.eq(targets).float())


def prototypical_loss(prototypes, embeddings, targets, **kwargs):
    """Compute the loss (i.e. negative log-likelihood) for the prototypical 
    network, on the test/query points.
    Parameters
    ----------
    prototypes : `torch.FloatTensor` instance
        A tensor containing the prototypes for each class. This tensor has shape 
        `(batch_size, num_classes, embedding_size)`.
    embeddings : `torch.FloatTensor` instance
        A tensor containing the embeddings of the query points. This tensor has 
        shape `(batch_size, num_examples, embedding_size)`.
    targets : `torch.LongTensor` instance
        A tensor containing the targets of the query points. This tensor has 
        shape `(batch_size, num_examples)`.
    Returns
    -------
    loss : `torch.FloatTensor` instance
        The negative log-likelihood on the query points.
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
    - 
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

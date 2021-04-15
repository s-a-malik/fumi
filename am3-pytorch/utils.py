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


def get_prototypes(im_embeddings, text_embeddings, lamdas, targets, num_classes):
    """Compute the prototypes (the mean vector of the embedded training/support 
    points belonging to its class) for each classes in the task.
    Params:
    - im_embeddings (torch.FloatTensor): image embeddings of the support points
    (b, N*K, emb_dim).
    - text_embeddings (torch.FloatTensor): text embeddings of the support points
    (b, N*K, emb_dim).
    - lamda (torch.FloatTensor): weighting of text for the prototype (b, N*K, 1).
    targets (torch.LongTensor): targets of the support points (b, N*K).
    num_classes (int): Number of classes in the task.
    Returns:
    - prototypes (torch.FloatTensor): prototypes for each class (b, N, emb_dim).
    """
    batch_size, embedding_size = im_embeddings.size(0), im_embeddings.size(-1)

    # num_samples common across all computations
    num_samples = get_num_samples(targets, num_classes, dtype=im_embeddings.dtype)
    num_samples.unsqueeze_(-1)  # (b x N x 1)
    num_samples = torch.max(num_samples, torch.ones_like(num_samples))      # prevents zero division error
    indices = targets.unsqueeze(-1).expand_as(im_embeddings)                # (b x N*K x 512)

    im_prototypes = im_embeddings.new_zeros((batch_size, num_classes, embedding_size))
    im_prototypes.scatter_add_(1, indices, im_embeddings).div_(num_samples)   # compute mean embedding of each class

    # should all be equal anyway (checked)
    text_prototypes = text_embeddings.new_zeros((batch_size, num_classes, embedding_size))
    text_prototypes.scatter_add_(1, indices, text_embeddings).div_(num_samples)

    # should all be equal (checked)
    lamdas_per_class = lamdas.new_zeros((batch_size, num_classes, 1))
    lamdas_per_class.scatter_add_(1, targets.unsqueeze(-1), lamdas).div_(num_samples)

    # convex combination
    prototypes = lamdas_per_class * im_prototypes + (1-lamdas_per_class) * text_prototypes
    return prototypes


def get_num_samples(targets, num_classes, dtype=None):
    """Returns a vector with the number of samples in each class.
    """
    batch_size = targets.size(0)
    with torch.no_grad():
        ones = torch.ones_like(targets, dtype=dtype)
        num_samples = ones.new_zeros((batch_size, num_classes))
        num_samples.scatter_add_(1, targets, ones)
    return num_samples


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

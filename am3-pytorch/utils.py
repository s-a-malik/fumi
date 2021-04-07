"""Utility functions and classes for AM3
"""

import torch


def get_accuracy(prototypes, embeddings, targets):
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
    _, predictions = torch.min(sq_distances, dim=-1)
    return torch.mean(predictions.eq(targets).float())

def am3_loss_fn()


# save/load checkpoints
def save_checkpoint(checkpoint_dict: dict, is_best: bool, checkpoint_file: str, best_file: str):
    """Saves a model checkpoint. Keeps most recent and best model.
    Params:
    - checkpoint_dict (dict): dict containing all model state info, to pickle
    - is_best (bool): whether this checkpoint is the best seen so fat
    - checkpoint_file (str): file to save most recent checkpoint to
    - best_file (str): file to save best checkpoint to 
    """
    raise NotImplementedError()

def load_checkpoint(model, checkpoint_file: str):
    """Loads a model checkpoint.
    """
    raise NotImplementedError()

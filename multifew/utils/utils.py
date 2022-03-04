"""Utility functions and classes for AM3
"""

import os
import shutil
import wandb
import argparse
from models import am3, maml, fumi, clip
import numpy as np

from transformers import AdamW, get_linear_schedule_with_warmup

import torch
import torch.nn.functional as F

from sklearn.metrics import precision_recall_fscore_support, accuracy_score


def parser():
    parser = argparse.ArgumentParser(
        description="Multimodal image classification")

    # data config
    parser.add_argument("--dataset",
                        type=str,
                        default="zanim",
                        help="Dataset to use")
    parser.add_argument("--data_dir",
                        type=str,
                        default="./data",
                        help="Directory to use for data")
    parser.add_argument(
        "--json_path",
        type=str,
        default="train.json",
        help="Location of the json file containing dataset annotations")
    parser.add_argument("--checkpoint",
                        type=str,
                        default=None,
                        help="Path to pretrained model")
    parser.add_argument("--log_dir",
                        type=str,
                        default="./am3",
                        help="Directory to use for logs and checkpoints")
    parser.add_argument('--remove_stop_words',
                        action='store_true',
                        help="Whether to remove stop words")
    parser.add_argument('--colab',
                        action='store_true',
                        help="Whether the script is running on Google Colab")

    # optimizer config
    parser.add_argument("--epochs",
                        type=int,
                        default=50000,
                        help="Number of meta-learning batches to train for")
    parser.add_argument("--optim", type=str, default="adam", help="Optimiser")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--momentum",
                        type=float,
                        default=0.9,
                        help="Momentum for SGD")
    parser.add_argument("--batch_size",
                        type=int,
                        default=4,
                        help="Number of tasks in mini-batch")
    parser.add_argument("--weight_decay",
                        type=float,
                        default=5e-4,
                        help="L2 regulariser")
    parser.add_argument("--num_warmup_steps",
                        type=float,
                        default=10,
                        help="Warm up lr scheduler")

    # dataloader config
    parser.add_argument("--num_shots",
                        type=int,
                        default=3,
                        help="Number of examples per class (k-shot)")
    parser.add_argument("--num_ways",
                        type=int,
                        default=5,
                        help="Number of classes per task (N-way)")
    parser.add_argument("--num_shots_test",
                        type=int,
                        default=32,
                        help="Number of examples per class in query set")
    parser.add_argument("--augment",
                        action="store_true",
                        help="Augment data with image transformations")
    parser.add_argument("--num_workers",
                        type=int,
                        default=0,
                        help="Number of workers for dataloader")
    parser.add_argument(
        "--image_embedding_model",
        type=str,
        default="resnet-152",
        help=
        "resnet-152 embedding (2048 dimensions) or resnet-34 (512 dimensions)")

    # model config
    parser.add_argument("--model",
                        type=str,
                        default="fumi",
                        help="Model to be trained")
    parser.add_argument("--prototype_dim",
                        type=int,
                        default=64,
                        help="Dimension of latent space")
    parser.add_argument(
        "--im_encoder",
        type=str,
        default="precomputed",
        help="Type of vision feature extractor (resnet, precomputed)")
    parser.add_argument("--im_emb_dim",
                        type=int,
                        default=2048,
                        help="Dimension of image embedding (if precomputed)")
    parser.add_argument("--im_hid_dim",
                        type=int,
                        default=64,
                        help="Hidden dimension of image model")
    parser.add_argument(
        "--text_encoder",
        type=str,
        choices=['glove', 'w2v', 'RNN', 'RNNhid', 'BERT', 'rand'],
        default="BERT",
        help="Type of text embedding (glove, w2v, RNN, RNNhid, BERT, rand)")
    parser.add_argument(
        "--pooling_strat",
        type=str,
        default="mean",
        help="Pooling strategy if using word embeddings (mean, max)")
    parser.add_argument("--fine_tune",
                        action="store_true",
                        help="Whether to fine tune text encoder")
    parser.add_argument(
        "--text_type",
        type=str,
        nargs="+",
        default=["description"],
        help=
        "What to use for text embedding (label, description or common_name) can take multiple arguments (appends the different text types) e.g. --text_type label description)"
    )
    parser.add_argument("--text_emb_dim",
                        type=int,
                        default=768,
                        help="Dimension of text embedding (if precomputed)")
    parser.add_argument(
        "--text_hid_dim",
        type=int,
        default=256,
        help="Hidden dimension for NN mapping to prototypes and lamda")
    parser.add_argument("--dropout",
                        type=float,
                        default=0.7,
                        help="Dropout rate")
    parser.add_argument("--step_size",
                        type=float,
                        default=0.01,
                        help="MAML step size")
    parser.add_argument("--first_order",
                        action="store_true",
                        help="Whether to use first-order MAML")
    parser.add_argument(
        "--num_train_adapt_steps",
        type=int,
        default=5,
        help="Number of MAML inner train loop adaptation steps")
    parser.add_argument("--num_test_adapt_steps",
                        type=int,
                        default=15,
                        help="Number of MAML inner test loop adaptation steps")
    parser.add_argument("--init_all_layers",
                        action="store_true",
                        help="Whether to initialise all (vs. last) layer weights in FUMI")
    parser.add_argument("--norm_hypernet",
                        action="store_true",
                        help="Whether to normalize output of the FUMI hypernetwork (tanh)")

    # clip config
    parser.add_argument("--clip_latent_dim",
                        type=int,
                        default=512,
                        help="Dimension of CLIP latent space")

    # run config
    parser.add_argument("--seed", type=int, default=123, help="patience for early stopping")
    parser.add_argument("--patience",
                        type=int,
                        default=10000,
                        help="Early stopping patience")
    parser.add_argument(
        "--eval_freq",
        type=int,
        default=500,
        help="Number of batches between validation/checkpointing")
    parser.add_argument("--experiment",
                        type=str,
                        default="debug",
                        help="Name for experiment (for wandb group)")
    parser.add_argument("--evaluate",
                        action="store_true",
                        help="skip training")
    parser.add_argument(
        "--num_ep_test",
        type=int,
        default=200,
        help="Number of few-shot episodes to compute test accuracy")
    parser.add_argument("--disable_cuda",
                        action="store_true",
                        help="don't use GPU")
    return parser


def init_model(args, dictionary, watch=True):
    """Initialise model
    """
    model = None
    if args.model == "maml":
        model = maml.PureImageNetwork(im_embed_dim=args.im_emb_dim,
                                      n_way=args.num_ways,
                                      hidden=args.im_hid_dim)
    elif args.model == "fumi":
        model = fumi.FUMI(n_way=args.num_ways,
                          im_emb_dim=args.im_emb_dim,
                          im_hid_dim=args.im_hid_dim,
                          text_encoder=args.text_encoder,
                          text_emb_dim=args.text_emb_dim,
                          text_hid_dim=args.text_hid_dim,
                          dictionary=dictionary,
                          pooling_strat=args.pooling_strat,
                          init_all_layers=args.init_all_layers,
                          norm_hypernet=args.norm_hypernet,
                          fine_tune=args.fine_tune)
    elif args.model == "clip":
        model = clip.CLIP(text_input_dim=args.text_emb_dim,
                          image_input_dim=args.im_emb_dim,
                          latent_dim=args.clip_latent_dim)
    else:
        model = am3.AM3(im_encoder=args.im_encoder,
                        im_emb_dim=args.im_emb_dim,
                        text_encoder=args.text_encoder,
                        text_emb_dim=args.text_emb_dim,
                        text_hid_dim=args.text_hid_dim,
                        prototype_dim=args.prototype_dim,
                        dropout=args.dropout,
                        fine_tune=args.fine_tune,
                        dictionary=dictionary,
                        pooling_strat=args.pooling_strat)

    if watch:
        wandb.watch(model, log="all")  #  for tracking gradients etc.
    model.to(args.device)
    return model


def init_optim(args, model):
    """Initialise optimizer
    """
    if args.optim == "adam":
        optimizer = torch.optim.Adam(params=model.parameters(),
                                     lr=args.lr,
                                     weight_decay=args.weight_decay)
    elif args.optim == "SGD":
        optimizer = torch.optim.SGD(params=model.parameters(),
                                    lr=args.lr,
                                    weight_decay=args.weight_decay,
                                    momentum=args.momentum)
    elif args.optim == "adamw":
        optimizer = AdamW(params=model.parameters(), lr=args.lr)
    elif args.optim == "adamw_lin_schedule":
        opt = AdamW(params=model.parameters(), lr=args.lr)
        scheduler = get_linear_schedule_with_warmup(opt, args.num_warmup_steps,
                                                    args.epochs)
        optimizer = (opt, scheduler)
    else:
        raise NotImplementedError()

    return optimizer


def get_preds(prototypes, embeddings, targets):
    """Compute the accuracy of the prototypical network on the test/query points.
    Params:
    - prototypes (torch.FloatTensor): prototypes for each class (b, N, emb_dim).
    - embeddings (torch.FloatTensor): embeddings of the query points (b, N*K, emb_dim).
    - targets (torch.LongTensor): targets of the query points (b, N*K).
    Returns:
    - preds (np.array): predicted classes of the query points (b, N*K)
    - accuracy (float): Mean accuracy on the query points.
    - f1 (float): f1 score
    - prec (float): precision
    - rec (float): recall
    """
    sq_distances = torch.sum(
        (prototypes.unsqueeze(1) - embeddings.unsqueeze(2))**2, dim=-1)
    _, preds = torch.min(sq_distances, dim=-1)

    # compute f1, prec, recall, acc
    preds = preds.detach().cpu().numpy()
    flat_preds = np.reshape(preds, -1)
    flat_targets = np.reshape(targets.detach().cpu().numpy(), -1)
    acc = accuracy_score(flat_targets, flat_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(flat_targets,
                                                       flat_preds,
                                                       average="macro")

    return preds, acc, f1, prec, rec


def get_prototypes(im_embeddings, text_embeddings, lamdas, targets,
                   num_classes):
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
    num_samples = get_num_samples(targets,
                                  num_classes,
                                  dtype=im_embeddings.dtype)
    num_samples.unsqueeze_(-1)  # (b x N x 1)
    num_samples = torch.max(
        num_samples,
        torch.ones_like(num_samples))  # prevents zero division error
    indices = targets.unsqueeze(-1).expand_as(im_embeddings)  # (b x N*K x 512)

    im_prototypes = im_embeddings.new_zeros(
        (batch_size, num_classes, embedding_size))
    im_prototypes.scatter_add_(1, indices, im_embeddings).div_(
        num_samples)  # compute mean embedding of each class

    # should all be equal anyway (checked)
    text_prototypes = text_embeddings.new_zeros(
        (batch_size, num_classes, embedding_size))
    text_prototypes.scatter_add_(1, indices, text_embeddings).div_(num_samples)

    # should all be equal (checked)
    lamdas_per_class = lamdas.new_zeros((batch_size, num_classes, 1))
    lamdas_per_class.scatter_add_(1, targets.unsqueeze(-1),
                                  lamdas).div_(num_samples)

    # convex combination
    prototypes = lamdas_per_class * im_prototypes + (
        1 - lamdas_per_class) * text_prototypes
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
    squared_distances = torch.sum(
        (prototypes.unsqueeze(2) - embeddings.unsqueeze(1))**2, dim=-1)
    return F.cross_entropy(-squared_distances, targets, **kwargs)


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
    wandb.save(checkpoint_file)

    if is_best:
        shutil.copyfile(checkpoint_file, best_file)
        wandb.save(best_file)


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
    print(
        f"Loaded {checkpoint_file}, "
        f"trained to epoch {checkpoint['batch_idx']} with best loss {checkpoint['best_loss']}"
    )

    return model, optimizer

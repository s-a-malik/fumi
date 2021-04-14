"""AM3 in Pytorch.
"""

import os
import sys
import argparse
from tqdm.autonotebook import tqdm
import wandb

import numpy as np
import pandas as pd

import torch

from model import AM3
from data import get_dataset
import utils


def main(args):
    # random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # TODO dataloader random seeding is special - if using augmentations etc. need to be careful

    # set up directories and logs
    model_path = f"{args.log_dir}/models"       # models are saved to wandb run, this is local storage for restoring
    results_path = f"{args.log_dir}/results"
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(results_path, exist_ok=True)
    os.environ['WANDB_DISABLE_CODE'] = "true"         # disable code logging
    run = wandb.init(entity="multimodal-image-cls", 
                     project="am3",
                     group=args.experiment)
    wandb.config.update(args)

    # load datasets
    train_loader, val_loader, test_loader, dictionary = get_dataset(args)
    # TODO fix this to give exactly 1000. Change in dataloader probs.
    max_test_batches = int(args.num_ep_test/args.batch_size)

    # initialise model and optim
    model = init_model(args, dictionary)
    optimizer = init_optim(args, model)

    # load previous state
    if args.checkpoint:
        # restore from wandb
        checkpoint_file = wandb.restore(
            "ckpt.pth.tar",
            run_path=f"multimodal-image-cls/am3/{args.checkpoint}",
            root=model_path)
        # load state dict
        model, optimizer = utils.load_checkpoint(model, optimizer, args.device, checkpoint_file.name)

    # skip training if just testing
    if not args.evaluate:
        # get best val loss
        best_loss, best_acc, _, _, _, _ = test_loop(model, val_loader, max_test_batches)
        print(f"initial loss: {best_loss}, acc: {best_acc}")
        best_batch_idx = 0
        
        # use try, except to be able to stop partway through training
        try:
            # Training loop
            # do in epochs with a max_num_batches instead?
            # for batch_idx, batch in enumerate(tqdm(train_loader, total=args.epochs)):
            for batch_idx, batch in enumerate(train_loader):
                # TODO make this into an evaluate function
                train_loss, train_acc = model.evaluate(
                    batch=batch,
                    optimizer=optimizer,
                    num_ways=args.num_ways,
                    device=args.device,
                    task="train")

                # log
                # pbar.set_postfix(accuracy='{0:.4f}'.format(train_acc.item()))
                wandb.log({"train/acc": train_acc,
                            "train/loss": train_loss}, step=batch_idx)

                # eval on validation set periodically
                if batch_idx % 100 == 0:
                    # evaluate on val set
                    val_loss, val_acc, _, _, _, _ = test_loop(
                        model, val_loader, max_test_batches)
                    is_best = val_loss < best_loss
                    if is_best:
                        best_loss = val_loss
                        best_batch_idx = batch_idx
                    # TODO could log examples
                    wandb.log({"val/acc": val_acc,
                                "val/loss": val_loss}, step=batch_idx)

                    # save checkpoint
                    checkpoint_dict = {
                        "batch_idx": batch_idx,
                        "state_dict": model.state_dict(),
                        "best_loss": best_loss,
                        "optimizer": optimizer.state_dict(),
                        "args": vars(args)
                    }
                    utils.save_checkpoint(checkpoint_dict, is_best)
                    # TODO save example outputs?

                    print(f"Batch {batch_idx+1}/{args.epochs}: \ntrain/loss: {train_loss}, train/acc: {train_acc}"
                            f"\nval/loss: {val_loss}, val/acc: {val_acc}")

                # break after max iters or early stopping
                if (batch_idx > args.epochs - 1) or (batch_idx - best_batch_idx > args.patience):
                    break
        except KeyboardInterrupt():
            pass
    
    # test
    test_loss, test_acc, test_preds, test_true, test_idx, task_idx = test_loop(
        model, test_loader, max_test_batches)
    print(f"test loss: {test_loss}, test acc: {test_acc}")
    
    # TODO more metrics - F1, precision, recall etc.

    # save results
    wandb.log({
        "test/acc": test_acc,
        "test/loss": test_loss}, step=batch_idx)
    df = pd.DataFrame({
        "image_idx": test_idx,
        "task_idx": task_idx,
        "preds": test_preds,
        "targets": test_true})
    df.to_csv(path_or_buf=f"{results_path}/run_{wandb.run.name}")

    wandb.finish()


def init_model(args, dictionary):
    """Initialise model
    """
    model = AM3(
        im_encoder=args.im_encoder,
        im_emb_dim=args.im_emb_dim,
        text_encoder=args.text_encoder,
        text_emb_dim=args.text_emb_dim, 
        text_hid_dim=args.text_hid_dim,
        prototype_dim=args.prototype_dim,
        dropout=args.dropout,
        fine_tune=args.fine_tune,
        dictionary=dictionary
    )

    wandb.watch(model, log="all")   # for tracking gradients etc.
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
    else:
        raise NotImplementedError()

    return optimizer


def test_loop(model, test_dataloader, max_num_batches):
    """Evaluate model on val/test set.
    Test on 1000 randomly sampled tasks, each with 100 query samples (as in AM3)
    Returns:
    - avg_test_acc (float): average test accuracy per task
    - avg_test_loss (float): average test loss per task
    """
    # TODO add more test metrics + example outputs?
    # TODO need to fix number of tasks/episodes etc. depending on batch, num_ways etc.

    avg_test_acc = utils.AverageMeter()
    avg_test_loss = utils.AverageMeter()
    test_preds = []
    test_trues = []
    test_idx = []
    task_idx = []
    
    for batch_idx, batch in enumerate(tqdm(test_dataloader, total=max_num_batches)): 
        test_loss, test_acc, preds, trues, idx = model.evaluate(
            batch=batch,
            optimizer=None,
            num_ways=args.num_ways,
            device=args.device,
            task="test")

        avg_test_acc.update(test_acc)
        avg_test_loss.update(test_loss)
        test_preds += preds.tolist()
        test_trues += trues.tolist()
        test_idx += idx.tolist()
        # TODO fix to get tasks not batches
        task_idx.append([batch_idx for i in range(len(idx))])

        if batch_idx > max_num_batches - 1:
            break
                    
    return avg_test_loss.avg, avg_test_acc.avg, test_preds, test_trues, test_idx, task_idx


def parse_args():
    """experiment arguments
    """

    parser = argparse.ArgumentParser(description="AM3")

    # data config
    parser.add_argument("--dataset",
                        type=str,
                        default="zanim",
                        help="Dataset to use")
    parser.add_argument("--data_dir",
                        type=str,
                        default="./data",
                        help="Directory to use for data")
    parser.add_argument("--checkpoint",
                        type=str,
                        default=None,
                        help="path to pretrained model")
    parser.add_argument("--log_dir",
                        type=str,
                        default="./am3",
                        help="Directory to use for logs and checkpoints")      
    parser.add_argument('--remove_stop_words',
                        action='store_true',
                        help="whether to remove stop words")
      

    # optimizer config
    parser.add_argument("--epochs",
                        type=int,
                        default=5000,
                        help="Number of meta-learning batches to train for")
    parser.add_argument("--optim",
                        type=str,
                        default="adam",
                        help="optimiser")                       
    parser.add_argument("--lr",
                        type=float,
                        default=1e-4,
                        help="learning rate")
    parser.add_argument("--momentum",
                        type=float,
                        default=0.9,
                        help="momentum for SGD")
    parser.add_argument("--batch_size",
                        type=int,
                        default=5,
                        help="number of tasks in mini-batch")
    parser.add_argument("--weight_decay",
                        type=float,
                        default=0.0005,
                        help="L2 regulariser")
    
    # dataloader config
    parser.add_argument("--num_shots",
                        type=int,
                        default=5,
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
                        help="augment data with image transformations")
    parser.add_argument("--num_workers",
                        type=int,
                        default=0,
                        help="Number of workers for dataloader")

    # model config
    parser.add_argument("--prototype_dim",
                        type=int,
                        default=64,
                        help="Dimension of latent space")
    parser.add_argument("--im_encoder",
                        type=str,
                        default="precomputed",
                        help="Type of vision feature extractor (resnet, precomputed)")
    parser.add_argument("--im_emb_dim",
                        type=int,
                        default=512,
                        help="Dimension of image embedding (if precomputed)")
    parser.add_argument("--text_encoder",
                        type=str,
                        default="glove",
                        help="Type of text embedding (glove, BERT)")
    parser.add_argument("--fine_tune",
                        action="store_true",
                        help="whether to fine tune text encoder")
    parser.add_argument("--text_type",
                        type=str,
                        default="label",
                        help="What to use for text embedding (label or description)")
    parser.add_argument("--text_emb_dim",
                        type=int,
                        default=512,
                        help="Dimension of text embedding (if precomputed)")
    parser.add_argument("--text_hid_dim",
                        type=int,
                        default=300,
                        help="Hidden dimension for NN mapping to prototypes and lamda")
    parser.add_argument("--dropout",
                        type=float,
                        default=0.7,
                        help="dropout rate")

    # run config
    parser.add_argument("--seed",
                        type=int,
                        default=123,
                        help="Random seed")    
    parser.add_argument("--patience",
                        type=int,
                        default=100,
                        help="Early stopping patience")   
    parser.add_argument("--experiment",
                        type=str,
                        default="debug",
                        help="Name for experiment (for wandb group)")
    parser.add_argument("--evaluate",
                        action="store_true",
                        help="skip training")
    parser.add_argument("--num_ep_test",
                        type=int,
                        default=1000,
                        help="Number of few-shot episodes to compute test accuracy")
    parser.add_argument("--disable_cuda",
                        action="store_true",
                        help="don't use GPU")

    args = parser.parse_args(sys.argv[1:])

    args.device = torch.device("cuda") if (not args.disable_cuda) and \
        torch.cuda.is_available() else torch.device("cpu")

    return args


if __name__ == "__main__":
    # read cmd line arguments
    args = parse_args()
    # run experiment
    main(args)

import os
import sys
import argparse
import wandb

import numpy as np
import pandas as pd

import torch

import models.am3 as am3
import models.maml as maml
import models.fumi as fumi
from dataset.data import get_dataset
import utils


def main(args):
    # random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # TODO dataloader random seeding is special - if using augmentations etc. need to be careful

    # set up directories and logs
    model_path = "./checkpoints"       # models are saved to wandb run, this is local storage for restoring
    results_path = f"{args.log_dir}/results"
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(results_path, exist_ok=True)
    os.environ["GENSIM_DATA_DIR"] = f"{args.log_dir}/word_embeddings"   # TODO changing the dir doesn't seem to work on colab
    job_type = "eval" if args.evaluate else "train"
    run = wandb.init(entity="multimodal-image-cls",
                     project=args.model,
                     group=args.experiment,
                     job_type=job_type,
                     save_code=True)
    wandb.config.update(args)

    # load datasets
    train_loader, val_loader, test_loader, dictionary = get_dataset(args)
    # TODO fix this to give exactly 1000 episodes. Change in test dataloader probs.
    max_test_batches = int(args.num_ep_test/args.batch_size)

    # initialise model and optim
    model = init_model(args, dictionary)
    optimizer = init_optim(args, model)

    # load previous state
    if args.checkpoint:
        # restore from wandb
        checkpoint_file = wandb.restore(
            "ckpt.pth.tar",
            run_path=f"multimodal-image-cls/{args.model}/{args.checkpoint}",
            root=model_path)
        # load state dict
        model, optimizer = utils.load_checkpoint(model, optimizer, args.device, checkpoint_file.name)

    # skip training if just testing
    if not args.evaluate:
        if args.model == "maml":
            model = maml.training_run(args, model, optimizer, train_loader, val_loader, max_test_batches)
        if args.model == "fumi":
            model = fumi.training_run(args, model, optimizer, train_loader, val_loader, max_test_batches)
        else:
            model = am3.training_run(args, model, optimizer, train_loader, val_loader, max_test_batches)

    # test
    if args.model in ["maml", "fumi"]:
        if args.model == "maml":
            test_loss, test_acc = maml.test_loop(
                args, model, test_loader, max_test_batches)
        else:
            test_loss, test_acc = fumi.test_loop(
                args, model, test_loader, max_test_batches)
        print(
            f"\n TEST: \ntest loss: {test_loss}, test acc: {test_acc}")

        wandb.log({
            "test/acc": test_acc,
            "test/loss": test_loss})
    else:
        test_loss, test_acc, test_avg_lamda, test_preds, test_true, query_idx, support_idx, support_lamda = am3.test_loop(
            args, model, test_loader, max_test_batches)
        print(f"\n TEST: \ntest loss: {test_loss}, test acc: {test_acc}, test avg lamda: {test_avg_lamda}")
        # TODO more metrics - F1, precision, recall etc.

        wandb.log({
            "test/acc": test_acc,
            "test/loss": test_loss,
            "test/avg_lamda": test_avg_lamda})

        # save results
        df = pd.DataFrame({
            "support_idx": support_idx,
            "support_lamda": support_lamda,
            "query_idx": query_idx,
            "query_preds": test_preds,
            "query_targets": test_true})
        df.to_csv(path_or_buf=f"{results_path}/run_{wandb.run.name}.csv")
    
    wandb.finish()


def init_model(args, dictionary):
    """Initialise model
    """
    if args.model == "maml":
        model = maml.PureImageNetwork(
            im_embed_dim=args.im_emb_dim,
            n_way=args.num_ways,
            hidden=args.im_hid_dim
        )
    if args.model == "fumi":
        model = fumi.FUMI(
            n_way=args.num_ways,
            im_emb_dim=args.im_emb_dim,
            im_hid_dim=args.im_hid_dim,
            text_encoder=args.text_encoder,
            text_emb_dim=args.text_emb_dim,
            text_hid_dim=args.text_hid_dim,
            dictionary=dictionary,
            pooling_strat=args.pooling_strat
        )
    else:
        model = am3.AM3(
            im_encoder=args.im_encoder,
            im_emb_dim=args.im_emb_dim,
            text_encoder=args.text_encoder,
            text_emb_dim=args.text_emb_dim, 
            text_hid_dim=args.text_hid_dim,
            prototype_dim=args.prototype_dim,
            dropout=args.dropout,
            fine_tune=args.fine_tune,
            dictionary=dictionary,
            pooling_strat=args.pooling_strat
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


def parse_args():
    """experiment arguments
    """

    parser = argparse.ArgumentParser(description="Multimodal image classification")

    # data config
    parser.add_argument("--dataset",
                        type=str,
                        default="zanim",
                        help="Dataset to use")
    parser.add_argument("--data_dir",
                        type=str,
                        default="./data",
                        help="Directory to use for data")
    parser.add_argument("--json_path",
                        type=str,
                        default="train.json",
                        help="Location of the json file containing dataset annotations"    
    )
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
      

    # optimizer config
    parser.add_argument("--epochs",
                        type=int,
                        default=5000,
                        help="Number of meta-learning batches to train for")
    parser.add_argument("--optim",
                        type=str,
                        default="adam",
                        help="Optimiser")                       
    parser.add_argument("--lr",
                        type=float,
                        default=1e-4,
                        help="Learning rate")
    parser.add_argument("--momentum",
                        type=float,
                        default=0.9,
                        help="Momentum for SGD")
    parser.add_argument("--batch_size",
                        type=int,
                        default=5,
                        help="Number of tasks in mini-batch")
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
                        help="Augment data with image transformations")
    parser.add_argument("--num_workers",
                        type=int,
                        default=0,
                        help="Number of workers for dataloader")

    # model config
    parser.add_argument("--model",
                        type=str,
                        default="am3",
                        help="Model to be trained")
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
                        default=2048,
                        help="Dimension of image embedding (if precomputed)")
    parser.add_argument("--im_hid_dim",
                        type=int,
                        default=64,
                        help="Hidden dimension of image model")
    parser.add_argument("--text_encoder",
                        type=str,
                        default="BERT",
                        help="Type of text embedding (glove, w2v, RNN, BERT, rand)")
    parser.add_argument("--pooling_strat",
                        type=str,
                        default="mean",
                        help="Pooling strategy if using word embeddings (mean, max)")
    parser.add_argument("--fine_tune",
                        action="store_true",
                        help="Whether to fine tune text encoder")
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
                        help="Dropout rate")
    parser.add_argument("--step_size",
                        type=float,
                        default=0.5,
                        help="MAML step size")
    parser.add_argument("--first_order",
                        action="store_true",
                        help="Whether to use first-order MAML")
    parser.add_argument("--num_train_adapt_steps",
                        type=int,
                        default=1,
                        help="Number of MAML inner train loop adaptation steps")
    parser.add_argument("--num_test_adapt_steps",
                        type=int,
                        default=1,
                        help="Number of MAML inner test loop adaptation steps")


    # run config
    parser.add_argument("--seed",
                        type=int,
                        default=123,
                        help="Random seed")    
    parser.add_argument("--patience",
                        type=int,
                        default=100,
                        help="Early stopping patience")   
    parser.add_argument("--eval_freq",
                        type=int,
                        default=20,
                        help="Number of batches between validation/checkpointing")  
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

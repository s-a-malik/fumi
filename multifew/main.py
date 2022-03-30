import os
import sys
import argparse
import wandb
import random

import numpy as np
import pandas as pd

import torch

import models.am3 as am3
import models.maml as maml
import models.fumi as fumi
import models.clip as clip
from dataset.data import get_dataset
import utils.utils as utils


def main(args):

    # set up directories and logs
    results_path = f"{args.log_dir}/results"
    os.makedirs(results_path, exist_ok=True)
    # TODO changing the dir doesn't seem to work on colab
    os.environ["GENSIM_DATA_DIR"] = f"{args.log_dir}/word_embeddings"
    job_type = "eval" if args.evaluate else "train"
    run = wandb.init(entity=args.wandb_entity,
                     project=args.model,
                     group=args.experiment,
                     job_type=job_type,
                     save_code=True)
    wandb.config.update(args)

    if args.image_embedding_model not in ["resnet-152", 'resnet-34']:
        raise ValueError(
            "Image embedding model must be one of resnet-152 resnet-34")
    if args.image_embedding_model == "resnet-152" and args.im_emb_dim != 2048:
        raise ValueError(
            "Resnet-152 outputs 2048-dimensional embeddings, hence --im_emb_dim should be set to 2048"
        )
    if args.image_embedding_model == "resnet-34" and args.im_emb_dim != 512:
        raise ValueError(
            "Resnet-34 outputs 512-dimensional embeddings, hence --im_emb_dim should be set to 512"
        )

    # load datasets
    train_loader, val_loader, test_loader, dictionary = get_dataset(args)
    # TODO fix this to give exactly 1000 episodes. Change in test dataloader probs.
    max_test_batches = int(args.num_ep_test / args.batch_size)

    # random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    # TODO dataloader random seeding is special - if using augmentations etc. need to be careful


    # initialise model and optim
    model = utils.init_model(args, dictionary)
    print(model)
    optimizer = utils.init_optim(args, model)

    # load previous state
    if args.checkpoint:
        model_path = f"./checkpoints/{args.model}/{args.checkpoint}"
        os.makedirs(model_path, exist_ok=True)
        # restore from wandb
        # checkpoint_file = wandb.restore(
        #     "ckpt.pth.tar",
        #     run_path=f"multimodal-image-cls/{args.model}/{args.checkpoint}",
        #     root=model_path)
        # get best
        checkpoint_file = wandb.restore(
            "best.pth.tar",
            run_path=f"multimodal-image-cls/{args.model}/{args.checkpoint}",
            root=model_path)
        # load state dict
        model, optimizer = utils.load_checkpoint(model, optimizer, args.device,
                                                 checkpoint_file.name)

    # skip training if just testing
    if not args.evaluate:
        if args.model == "maml":
            model = maml.training_run(args, model, optimizer, train_loader,
                                      val_loader, max_test_batches // 2)
        elif args.model == "fumi":
            model = fumi.training_run(args, model, optimizer, train_loader,
                                      val_loader, max_test_batches // 2)
        elif args.model == 'clip':
            model = clip.training_run(args,
                              model,
                              optimizer,
                              train_loader,
                              val_loader,
                              n_epochs=args.epochs)
        else:
            model = am3.training_run(args, model, optimizer, train_loader,
                                     val_loader, max_test_batches // 2)

    # test
    if args.model in ["maml", "fumi"]:
        if args.model == "maml":
            test_loss, test_acc = maml.test_loop(args, model, test_loader,
                                                 max_test_batches)
        else:
            test_loss, test_acc, _, _ = fumi.test_loop(args, model,
                                                       test_loader,
                                                       max_test_batches)
        print(f"\n TEST: \ntest loss: {test_loss}, test acc: {test_acc}")

        wandb.log({"test/acc": test_acc, "test/loss": test_loss})
    elif args.model == 'clip':
        test_acc = clip.evaluate(args, model, test_loader)
        wandb.log({'test/acc': test_acc})
    else:
        test_loss, test_acc, test_f1, test_prec, test_rec, test_avg_lamda, test_preds, test_true, query_idx, support_idx, support_lamda = am3.test_loop(
            args, model, test_loader, max_test_batches)
        print(
            f"\n TEST: \ntest loss: {test_loss}, test acc: {test_acc},\ntest f1: {test_f1}, test prec: {test_prec}, test rec: {test_rec}, test avg lamda: {test_avg_lamda}"
        )

        wandb.log({
            "test/acc": test_acc,
            "test/f1": test_f1,
            "test/prec": test_prec,
            "test/rec": test_rec,
            "test/loss": test_loss,
            "test/avg_lamda": test_avg_lamda
        })

        # save results
        df = pd.DataFrame({
            "support_idx": support_idx,
            "support_lamda": support_lamda,
            "query_idx": query_idx,
            "query_preds": test_preds,
            "query_targets": test_true
        })
        df.to_csv(path_or_buf=f"{results_path}/run_{wandb.run.name}.csv")

    wandb.finish()


def parse_args():
    parser = utils.parser()
    args = parser.parse_args(sys.argv[1:])

    args.device = torch.device("cuda") if (not args.disable_cuda) and \
        torch.cuda.is_available() else torch.device("cpu")
    print(f"running on device {args.device}")

    return args


if __name__ == "__main__":
    # read cmd line arguments
    args = parse_args()
    # run experiment
    main(args)

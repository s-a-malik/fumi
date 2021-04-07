"""AM3 in Pytorch.
"""

import os
import argparse
from tqdm import tqdm

import torch
from torchmeta.utils.data import BatchMetaDataLoader

from model import AM3
import utils

def init_model(args):
    """Initialise model
    """
    model_params = args.embed_dim

    # TODO model args 
    model = AM3()

def init_optim(args, model):
    """Initialise optimizer and loss functions
    """

    if args.optim == "adam":
        optimizer = torch.optim.Adam(params=model.parameters(),
                                     lr=args.lr,
                                     weight_decay=arg.weight_decay)
    else:
        raise NotImplementedError()

    return optimizer


def main(args):
    
    # load datasets
    train, val, test = get_dataset(args.dataset, args.data_dir, args.num_ways)

    train_loader = BatchMetaDataLoader(dataset,
                                        batch_size=args.batch_size,
                                        shuffle=True,
                                        num_workers=args.num_workers)

    # initialise model
    model = init_model(args)

    # load previous state
    if args.checkpoint:
        model, optimizer = utils.load_checkpoint(args.checkpoint)

    # skip training if just testing
    if not args.evaluate:
        # run training loop
        model.to(args.device)
        model.train()

        # Training loop
        with tqdm(train_dataloader, total=args.epochs) as pbar:
            for batch_idx, batch in enumerate(pbar):

                # TODO sort this to get specific to AM3 
                train_inputs, train_targets = batch['train']
                train_inputs = train_inputs.to(device=args.device)
                train_targets = train_targets.to(device=args.device)
                train_embeddings = model(train_inputs)

                test_inputs, test_targets = batch['test']
                test_inputs = test_inputs.to(device=args.device)
                test_targets = test_targets.to(device=args.device)
                test_embeddings = model(test_inputs)

                prototypes = model.get_prototypes(train_embeddings, train_targets,
                                            train.num_classes_per_task)
                
                loss = prototypical_loss(prototypes, test_embeddings, test_targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    accuracy = get_accuracy(prototypes, test_embeddings, test_targets)
                    pbar.set_postfix(accuracy='{0:.4f}'.format(accuracy.item()))

                if batch_idx >= args.epochs:
                    break
        # save model
        
    # load saved model

    # test
    test_loop(model, test)


def test_loop(model, test_dataset):
    """Evaluate model on test set
    """
    raise NotImplementedError()


def parse_args():
    """experiment arguments
    """

    parser = argparse.ArgumentParser(description="AM3")

    # data config
    parser.add_argument("--dataset",
                        type=str,
                        default="CUB",
                        help="Dataset to use")
    parser.add_argument("--data_dir",
                        type=str,
                        default="./data",
                        help="Directory to use for data")
    parser.add_argument("--checkpoint",
                        type=str,
                        default="",
                        help="path to pretrained model")
    parser.add_argument("--log_dir",
                        type=str,
                        default="./runs",
                        help="Directory to use for logs and checkpoints")               

    # optimizer config
    parser.add_argument("--epochs",
                        type=int,
                        default=50,
                        help="Number of epochs (meta-learning steps)")
    parser.add_argument("--optim",
                        type=str,
                        default="adam",
                        help="optimiser")                       
    parser.add_argument("--lr",
                        type=float,
                        default=1e-4,
                        help="learning rate")
    parser.add_argument("--batch_size",
                        type=int,
                        default=16,
                        help="number of tasks in mini-batch")
    
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
                        default=5,
                        help="Number of examples per class (k-shot)")
    parser.add_argument("--num_ways_test",
                        type=int,
                        default=5,
                        help="Number of classes per task (N-way)")
    parser.add_argument("--augment",
                        action="store_true"
                        help="augment data with image transformations")
    parser.add_argument("--num_workers",
                        type=int,
                        default=0,
                        help="Number of workers for dataloader")

    # model config
    parser.add_argument("--embed_dim",
                        type=int,
                        default=64,
                        help="Dimension of latent space")
    parser.add_argument("--lamda",
                        type=int,
                        default=0.8,
                        help="Balance between text and image prototypes")
    parser.add_argument("--feature_extractor",
                        type=str,
                        default="resnet",
                        help="Type of vision feature extractor (resnet, precomputed)")
    parser.add_argument("--text_emb",
                        type=str,
                        default="glove",
                        help="Type of text embedding (glove, BERT)")
    parser.add_argument("--text_type",
                        type=str,
                        default="label",
                        help="What to use for text embedding (label or description)")
    parser.add_argument("--weight_decay",
                        type=float,
                        default=0.0005,
                        help="L2 regulariser")
   parser.add_argument("--dropout",
                        type=float,
                        default=0.0,
                        help="dropout rate")

    # run config
    parser.add_argument("--evaluate",
                        action="store_true"
                        help="skip training")
    parser.add_argument("--num_cases_test",
                        type=int,
                        default=50000,
                        help="Number of few-shot cases to compute test accuracy")
    parser.add_argument("--disable_cuda",
                        action="store_true"
                        help="don't use GPU")

    args = parser.parse_args(sys.argv[1:])

    args.device = torch.device("cuda") if (not args.disable_cuda) and \
        torch.cuda.is_available() else torch.device("cpu")

    return args


if __name__ == "__main__":

    # read cmd line arguments
    args = parse_args()
    
    main(args)

import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from torchmeta.modules import MetaModule, MetaSequential, MetaLinear
from torchmeta.utils.gradient_based import gradient_update_parameters

import utils


class PureImageNetwork(MetaModule):
    def __init__(self, im_embed_dim=2048, n_way=5, hidden=64):
        super(PureImageNetwork, self).__init__()
        self.im_embed_dim = im_embed_dim
        self.n_way = n_way
        self.hidden = hidden

        self.net = MetaSequential(
            MetaLinear(im_embed_dim, hidden),
            nn.ReLU(),
            MetaLinear(hidden, n_way)
        )

    def forward(self, inputs, params=None):
      logits = self.net(inputs, params=params)
      return logits


def training_run(args, model, optimizer, train_loader, val_loader, max_test_batches):
    """
    MAML training loop

    Returns:
    - model (MetaModule): Trained model
    """

    best_loss, best_acc = test_loop(args, model, val_loader, max_test_batches)
    print(f"\ninitial loss: {best_loss}, acc: {best_acc}")
    best_batch_idx = 0

    try:
        # Training loop
        for batch_idx, batch in enumerate(train_loader):
            train_loss, train_acc, train_lamda = evaluate(
                model=model,
                batch=batch,
                optimizer=optimizer,
                device=args.device,
                step_size=args.step_size,
                first_order=args.first_order,
                task="train")

            wandb.log({"train/acc": train_acc,
                       "train/loss": train_loss,
                       "num_episodes": (batch_idx+1)*args.batch_size}, step=batch_idx)

            # Eval on validation set periodically
            if batch_idx % args.eval_freq == 0:
                val_loss, val_acc = test_loop(
                    args, model, val_loader, max_test_batches)
                is_best = val_loss < best_loss
                if is_best:
                    best_loss = val_loss
                    best_batch_idx = batch_idx
                wandb.log({"val/acc": val_acc,
                           "val/loss": val_loss}, step=batch_idx)

                checkpoint_dict = {
                    "batch_idx": batch_idx,
                    "state_dict": model.state_dict(),
                    "best_loss": best_loss,
                    "optimizer": optimizer.state_dict(),
                    "args": vars(args)
                }
                utils.save_checkpoint(checkpoint_dict, is_best)

                print(f"\nBatch {batch_idx+1}/{args.epochs}: \ntrain/loss: {train_loss}, train/acc: {train_acc}"
                      f"\nval/loss: {val_loss}, val/acc: {val_acc}")

            # break after max iters or early stopping
            if (batch_idx > args.epochs - 1) or (batch_idx - best_batch_idx > args.patience):
                break
    except KeyboardInterrupt:
        pass

    return model


def test_loop(args, model, test_loader, max_num_batches):
    """
    Evaluate model on val/test set.
    Returns:
    - avg_test_acc (float): average test accuracy per task
    - avg_test_loss (float): average test loss per task
    """

    avg_test_acc = utils.AverageMeter()
    avg_test_loss = utils.AverageMeter()
    for batch_idx, batch in enumerate(tqdm(test_loader, total=max_num_batches, position=0, leave=True)):
        with torch.no_grad():
            test_loss, test_acc, preds, trues, idx, lamda = evaluate(
                model=model,
                batch=batch,
                optimizer=None,
                device=args.device,
                step_size=args.step_size,
                first_order=args.first_order,
                task="test")
        avg_test_acc.update(test_acc)
        avg_test_loss.update(test_loss)
        if batch_idx > max_num_batches - 1:
            break
    return avg_test_loss.avg, avg_test_acc.avg


def evaluate(model, batch, optimizer, device, step_size, first_order, task="train"):
    """
    Evaluate batch on model

    Returns:
    - outer_loss: outer loop loss
    - acc: accuracy on query set
    """
    # Require model in train mode for inner loop update
    model.train()

    # Support set
    train_inputs, train_targets = batch['train']
    train_inputs = train_inputs[3].to(device=device)
    train_targets = train_targets.to(device=device)

    # Query set
    test_inputs, test_targets = batch['test']
    test_inputs = test_inputs[3].to(device=device)
    test_targets = test_targets.to(device=device)

    outer_loss = torch.tensor(0., device=device)
    accuracy = torch.tensor(0., device=device)
    for task_idx, (train_input, train_target, test_input,
                   test_target) in enumerate(zip(train_inputs, train_targets,
                                                 test_inputs, test_targets)):
        train_logit = model(train_input)
        inner_loss = F.cross_entropy(train_logit, train_target)

        model.zero_grad()
        params = gradient_update_parameters(model,
                                            inner_loss,
                                            step_size=step_size,
                                            first_order=first_order)

        test_logit = model(test_input, params=params)
        outer_loss += F.cross_entropy(test_logit, test_target)

        with torch.no_grad():
            accuracy += get_accuracy(test_logit, test_target)

    outer_loss.div_(train_inputs.shape[0])
    accuracy.div_(train_inputs.shape[0])

    if task == "train":
        optimizer.zero_grad()
        outer_loss.backward()
        optimizer.step()

    return outer_loss.detach().cpu().numpy(), accuracy.detach().cpu().numpy()


def get_accuracy(logits, targets):
    _, predictions = torch.max(logits, dim=-1)
    return torch.mean(predictions.eq(targets).float())

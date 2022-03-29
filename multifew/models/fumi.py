import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmeta.modules import (MetaSequential, MetaLinear)
from torchmeta.utils.gradient_based import gradient_update_parameters

import os
from tqdm import tqdm
from collections import OrderedDict

from utils.average_meter import AverageMeter
from utils import utils as utils
from .common import WordEmbedding, RnnHid, RNN


class FUMI(nn.Module):
    def __init__(self,
                 n_way=5,
                 im_emb_dim=2048,
                 im_hid_dim=[64],
                 text_encoder="BERT",
                 text_emb_dim=300,
                 text_hid_dim=1024,
                 dropout_rate=0.0,
                 dictionary=None,
                 pooling_strat="mean",
                 init_all_layers=False,
                 norm_hypernet=True,
                 fine_tune=False):
        super(FUMI, self).__init__()
        self.n_way = n_way
        self.im_emb_dim = im_emb_dim
        self.im_hid_dim = im_hid_dim
        self.text_encoder_type = text_encoder
        self.text_emb_dim = text_emb_dim  # only applicable if precomputed or RNN hid dim
        self.text_hid_dim = text_hid_dim
        self.dropout_rate = dropout_rate
        self.dictionary = dictionary  # for word embeddings
        self.pooling_strat = pooling_strat
        self.norm_hypernet = norm_hypernet
        self.fine_tune = fine_tune

        if self.text_encoder_type == "BERT" or self.text_encoder_type == "precomputed":
            # BERT embeddings precomputed in dataloader
            self.text_encoder = nn.Identity()
        elif self.text_encoder_type == "w2v" or self.text_encoder_type == "glove":
            # load pretrained word embeddings as weights
            self.text_encoder = WordEmbedding(self.text_encoder_type,
                                              self.pooling_strat,
                                              self.dictionary)
            self.text_emb_dim = self.text_encoder.embedding_dim
        elif self.text_encoder_type == "rand":
            self.text_encoder = nn.Linear(self.text_emb_dim, self.text_emb_dim)
        elif self.text_encoder_type == "RNN":
            self.text_encoder = RNN("glove", self.pooling_strat, self.dictionary, self.text_emb_dim)
        elif self.text_encoder_type == "RNNhid":
            self.text_encoder = RnnHid("glove", self.pooling_strat, self.dictionary, self.text_emb_dim)
        else:
            raise NameError(f"{text_encoder} not allowed as text encoder")

        if not self.fine_tune:
            for param in self.text_encoder.parameters():
                param.requires_grad = False

        # Text embedding to image parameters
        hyper_net_layers = [
            nn.Linear(self.text_emb_dim, self.text_hid_dim),
            nn.ReLU()
        ]
        self.init_all_layers = init_all_layers
        if not self.init_all_layers:
            hyper_net_layers.append(nn.Linear(
                self.text_hid_dim,
                self.im_hid_dim[-1]  # Weights
                + 1)  # Biases
            )
            # Preceding layers
            im_net_layers = OrderedDict()
            if len(self.im_hid_dim) > 0:
                im_net_layers['linear0'] = MetaLinear(self.im_emb_dim, self.im_hid_dim[0])
                im_net_layers['relu0'] = nn.ReLU()
                if self.dropout_rate > 0:
                    im_net_layers['dropout0'] = nn.Dropout(self.dropout_rate)
                for i in range(len(self.im_hid_dim)-1):
                    im_net_layers['linear'+str(i+1)] = MetaLinear(self.im_hid_dim[i], self.im_hid_dim[i+1])
                    im_net_layers['relu'+str(i+1)] = nn.ReLU()
                    if self.dropout_rate > 0:
                        im_net_layers['dropout'+str(i+1)] = nn.Dropout(self.dropout_rate)
                im_net_layers['linear_final'] = MetaLinear(self.im_hid_dim[-1], self.n_way)
            else:
                im_net_layers['linear_final'] = MetaLinear(self.im_emb_dim, self.n_way)
            self.im_net = MetaSequential(im_net_layers)
        else:
            raise NotImplementedError("Entire model hypernet initialisation removed")

        if self.norm_hypernet:
            hyper_net_layers.append(nn.Tanh())

        self.hyper_net = nn.Sequential(*hyper_net_layers)

    def forward(self, text_embed):
        """
        Hyper-network forward pass (text -> image params)
        """
        return self.hyper_net(text_embed)

    def evaluate(self, args, batch, optimizer, task="train"):
        """
        Evaluate batch on model

        Returns:
        - outer_loss: outer loop loss
        - acc: accuracy on query set
        """
        if task == "train":
            self.train()
            self.zero_grad()
        else:
            self.eval()

        # Support set
        train_inputs, train_targets = batch['train']
        train_inputs = [x.to(args.device) for x in train_inputs]
        # train_inputs = train_inputs[3].to(device=args.device)
        train_targets = train_targets.to(device=args.device)

        # Query set
        test_inputs, test_targets = batch['test']
        test_inputs = [x.to(args.device) for x in test_inputs]
        # test_inputs = test_inputs[3].to(device=args.device)
        test_targets = test_targets.to(device=args.device)
        test_preds = torch.zeros(test_targets.shape).to(device=args.device)

        # Unpack input
        _, train_texts, train_imss = train_inputs
        _, test_texts, test_imss = test_inputs

        outer_loss = torch.tensor(0., device=args.device)
        accuracy = torch.tensor(0., device=args.device)
        for task_idx, (train_target, test_target) in enumerate(
                zip(train_targets, test_targets)):
            n_steps = 0
            if task == "train":
                n_steps = args.num_train_adapt_steps
            else:
                n_steps = args.num_test_adapt_steps

            im_params = self.get_im_params(train_texts[task_idx],
                                           train_target, args.device)

            # Initialise image network, see https://discuss.pytorch.org/t/autograd-isnt-functioning-when-networkss-parameters-are-taken-from-other-networks/27424/6
            self.im_net.linear_final.weight.copy_(im_params[:, :-1])
            self.im_net.linear_final.bias.copy_(im_params[:, -1])
            params = self.im_net.parameters()
            for _ in range(n_steps):
                train_logit = self.im_net(train_imss[task_idx], params=params)
                inner_loss = F.cross_entropy(train_logit, train_target)
                self.im_net.zero_grad()
                params = gradient_update_parameters(model,
                                                    inner_loss,
                                                    params=params,
                                                    step_size=args.step_size,
                                                    first_order=args.first_order)

            test_logit = self.im_forward(test_imss[task_idx], params=params)
            _, test_preds[task_idx] = test_logit.max(dim=-1)

            outer_loss += F.cross_entropy(test_logit, test_target)

            with torch.no_grad():
                accuracy += get_accuracy(test_logit, test_target)

        outer_loss.div_(train_imss.shape[0])
        accuracy.div_(train_imss.shape[0])

        if task == "train":
            optimizer.zero_grad()
            outer_loss.backward()
            optimizer.step()

        return outer_loss.detach().cpu().numpy(), accuracy.detach().cpu(
        ).numpy(), test_preds, test_targets

    def get_im_params(self, text, targets, device, attn_mask=None):
        NK, seq_len = text.shape
        if self.text_encoder_type == "rand":
            # Get a random tensor as the encoding
            text_encoding = 2 * torch.rand(NK, self.text_emb_dim) - 1
        else:
            text_encoding = self.text_encoder(text.unsqueeze(0)).squeeze()

        # Transform to per-class descriptions
        class_text_enc = torch.empty(self.n_way, self.text_emb_dim).to(device)
        for i in range(self.n_way):
            class_text_enc[i] = text_encoding[(targets == i).nonzero(
                as_tuple=True)[0][0]]

        return self(class_text_enc)


def training_run(args, model, optimizer, train_loader, val_loader,
                 max_test_batches):
    """
    FUMI training loop
    """

    best_loss, best_acc, _, _ = test_loop(args, model, val_loader, max_test_batches)
    print(f"\ninitial loss: {best_loss}, acc: {best_acc}")
    best_batch_idx = 0

    # check if scheduled
    if type(optimizer) == tuple:
        opt, scheduler = optimizer
    else:
        opt = optimizer
        scheduler = None

    try:
        # Training loop
        t = tqdm(total=args.eval_freq, leave=True, position=0, desc='Train')
        t.refresh()
        for batch_idx, batch in enumerate(train_loader):
            train_loss, train_acc, _, _ = model.evaluate(args=args,
                                                   batch=batch,
                                                   optimizer=opt,
                                                   task="train")

            t.update()
            wandb.log(
                {
                    "train/acc": train_acc,
                    "train/loss": train_loss,
                    "num_episodes": (batch_idx + 1) * args.batch_size
                },
                step=batch_idx)

            # Eval on validation set periodically
            if batch_idx % args.eval_freq == 0 and batch_idx != 0:
                t.close()
                val_loss, val_acc, _, _ = test_loop(args, model, val_loader,
                                              max_test_batches)
                is_best = val_loss < best_loss
                if is_best:
                    best_loss = val_loss
                    best_batch_idx = batch_idx
                wandb.log({
                    "val/acc": val_acc,
                    "val/loss": val_loss
                },
                          step=batch_idx)

                checkpoint_dict = {
                    "batch_idx": batch_idx,
                    "state_dict": model.state_dict(),
                    "best_loss": best_loss,
                    "optimizer": opt.state_dict(),
                    "args": vars(args)
                }
                utils.save_checkpoint(checkpoint_dict, is_best)

                print(
                    f"\nBatch {batch_idx+1}/{args.epochs}: \ntrain/loss: {train_loss}, train/acc: {train_acc}"
                    f"\nval/loss: {val_loss}, val/acc: {val_acc}")

                t = tqdm(total=args.eval_freq, leave=True, position=0, desc='Train batch')
                t.refresh()

            # break after max iters or early stopping
            if (batch_idx > args.epochs - 1) or (
                    args.patience > 0
                    and batch_idx - best_batch_idx > args.patience):
                break
    except KeyboardInterrupt:
        pass

    # load best model    
    best_file = os.path.join(wandb.run.dir, "best.pth.tar")
    model, _ = utils.load_checkpoint(model, opt, args.device, best_file)

    return model


def test_loop(args, model, test_loader, max_num_batches):
    """
    Evaluate model on val/test set.
    Returns:
    - avg_test_acc (float): average test accuracy per task
    - avg_test_loss (float): average test loss per task
    """

    avg_test_acc = AverageMeter()
    avg_test_loss = AverageMeter()
    test_preds = []
    test_targets = []
    for batch_idx, batch in enumerate(
            tqdm(test_loader, total=max_num_batches, position=0, leave=True, desc='Test')):
        test_loss, test_acc, preds, target = model.evaluate(args=args,
                                                            batch=batch,
                                                            optimizer=None,
                                                            task="test")
        avg_test_acc.update(test_acc)
        avg_test_loss.update(test_loss)
        test_preds.append(preds)
        test_targets.append(target)
        if batch_idx > max_num_batches - 1:
            break
    return avg_test_loss.avg, avg_test_acc.avg, test_preds, test_targets


def get_accuracy(logits, targets):
    _, predictions = torch.max(logits, dim=-1)
    return torch.mean(predictions.eq(targets).float())
